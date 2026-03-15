"""
QA Service for IRIS
Handles retrieval-based question answering using UltraRAG and Llama3B.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import requests


logger = logging.getLogger(__name__)


class QAService:
    """Service for question answering using UltraRAG and vLLM."""

    def __init__(
        self,
        ultrarag_path: str,
        embedding_model: str,
        reranker_model: str,
        generation_model: str,
        vllm_base_url: str = "http://127.0.0.1:65504/v1",
        vllm_port: str = "65504",
        served_model_name: str = "llama3-3b-instruct",
        system_prompt: str = "你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业",
        index_backend: str = "faiss"
    ):
        """
        Initialize QA service.

        Args:
            ultrarag_path: Path to UltraRAG project root
            embedding_model: Path to embedding model
            reranker_model: Path to reranker model
            generation_model: Path to generation model (Llama3B)
            vllm_base_url: Base URL for vLLM service
            vllm_port: Port for vLLM service
            served_model_name: Model name served by vLLM
            system_prompt: System prompt for generation
            index_backend: Index backend type (faiss or milvus)
        """
        self.ultrarag_path = Path(ultrarag_path).absolute()
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.generation_model = generation_model
        self.vllm_base_url = vllm_base_url
        self.vllm_port = vllm_port
        self.served_model_name = served_model_name
        self.system_prompt = system_prompt
        self.index_backend = index_backend

        # Paths to UltraRAG components
        self.pipeline_yaml = self.ultrarag_path / "pipelines" / "online_rag_qa_batch.yaml"
        self.parameter_template = self.ultrarag_path / "parameter" / "online_rag_qa_batch_parameter.yaml.template"

        # vLLM process management
        self.vllm_process = None
        self.vllm_auto_started = False

        logger.info(f"QA service initialized with LLM: {generation_model}")

    def query_knowledge_base(
        self,
        question: str,
        index_path: Path,
        chunks_path: Path,
        top_k: int = 5
    ) -> Optional[str]:
        """
        Query the knowledge base with a single question.

        Args:
            question: The question to ask
            index_path: Path to the index file
            chunks_path: Path to the chunks file
            top_k: Number of top results to retrieve

        Returns:
            Answer string, or None if failed
        """
        logger.info(f"Querying knowledge base: {question}")

        # Create temporary questions file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({"id": 0, "question": question}, f, ensure_ascii=False)
            f.write('\n')
            questions_file = f.name

        try:
            result = self._run_qa_batch(
                questions_file=questions_file,
                chunks_path=chunks_path,
                index_path=index_path
            )

            if result and len(result) > 0:
                return result[0].get("answer")

            return None

        finally:
            # Clean up temporary file
            Path(questions_file).unlink(missing_ok=True)

    def query_knowledge_base_batch(
        self,
        questions: List[str],
        index_path: Path,
        chunks_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base with multiple questions.

        Args:
            questions: List of questions to ask
            index_path: Path to the index file
            chunks_path: Path to the chunks file

        Returns:
            List of result dictionaries with 'question' and 'answer' keys
        """
        logger.info(f"Querying knowledge base with {len(questions)} questions")

        # Create temporary questions file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for idx, q in enumerate(questions):
                json.dump({"id": idx, "question": q}, f, ensure_ascii=False)
                f.write('\n')
            questions_file = f.name

        try:
            results = self._run_qa_batch(
                questions_file=questions_file,
                chunks_path=chunks_path,
                index_path=index_path
            )

            return results if results else []

        finally:
            # Clean up temporary file
            Path(questions_file).unlink(missing_ok=True)

    def _run_qa_batch(
        self,
        questions_file: str,
        chunks_path: Path,
        index_path: Path
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run QA batch processing using UltraRAG.

        Args:
            questions_file: Path to questions JSONL file
            chunks_path: Path to chunks.jsonl file
            index_path: Path to index file

        Returns:
            List of answers, or None if failed
        """
        # Create runtime parameter file
        runtime_param = self.ultrarag_path / "parameter" / "_runtime" / "online_rag_qa_batch_parameter.yaml"

        replacements = {
            "QUESTIONS_PATH": str(Path(questions_file).absolute()),
            "CHUNKS_PATH": str(chunks_path.absolute()),
            "EMBEDDING_MODEL_PATH": self.embedding_model,
            "RERANKER_MODEL_PATH": self.reranker_model,
            "GENERATION_MODEL_PATH": self.generation_model,
            "VLLM_BASE_URL": self.vllm_base_url,
            "SERVED_MODEL_NAME": self.served_model_name,
            "SYSTEM_PROMPT": self.system_prompt,
            "INDEX_BACKEND": self.index_backend,
            "INDEX_PATH": str(index_path.absolute()) if self.index_backend == "faiss" else "",
            "MILVUS_URI": "tcp://127.0.0.1:29901",
            "COLLECTION_NAME": "iris_papers",
        }

        self._create_runtime_parameter(
            self.parameter_template,
            runtime_param,
            replacements
        )

        # Also copy to the location where UltraRAG expects it
        target_param = self.ultrarag_path / "pipelines" / "parameter" / "online_rag_qa_batch_parameter.yaml"
        target_param.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(runtime_param, target_param)
        logger.info(f"Copied parameter file to: {target_param}")

        # Run UltraRAG pipeline
        success = self._run_ultrarag_pipeline(self.pipeline_yaml)

        if not success:
            logger.error("QA pipeline failed")
            return None

        # Extract answers from UltraRAG output
        return self._extract_answers(self.ultrarag_path / "output", questions_file)

    def _create_runtime_parameter(
        self,
        template_path: Path,
        output_path: Path,
        replacements: dict
    ):
        """Create a runtime parameter file from template."""
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        for key, value in replacements.items():
            placeholder = f"__{key}__"
            content = content.replace(placeholder, str(value))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Created runtime parameter: {output_path}")

    def _run_ultrarag_pipeline(self, pipeline_yaml: Path) -> bool:
        """Run UltraRAG run command."""
        logger.info(f"Running UltraRAG pipeline: {pipeline_yaml}")

        # Find ultrarag command (prefer venv)
        venv_ultrarag = self.ultrarag_path / ".venv" / "bin" / "ultrarag"
        venv_bin = self.ultrarag_path / ".venv" / "bin"

        if venv_ultrarag.exists():
            ultrarag_cmd = str(venv_ultrarag)
            logger.debug(f"Using venv ultrarag")
            env = os.environ.copy()
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            env["PYTHONPATH"] = str(self.ultrarag_path / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["VIRTUAL_ENV"] = str(self.ultrarag_path / ".venv")
        else:
            ultrarag_cmd = "ultrarag"
            logger.debug(f"Using system ultrarag")
            env = None

        try:
            result = subprocess.run(
                [ultrarag_cmd, "run", str(pipeline_yaml)],
                cwd=self.ultrarag_path,
                capture_output=True,
                text=True,
                check=False,
                env=env
            )

            if result.stdout:
                logger.debug(f"UltraRAG run stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"UltraRAG run stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"UltraRAG run failed with code {result.returncode}")
                return False

            logger.info("UltraRAG pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error running UltraRAG pipeline: {e}")
            return False

    def _extract_answers(
        self,
        output_dir: Path,
        questions_file: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract answers from UltraRAG output JSON file.

        Args:
            output_dir: UltraRAG output directory
            questions_file: Path to questions file (for question texts)

        Returns:
            List of answer dictionaries
        """
        # Find the most recent output file
        output_files = list(output_dir.glob("*online_rag_qa*.json"))
        if not output_files:
            logger.error(f"No output files found in {output_dir}")
            return None

        # Get the most recently modified file
        output_file = max(output_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"UltraRAG output file: {output_file}")

        # Read the entire output file
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Parse as JSON array
        try:
            steps_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse output file as JSON: {e}")
            return None

        # Find final answers from "custom.output_extract_from_boxed" step
        pred_ls = None
        q_ls = None

        for step_record in steps_data:
            if isinstance(step_record, dict):
                step_name = step_record.get("step", "")
                memory = step_record.get("memory", {})

                # Get questions from benchmark.get_data step
                if step_name == "benchmark.get_data":
                    q_ls = memory.get("memory_q_ls", [])

                # Get final answers from custom.output_extract_from_boxed step
                elif step_name == "custom.output_extract_from_boxed":
                    pred_ls = memory.get("memory_pred_ls", [])

        if pred_ls is None:
            logger.warning("No answers found in output")
            return None

        # Read questions to get question texts
        questions = []
        if Path(questions_file).exists():
            with open(questions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        questions.append(data.get("question", ""))

        # Build answer records
        answers = []
        for idx, answer in enumerate(pred_ls):
            answers.append({
                "id": idx,
                "question": questions[idx] if idx < len(questions) else "",
                "answer": answer
            })

        logger.info(f"Extracted {len(answers)} answers")
        return answers

    def start_vllm_service(
        self,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.7,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True
    ) -> bool:
        """
        Start vLLM service in background as a subprocess.

        Args:
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            tensor_parallel_size: Tensor parallel size
            enforce_eager: Whether to use eager execution

        Returns:
            True if vLLM was started successfully, False otherwise
        """
        if self.vllm_process is not None and self.vllm_process.poll() is None:
            logger.info("vLLM service is already running")
            return True

        logger.info(f"Starting vLLM service with model: {self.generation_model}")

        # Extract host and port from vllm_base_url
        # Expected format: http://127.0.0.1:65504/v1
        from urllib.parse import urlparse
        parsed = urlparse(self.vllm_base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or int(self.vllm_port)

        # Build vLLM command
        # Use the same venv that UltraRAG uses
        venv_python = self.ultrarag_path / ".venv" / "bin" / "python"

        cmd = [
            str(venv_python),
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.generation_model,
            "--trust-remote-code",
            "--host", host,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--served-model-name", self.served_model_name,
        ]

        if enforce_eager:
            cmd.append("--enforce-eager")

        logger.info(f"vLLM command: {' '.join(cmd)}")

        try:
            # Start vLLM in background with stdout/stderr redirected to devnull
            import io
            log_file = Path("logs") / "vllm_output.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log_f:
                self.vllm_process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True  # Detach from parent process
                )

            self.vllm_auto_started = True
            logger.info(f"vLLM process started with PID: {self.vllm_process.pid}")
            logger.info(f"vLLM output will be logged to: {log_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to start vLLM service: {e}")
            self.vllm_process = None
            self.vllm_auto_started = False
            return False

    def stop_vllm_service(self) -> None:
        """
        Stop the vLLM service if it was started by this QAService instance.
        """
        if self.vllm_process is None:
            logger.debug("No vLLM process to stop")
            return

        if self.vllm_process.poll() is not None:
            logger.debug("vLLM process already terminated")
            self.vllm_process = None
            self.vllm_auto_started = False
            return

        logger.info(f"Stopping vLLM service (PID: {self.vllm_process.pid})")

        try:
            # Try graceful shutdown first
            self.vllm_process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                self.vllm_process.wait(timeout=10)
                logger.info("vLLM service stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("vLLM service did not stop gracefully, forcing...")
                self.vllm_process.kill()
                self.vllm_process.wait()

        except Exception as e:
            logger.error(f"Error stopping vLLM service: {e}")

        self.vllm_process = None
        self.vllm_auto_started = False

    def __del__(self):
        """Cleanup when QAService instance is destroyed."""
        if self.vllm_auto_started:
            self.stop_vllm_service()

    def check_vllm_service(self, timeout: int = 30, auto_start: bool = False) -> bool:
        """
        Check if vLLM service is ready.

        Args:
            timeout: Timeout in seconds
            auto_start: If True and service is not available, start vLLM automatically

        Returns:
            True if service is ready, False otherwise
        """
        health_url = self.vllm_base_url.rstrip("/").replace("/v1", "") + "/v1/models"

        logger.info(f"Checking vLLM service at {health_url}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM service is ready")
                    return True
            except Exception as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)

        logger.error(f"vLLM service not ready after {timeout} seconds")

        # Auto-start vLLM if enabled
        if auto_start:
            logger.info("vLLM service not available, attempting to start...")
            if self.start_vllm_service():
                logger.info("vLLM service started, waiting for it to be ready...")
                # Wait for vLLM to be ready (give it more time for initial startup)
                startup_timeout = 120  # Give vLLM 2 minutes to fully start
                start_time = time.time()
                while time.time() - start_time < startup_timeout:
                    try:
                        response = requests.get(health_url, timeout=5)
                        if response.status_code == 200:
                            logger.info("vLLM service is now ready")
                            return True
                    except Exception as e:
                        logger.debug(f"Waiting for vLLM to be ready: {e}")

                    time.sleep(3)

                logger.error(f"vLLM service failed to start within {startup_timeout} seconds")
                return False

        return False
