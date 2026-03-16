"""
QA Service for IRIS
Handles retrieval-based question answering using UltraRAG and Llama3B.
"""

import json
import logging
import os
import shutil
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
        index_vllm_base_url: str = "http://127.0.0.1:65503/v1",
        served_model_name: str = "llama3-3b-instruct",
        system_prompt: str = "你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业"
    ):
        """
        Initialize QA service.

        Args:
            ultrarag_path: Path to UltraRAG project root
            embedding_model: Path to embedding model
            reranker_model: Path to reranker model
            generation_model: Path to generation model (Llama3B)
            vllm_base_url: Base URL for QA vLLM service
            index_vllm_base_url: Base URL for index vLLM service
            served_model_name: Model name served by vLLM
            system_prompt: System prompt for generation
        """
        self.ultrarag_path = Path(ultrarag_path).absolute()
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.generation_model = generation_model
        self.vllm_base_url = vllm_base_url
        self.index_vllm_base_url = index_vllm_base_url
        self.served_model_name = served_model_name
        self.system_prompt = system_prompt

        # Paths to UltraRAG components
        self.pipeline_yaml = self.ultrarag_path / "pipelines" / "online_rag_qa_batch.yaml"
        # Template from IRIS project (for independence from UltraRAG)
        script_dir = Path(__file__).parent.parent.absolute()
        self.parameter_template = script_dir / "configs" / "ultrarag" / "templates" / "online_rag_qa_parameter.yaml.template"

        logger.info(f"QA service initialized with LLM: {generation_model}")

    def query_knowledge_base(
        self,
        question: str,
        chunks_path: Path,
        top_k: int = 5
    ) -> Optional[str]:
        """
        Query knowledge base with a single question.

        Args:
            question: The question to ask
            chunks_path: Path to chunks file
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
        chunks_path: Path,
        collection_name: str = "iris_papers"
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge base with multiple questions.

        Args:
            questions: List of questions to ask
            chunks_path: Path to chunks.jsonl file
            collection_name: Milvus collection name

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
                collection_name=collection_name
            )

            return results if results else []

        finally:
            # Clean up temporary file
            Path(questions_file).unlink(missing_ok=True)

    def _run_qa_batch(
        self,
        questions_file: str,
        chunks_path: Path,
        collection_name: str = "iris_papers"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run QA batch processing using UltraRAG.

        Args:
            questions_file: Path to questions JSONL file
            chunks_path: Path to chunks.jsonl file
            collection_name: Milvus collection name

        Returns:
            List of answers, or None if failed
        """
        # Create runtime parameter file
        runtime_param = self.ultrarag_path / "parameter" / "_runtime" / "online_rag_qa_parameter.yaml"

        replacements = {
            "QUESTIONS_PATH": str(Path(questions_file).absolute()),
            "CHUNKS_PATH": str(chunks_path.absolute()),
            "EMBEDDING_MODEL_PATH": self.embedding_model,
            "RERANKER_MODEL_PATH": self.reranker_model,
            "GENERATION_MODEL_PATH": self.generation_model,
            "GENERATION_BASE_URL": self.vllm_base_url,
            "SERVED_MODEL_NAME": self.served_model_name,
            "SYSTEM_PROMPT": self.system_prompt,
            "INDEX_PATH": "",
            "MILVUS_URI": "tcp://127.0.0.1:29901",
            "COLLECTION_NAME": collection_name,
            "INDEX_BASE_URL": self.index_vllm_base_url,
            "INDEX_MODEL_NAME": "qwen3-embedding-0.6b",
        }

        self._create_runtime_parameter(
            self.parameter_template,
            runtime_param,
            replacements
        )

        # Also copy to location where UltraRAG expects it
        target_param = self.ultrarag_path / "pipelines" / "parameter" / "online_rag_qa_parameter.yaml"
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
        # Find most recent output file
        output_files = list(output_dir.glob("*online_rag_qa*.json"))
        if not output_files:
            logger.error(f"No output files found in {output_dir}")
            return None

        # Get most recently modified file
        output_file = max(output_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"UltraRAG output file: {output_file}")

        # Read entire output file
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
