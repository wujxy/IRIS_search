"""
Deploy Server for IRIS
Controls Milvus container and vLLM model services.
"""

import logging
import subprocess
import time
from pathlib import Path

import requests


logger = logging.getLogger(__name__)


class MilvusControl:
    """Control Milvus vector database startup and shutdown."""

    def __init__(self, config: dict):
        """
        Initialize Milvus control.

        Args:
            config: Configuration dictionary containing milvus settings
        """
        self.container_name = config["milvus"]["container_name"]
        self.data_dir = config["milvus"]["data_dir"]
        self.grpc_port = config["milvus"]["grpc_port"]
        self.http_port = config["milvus"]["http_port"]
        self.image = config["milvus"]["image"]

    def search(self) -> bool:
        """
        Check if Milvus container is running.

        Returns:
            True if container is running, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}"],
                capture_output=True,
                text=True
            )
            is_running = self.container_name in result.stdout
            logger.debug(f"Milvus container running: {is_running}")
            return is_running
        except Exception as e:
            logger.error(f"Error checking Milvus status: {e}")
            return False

    def _container_exists(self) -> bool:
        """
        Check if Milvus container exists (running or stopped).

        Returns:
            True if container exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}"],
                capture_output=True,
                text=True
            )
            is_running = self.container_name in result.stdout
            logger.debug(f"Milvus container exists: {is_running}")
            return is_running
        except Exception as e:
            logger.error(f"Error checking Milvus status: {e}")
            return False

    def start(self) -> bool:
        """
        Start Milvus container using Docker.
        If container already exists, use 'docker start'; otherwise create with 'docker run'.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting Milvus container: {self.container_name}")

        # Create data directory
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Check if container already exists
        container_exists = self._container_exists()

        if container_exists:
            # Container exists, start it
            logger.info(f"Container exists, starting existing container")
            cmd = ["docker", "start", self.container_name]
        else:
            # Container doesn't exist, create new one
            logger.warn(f"Container does not exist, Creating new container")
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--restart", "unless-stopped",
                "--security-opt", "seccomp:unconfined",
                "-e", "DEPLOY_MODE=STANDALONE",
                "-e", "ETCD_USE_EMBED=true",
                "-e", "COMMON_STORAGETYPE=local",
                "-v", f"{self.data_dir}:/var/lib/milvus",
                "-p", f"{self.grpc_port}:19530",
                "-p", f"{self.http_port}:9091",
                "--health-cmd", "curl -f http://localhost:9091/healthz",
                "--health-interval", "30s",
                "--health-start-period", "60s",
                "--health-timeout", "10s",
                "--health-retries", "3",
                self.image,
                "milvus", "run", "standalone"
            ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Milvus container started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Milvus: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop Milvus container.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Stopping Milvus container: {self.container_name}")

        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                check=True,
                capture_output=True
            )
            logger.info(f"Milvus container stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Milvus: {e}")
            return False


class VllmControl:
    """Control vLLM model startup and shutdown."""

    def __init__(self, config: dict, model_type: str):
        """
        Initialize vLLM control.

        Args:
            config: Configuration dictionary
            model_type: 'index' or 'qa'
        """
        self.model_type = model_type

        if model_type == "index":
            self.model_name = config["models"]["vllm"]["index"]["model_name"]
            self.host = config["models"]["vllm"]["index"]["host"]
            self.port = config["models"]["vllm"]["index"]["port"]
            self.base_url = config["models"]["vllm"]["index"]["base_url"]
            self.served_model_name = config["models"]["vllm"]["index"]["served_model_name"]
            self.max_model_len = config["models"]["vllm"]["index"]["max_model_len"]
            self.gpu_memory_utilization = config["models"]["vllm"]["index"]["gpu_memory_utilization"]
            self.tensor_parallel_size = 1
        else:  # qa
            self.model_name = config["models"]["vllm"]["served_model_name"]
            self.host = config["models"]["vllm"]["host"]
            self.port = config["models"]["vllm"]["port"]
            self.base_url = config["models"]["vllm"]["base_url"]
            self.served_model_name = config["models"]["vllm"]["served_model_name"]
            self.max_model_len = config["models"]["vllm"]["max_model_len"]
            self.gpu_memory_utilization = config["models"]["vllm"]["gpu_memory_utilization"]
            self.tensor_parallel_size = config["models"]["vllm"]["tensor_parallel_size"]

        self.enforce_eager = config["models"]["vllm"].get("enforce_eager", True)

        # Model path
        self.ultrarag_path = Path(config["ultrarag"]["ultrarag_path"])
        if model_type == "index":
            self.model_path = config["models"]["embedding_model_path"]
        else:
            self.model_path = config["models"]["llm_model_path"]

        # Process management
        self.process = None
        self.venv_python = self.ultrarag_path / ".venv" / "bin" / "python"

        logger.info(f"VllmControl initialized for {model_type} model")

    def search(self, timeout: int = 30) -> bool:
        """
        Check if vLLM service is available.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if service is ready, False otherwise
        """
        health_url = self.base_url.rstrip("/").replace("/v1", "") + "/v1/models"

        logger.info(f"Checking vLLM {self.model_type} service at {health_url}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM {self.model_type} service is ready")
                    return True
            except Exception as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)

        logger.warn(f"vLLM {self.model_type} service not ready after {timeout} seconds")
        return False

    def start(self) -> bool:
        """
        Start vLLM model as a subprocess.

        Returns:
            True if successful, False otherwise
        """
        if self.process is not None and self.process.poll() is None:
            logger.info(f"vLLM {self.model_type} is already running")
            return True

        logger.info(f"Starting vLLM {self.model_type} with model: {self.model_path}")

        cmd = [
            str(self.venv_python),
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--trust-remote-code",
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--served-model-name", self.served_model_name,
        ]

        if self.enforce_eager:
            cmd.append("--enforce-eager")

        logger.info(f"vLLM command: {' '.join(cmd)}")

        try:
            log_file = Path("logs") / f"vllm_{self.model_type}_output.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log_f:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )

            logger.info(f"vLLM {self.model_type} started with PID: {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start vLLM {self.model_type}: {e}")
            self.process = None
            return False

    def stop(self) -> bool:
        """
        Stop vLLM model.

        Returns:
            True if successful, False otherwise
        """
        if self.process is None:
            logger.debug(f"No vLLM {self.model_type} process to stop")
            return True

        if self.process.poll() is not None:
            logger.debug(f"vLLM {self.model_type} process already terminated")
            self.process = None
            return True

        logger.info(f"Stopping vLLM {self.model_type} (PID: {self.process.pid})")

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info(f"vLLM {self.model_type} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"vLLM {self.model_type} did not stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
        except Exception as e:
            logger.error(f"Error stopping vLLM {self.model_type}: {e}")

        self.process = None
        return True


class DeployService:
    """Deployment server controller, unified management of Milvus and vLLM."""

    def __init__(self, config: dict):
        """
        Initialize deployment server.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.milvus_control = MilvusControl(config)
        self.index_vllm = VllmControl(config, "index")
        self.qa_vllm = VllmControl(config, "qa")
        logger.info("DeployServer initialized")

    def start_infrastructure(self) -> bool:
        """
        Start infrastructure: Milvus + vLLM (index model and QA model simultaneously).

        Returns:
            True if successful, False otherwise
        """
        # Check and start Milvus
        if not self.milvus_control.search():
            if not self.milvus_control.start():
                logger.error("Failed to start Milvus")
                return False

        # Wait for Milvus to be ready
        logger.info("Waiting for Milvus to be ready...")
        time.sleep(10)

        # Start index model and QA model simultaneously
        logger.info("Starting vLLM models...")

        if not self.index_vllm.search(timeout=5):
            if not self.index_vllm.start():
                logger.error("Failed to start index vLLM")
                return False

        if not self.qa_vllm.search(timeout=5):
            if not self.qa_vllm.start():
                logger.error("Failed to start QA vLLM")
                return False

        logger.info("Infrastructure started successfully")
        return True

    def stop_infrastructure(self) -> bool:
        """
        Stop infrastructure: vLLM (index model and QA model simultaneously) + Milvus.

        Returns:
            True if successful, False otherwise
        """
        # Stop all vLLM simultaneously
        logger.info("Stopping vLLM models...")
        self.index_vllm.stop()
        self.qa_vllm.stop()

        # Wait for vLLM to stop
        time.sleep(3)

        # Stop Milvus
        if self.milvus_control.search():
            self.milvus_control.stop()

        logger.info("Infrastructure stopped successfully")
        return True
