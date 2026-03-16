"""
Index Service for IRIS
Handles UltraRAG-based document chunking and vector indexing.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IndexService:
    """Service for chunking documents and building vector indexes using UltraRAG."""

    def __init__(
        self,
        ultrarag_path: str,
        embedding_model: str,
        collection_name: str = "iris_papers"
    ):
        """
        Initialize UltraRAG index service.

        Args:
            ultrarag_path: Path to UltraRAG project root
            embedding_model: Path to embedding model (Qwen0.3B)
            collection_name: Collection name for Milvus
        """
        self.ultrarag_path = Path(ultrarag_path).absolute()
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Paths to UltraRAG components
        # Use IRIS pipeline file directly (for independence from UltraRAG)
        script_dir = Path(__file__).parent.parent.absolute()
        self.pipeline_yaml = script_dir / "configs" / "ultrarag" / "pipelines" / "offline_build_index.yaml"
        # Template from IRIS project (for independence from UltraRAG)
        self.parameter_template = script_dir / "configs" / "ultrarag" / "templates" / "offline_build_index_parameter.yaml.template"

        logger.info(f"Index service initialized with embedding model: {embedding_model}")

    def chunk_and_index(
        self,
        pdf_dir: Path,
        output_dir: Path,
        collection_name: str = None,
        overwrite: bool = False
    ) -> bool:
        """
        Chunk PDFs and build vector index using UltraRAG.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save index files
            collection_name: Milvus collection name (defaults to self.collection_name)
            overwrite: Whether to overwrite existing index (for Milvus incremental update)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting index build for PDFs in: {pdf_dir}")

        # Check if paths exist
        if not pdf_dir.exists():
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            return False

        if not self.pipeline_yaml.exists():
            logger.error(f"Pipeline file not found: {self.pipeline_yaml}")
            return False

        if not self.parameter_template.exists():
            logger.error(f"Parameter template not found: {self.parameter_template}")
            return False

        # Create output directory structure
        output_dir = Path(output_dir).absolute()
        intermediate_dir = output_dir / ".intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        corpus_output = intermediate_dir / "corpus.jsonl"
        chunks_output = output_dir / "chunks.jsonl"
        embedding_output = intermediate_dir / "embeddings.npy"
        index_output = output_dir  # Milvus doesn't use file path

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"  Chunks: {chunks_output}")
        logger.info(f"  Index: Milvus")
        logger.info(f"  Collection: {collection_name or self.collection_name}")
        logger.info(f"  Overwrite: {overwrite}")

        # Step 1: Copy IRIS pipeline to UltraRAG
        ultrarag_pipeline_path = self._copy_pipeline_to_ultrarag()

        # Step 2: Initialize UltraRAG build (creates parameter/ and server/ folders)
        if not self._initialize_ultrarag_build(ultrarag_pipeline_path):
            logger.error("UltraRAG build initialization failed")
            return False

        # Step 3: Create runtime parameter file
        runtime_param = self.ultrarag_path / "parameter" / "_runtime" / "offline_build_index_parameter.yaml"

        replacements = {
            "RAW_PDF_DIR": str(pdf_dir.absolute()),
            "CORPUS_OUTPUT_PATH": str(corpus_output.absolute()),
            "CHUNKS_OUTPUT_PATH": str(chunks_output.absolute()),
            "EMBEDDING_OUTPUT_PATH": str(embedding_output.absolute()),
            "INDEX_OUTPUT_PATH": str(index_output.absolute()),
            "EMBEDDING_MODEL_PATH": self.embedding_model,
            "COLLECTION_NAME": collection_name or self.collection_name,
            "MILVUS_URI": "http://localhost:29901",
            "MILVUS_TOKEN": "null",
            "INDEX_BASE_URL": "http://127.0.0.1:65503/v1",
            "INDEX_MODEL_NAME": "qwen3-embedding-0.6b",
            "OVERWRITE": str(overwrite).lower(),
        }

        self._create_runtime_parameter(self.parameter_template, runtime_param, replacements)

        # Step 4: Copy to location where UltraRAG expects it
        target_param = self.ultrarag_path / "pipelines" / "parameter" / "offline_build_index_parameter.yaml"
        target_param.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(runtime_param, target_param)
        logger.info(f"Copied parameter file to: {target_param}")

        # Step 5: Run UltraRAG pipeline
        success = self._run_ultrarag_pipeline(ultrarag_pipeline_path)

        if success:
            logger.info("Index build completed successfully")
            return True
        else:
            logger.error("Index build failed")
            return False

    def _create_runtime_parameter(
        self,
        template_path: Path,
        output_path: Path,
        replacements: dict
    ):
        """
        Create a runtime parameter file from template.

        Args:
            template_path: Path to template file
            output_path: Path to save parameter file
            replacements: Dictionary of placeholder replacements
        """
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        for key, value in replacements.items():
            placeholder = f"__{key}__"
            content = content.replace(placeholder, str(value))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Created runtime parameter: {output_path}")

    def _copy_pipeline_to_ultrarag(self) -> Path:
        """Copy IRIS pipeline file to UltraRAG pipelines directory.

        Returns:
            Path to the copied pipeline in UltraRAG
        """
        import shutil

        # Copy pipeline file directly (no timestamp, just overwrite)
        ultrarag_pipeline_dir = self.ultrarag_path / "pipelines"
        target_path = ultrarag_pipeline_dir / self.pipeline_yaml.name

        # Ensure target directory exists
        ultrarag_pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Copy pipeline file
        shutil.copy(str(self.pipeline_yaml), str(target_path))
        logger.info(f"Copied IRIS pipeline to UltraRAG: {target_path}")

        return target_path

    def _initialize_ultrarag_build(self, pipeline_path: Path) -> bool:
        """Initialize UltraRAG by running ultrarag build command.

        Args:
            pipeline_path: Path to the pipeline file in UltraRAG

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Initializing UltraRAG build: {pipeline_path}")

        # Find ultrarag command (prefer venv)
        venv_ultrarag = self.ultrarag_path / ".venv" / "bin" / "ultrarag"
        venv_bin = self.ultrarag_path / ".venv" / "bin"

        if venv_ultrarag.exists():
            ultrarag_cmd = str(venv_ultrarag)
            env = os.environ.copy()
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            env["PYTHONPATH"] = str(self.ultrarag_path / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["VIRTUAL_ENV"] = str(self.ultrarag_path / ".venv")
        else:
            ultrarag_cmd = "ultrarag"
            env = None

        try:
            result = subprocess.run(
                [ultrarag_cmd, "build", str(pipeline_path)],
                cwd=self.ultrarag_path,
                capture_output=True,
                text=True,
                check=False,
                env=env
            )

            if result.stdout:
                logger.info(f"UltraRAG build stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"UltraRAG build stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"UltraRAG build failed with code {result.returncode}")
                return False

            logger.info("UltraRAG build completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing UltraRAG build: {e}")
            return False

    def _run_ultrarag_pipeline(self, pipeline_yaml: Path) -> bool:
        """
        Run UltraRAG run command.

        Args:
            pipeline_yaml: Path to the pipeline YAML file

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running UltraRAG pipeline: {pipeline_yaml}")

        # Find ultrarag command (prefer venv)
        venv_ultrarag = self.ultrarag_path / ".venv" / "bin" / "ultrarag"
        venv_bin = self.ultrarag_path / ".venv" / "bin"

        if venv_ultrarag.exists():
            ultrarag_cmd = str(venv_ultrarag)
            logger.debug(f"Using venv ultrarag: {ultrarag_cmd}")
            # Set up environment for venv
            env = os.environ.copy()
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            env["PYTHONPATH"] = str(self.ultrarag_path / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["VIRTUAL_ENV"] = str(self.ultrarag_path / ".venv")
        else:
            ultrarag_cmd = "ultrarag"
            logger.debug(f"Using system ultrarag")
            env = None

        try:
            import subprocess
            result = subprocess.run(
                [ultrarag_cmd, "run", str(pipeline_yaml)],
                cwd=self.ultrarag_path,
                capture_output=True,
                text=True,
                check=False,
                env=env
            )

            if result.stdout:
                logger.info(f"UltraRAG run stdout:\n{result.stdout}")
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

    def get_index_info(self, output_dir: Path) -> dict:
        """
        Get information about the generated index.

        Args:
            output_dir: Directory containing the index files

        Returns:
            Dictionary with index information
        """
        output_dir = Path(output_dir)
        chunks_file = output_dir / "chunks.jsonl"
        index_file = output_dir / "index.index"

        info = {
            "chunks_file": str(chunks_file) if chunks_file.exists() else None,
            "index_file": str(index_file) if index_file.exists() else None,
            "chunk_count": 0
        }

        # Count chunks
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    info["chunk_count"] = sum(1 for _ in f)
            except Exception as e:
                logger.warning(f"Could not count chunks: {e}")

        return info
