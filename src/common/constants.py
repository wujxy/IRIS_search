"""
IRIS Constants

Centralized constant definitions for the IRIS project.
"""

# Project metadata
PROJECT_NAME = "IRIS"
PROJECT_VERSION = "1.2.0"
PROJECT_DESCRIPTION = "Intelligent Research Information System"

# Default paths
DEFAULT_CONFIG_FILE = "configs/config.yaml"
DEFAULT_QUESTIONS_FILE = "configs/questions.txt"
DEFAULT_DATABASE_ROOT = "./data/iris_db"
DEFAULT_PAPER_DB_PATH = "./data/iris_db/papers.db"
DEFAULT_PDF_DIR = "./data/pdfs"
DEFAULT_OUTPUT_DIR = "./data/output"

# Milvus defaults
DEFAULT_MILVUS_URI = "http://localhost:29901"
DEFAULT_MILVUS_COLLECTION = "iris_papers"
DEFAULT_MILVUS_MASTER_COLLECTION = "iris_master"
DEFAULT_EMBEDDING_DIM = 1024

# API defaults
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:65503/v1"
DEFAULT_QA_BASE_URL = "http://127.0.0.1:65504/v1"
DEFAULT_RERANKER_BASE_URL = "http://127.0.0.1:65502/v1"

# Model defaults
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Embedding service defaults
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY = 0.15
DEFAULT_TENSOR_PARALLEL = 1

# QA service defaults
DEFAULT_QA_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_QA_TIMEOUT = 120.0

# Document processing defaults
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_USE_TITLE = True
DEFAULT_USE_SEMANTIC = False

# Retrieval defaults
DEFAULT_TOP_K = 5
DEFAULT_RERANK_MULTIPLIER = 3

# Web service defaults
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000
DEFAULT_WEB_RELOAD = True

# Update defaults
DEFAULT_UPDATE_INTERVAL_HOURS = 2

# arXiv defaults
DEFAULT_ARXIV_MAX_RESULTS = 20
DEFAULT_ARXIV_SORT_BY = "SubmittedDate"

# Filtering defaults
DEFAULT_EXCLUDE_REVIEWS = True
DEFAULT_REVIEW_KEYWORDS = ["review", "survey", "overview"]

# HTTP defaults
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Supported providers
SUPPORTED_LLM_PROVIDERS = ["local", "openai", "anthropic", "cohere"]
SUPPORTED_EMBEDDING_PROVIDERS = ["local", "openai"]

# Environment variable names
ENV_CONFIG_PATH = "IRIS_CONFIG_PATH"
ENV_API_KEY = "IRIS_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_COHERE_API_KEY = "COHERE_API_KEY"
