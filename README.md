# IRIS - Intelligent Research Information System

AI-powered literature monitoring and knowledge extraction system for arXiv papers.

## Overview

IRIS automatically:
- Searches arXiv for papers matching your keywords
- Downloads new PDF papers
- Filters out review papers and duplicates
- Chunks and indexes documents using UltraRAG (Qwen3-Embedding-0.6B)
- Generates summaries using Llama3B model
- Sends email notifications (optional)
- Provides a query interface for the knowledge base

## Project Structure

```
IRIS_search/
├── README.md                      # This file
├── IRIS.md                         # Documentation
├── start_IRIS.sh                   # Startup script
├── configs/
│   ├── config.yaml                 # Main configuration
│   └── questions.txt               # Standard questions for summarization
├── services/
│   ├── arxiv_service.py            # arXiv search service
│   ├── index_service.py             # UltraRAG indexing service
│   ├── qa_service.py                # QA service
│   └── email_service.py            # Email notification service
├── scripts/
│   ├── run_update_cycle.py          # Main orchestrator
│   └── iris_query.py               # Query interface
└── utils/
    └── helpers.py                  # Utility functions
```

## Prerequisites

### Required Software
- Python 3.10+
- UltraRAG: `/home/NagaiYoru/LLM_tuning/UltraRAG`
- Models:
  - Embedding: `/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B`
  - Reranker: `/home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light`
  - LLM: `/home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct`

### Python Dependencies
The system uses the following Python packages. Install them in your llm_env:
```bash
source /home/NagaiYoru/LLM_tuning/llm_env/bin/activate
pip install arxiv PyYAML requests
# UltraRAG, vllm, sentence-transformers, faiss-gpu are assumed to be installed
```

## Configuration

Edit `configs/config.yaml` to customize:
- Update interval (default: 2 hours)
- arXiv keywords
- Model paths
- Email settings
- Storage paths (default: `/home/NagaiYoru/research/IRIS_papers`)

## Usage

### Starting vLLM Service (Required for QA)

The Llama3B model needs to be served via vLLM before using QA features:

```bash
cd /home/NagaiYoru/LLM_tuning/UltraRAG
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct \
    --host 127.0.0.1 \
    --port 65504 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7
```

### Running Update Cycle

**Single update:**
```bash
./start_IRIS.sh
# or
python scripts/run_update_cycle.py
```

**Daemon mode (periodic updates):**
```bash
./start_IRIS.sh --daemon --interval 4
```

### Querying the Knowledge Base

**Interactive mode:**
```bash
python scripts/iris_query.py --interactive
```

**Single query:**
```bash
python scripts/iris_query.py "What machine learning methods are used in neutrino experiments?"
```

**List available updates:**
```bash
python scripts/iris_query.py --list-updates
```

**Query specific update:**
```bash
python scripts/iris_query.py --update update_2026_03_15_1200 "What are the key findings?"
```

## Database Structure

The literature database is stored at: `/home/NagaiYoru/research/IRIS_papers`

Each update creates a timestamped folder:
```
IRIS_papers/
├── update_2026_03_15_1200/
│   ├── pdfs/                      # Downloaded PDF files
│   ├── logs/                        # Summary and knowledge logs
│   │   ├── summary_log.md
│   │   └── knowledge_log.md
│   ├── metadata.json              # Paper metadata
│   └── update_log.md             # Update summary
├── update_2026_03_15_1400/
│   └── ...
```

## Email Notifications

To enable email notifications, edit `configs/config.yaml`:
```yaml
email:
  enabled: true
  sender: your_email@gmail.com
  smtp_server: smtp.gmail.com
  smtp_port: 587
  password: your_app_password  # Use app-specific password for Gmail
  receiver: your_email@gmail.com
```

**For Gmail:** Use an app-specific password (not your account password).
Generate one at: https://myaccount.google.com/apppasswords

## Troubleshooting

### vLLM Service Not Available
If you see "vLLM service is not running":
1. Start vLLM service (see "Starting vLLM Service" above)
2. Check service is running: `curl http://127.0.0.1:65504/v1/models`
3. Verify port is correct in `config.yaml`

### UltraRAG Issues
If indexing or QA fails:
1. Verify UltraRAG path is correct in `config.yaml`
2. Check UltraRAG is properly installed: `cd ~/LLM_tuning/UltraRAG && source .venv/bin/activate && which ultrarag`
3. Check model paths are correct

### No Papers Downloaded
1. Check keywords in `config.yaml`
2. Verify internet connectivity to arXiv
3. Check logs in `./logs/` directory

## License

This IRIS implementation is provided as-is for educational and research purposes.
