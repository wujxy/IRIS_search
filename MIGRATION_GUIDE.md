# IRIS Services Migration Guide

## Overview

This guide helps migrate from UltraRAG-dependent services to new independent services with improved directory structure.

## What Changed

### Old (UltraRAG-dependent)
- `index_service.py` - Used UltraRAG subprocess calls
- `qa_service.py` - Used UltraRAG subprocess calls
- Required UltraRAG installation and configuration files in `configs/ultrarag/`

### New (Independent) - Improved Directory Structure

**infrastructure/** (底层基础设施服务)**
- `infrastructure/milvus_service.py` - Direct Milvus operations
- `infrastructure/embedding_service.py` - vLLM embedding API
- `infrastructure/document_processor.py` - PDF parsing and chunking
- `infrastructure/reranker_service.py` - CrossEncoder reranking

**core/** (上层业务逻辑服务)**
- `core/retriever.py` - Integrated retrieval logic
- `core/index_service.py` - Rewritten, uses infrastructure services
- `core/qa_service.py` - Rewritten, uses infrastructure services

**services/** (保留的业务服务)**
- `services/arxiv_service.py` - ArXiv API service
- `services/deploy_service.py` - Deployment service
- `services/email_service.py` - Email service
- `services/paper_service.py` - Paper management service (SQLite)

## Migration Steps

### Step 1: Install New Dependencies

```bash
pip install -r requirements.txt
```

Key new dependencies:
- `pymupdf>=1.23.0` - PDF parsing
- `chonkie>=0.3.0` - Document chunking
- `pymilvus>=2.3.0` - Milvus client
- `openai>=1.12.0` - OpenAI compatible client for vLLM
- `sentence-transformers>=2.3.0` - Reranker models

### Step 2: Update Configuration

Edit `configs/config.yaml` to configure new services:

```yaml
# Add/update these sections
milvus:
  uri: http://localhost:29901
  collection_name: iris_papers
  embedding_dim: 1024

embedding:
  base_url: http://127.0.0.1:65503/v1
  model_name: qwen3-embedding-0.6b
  batch_size: 32

reranker:
  model_path: /path/to/bge-reranker-v2-m3
  device: cpu  # or "cuda:0"
  enabled: false  # Set to true to enable

qa:
  system_prompt: "你的系统提示..."
  top_k: 5

# Update existing config structure with new values
```

### Step 3: Update Code Usage

#### Indexing

**Old code:**
```python
from services.index_service import IndexService

index_svc = IndexService(
    ultrarag_path="/path/to/UltraRAG",
    embedding_model="...",
    reranker_model="...",
    generation_model="...",
)

await index_svc.chunk_and_index(pdf_dir, output_dir)
```

**New code:**
```python
from core.index_service import IndexService

index_svc = IndexService(
    pdf_dir="./data/pdfs",
    output_dir="./data/index",
    # embedding_service, milvus_service are handled internally
)

await index_svc.chunk_and_index()
```

#### QA

**Old code:**
```python
from services.qa_service import QAService

qa_svc = QAService(
    ultrarag_path="/path/to/UltraRAG",
    embedding_model="...",
    generation_model="...",
)

results = await qa_svc.query_knowledge_base_with_mode(
    questions=["问题1", "问题2"],
    chunks_path=chunks_file,
    mode="specific",
    paper_id="2401.12345"
)
```

**New code:**
```python
from core.qa_service import QAService

qa_svc = QAService(
    # Retriever is created internally from config
    generation_base_url="http://127.0.0.1:65504/v1",
    generation_model="llama3-3b-instruct",
)

# Single query
answer = await qa_svc.query(
    question="问题",
    mode="specific",
    paper_id="2401.12345"
)

# Batch query
results = await qa_svc.query_batch(
    questions=["问题1", "问题2"],
    mode="global"
)

# Multi-turn conversation
session_id = qa_svc.create_conversation()
answer1 = await qa_svc.query_with_conversation(
    session_id, "第一轮问题"
    mode="specific",
    paper_id="2401.12345"
)
answer2 = await qa_svc.query_with_conversation(
    session_id, "第二轮问题"
    mode="specific",
    paper_id="2401.12345"
)
```

### Step 4: Run Tests

Test the new services:

```bash
python test_new_services.py
```

This will test:
1. Document Processor (PDF parsing, chunking)
2. Milvus Service (connection, insert, search)
3. Embedding Service (vLLM API call)
4. Reranker Service (CrossEncoder)
5. Retriever (Integrated retrieval)
6. QA Service (RAG with generation)
7. Specific Mode (Paper ID filtering in Milvus)

## Key Differences

| Feature | Old (UltraRAG) | New (Independent) |
|---------|-------------------|-------------------|
| Dependencies | UltraRAG | pymilvus, chonkie, etc. |
| Specific Mode | Chunks file filter | Milvus metadata filter |
| Multi-turn | Manual implementation | Built-in conversation support |
| Reranking | Via UltraRAG | Direct sentence-transformers |
| Configuration | UltraRAG YAML | Direct in config.yaml |
| Directory | All in services/ | infrastructure/ + core/ + services/ |
| Imports | from services.* | from infrastructure.* / from core.* |

## Benefits of New Structure

1. **Better Organization**: Clear separation between infrastructure and business logic
2. **Improved Readability**: Services organized by layer (infrastructure/core/services)
3. **No UltraRAG Dependency**: Fully independent, easier to maintain
4. **Specific Mode Works**: Milvus filter ensures only matching paper chunks are retrieved
5. **Better Performance**: No subprocess overhead, direct Python calls
6. **Multi-turn Support**: Built-in conversation management
7. **Easier Debugging**: All code in project, no external framework
8. **Flexible Customization**: Can modify behavior without touching UltraRAG

## Troubleshooting

### Common Issues

**"pymilvus is not installed"**
```bash
pip install pymilvus>=2.3.0
```

**"chonkie is not installed"**
```bash
pip install chonkie>=0.3.0
```

**"Connection refused on port 29901"**
- Make sure Milvus is running
- Check config: milvus.uri

**"Connection refused on port 65503"**
- Make sure vLLM embedding server is running
- Check config: embedding.base_url

**"Connection refused on port 65504"**
- Make sure vLLM generation server is running
- Check config: generation_base_url (or vllm.base_url in old config)

**"Reranker not found"**
- Check reranker.model_path in config
- Download model if needed:
  ```bash
  wget https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/pytorch_model.bin
  ```

### Testing Individual Components

Test each component separately:

```python
# Test Milvus
python -c "
from infrastructure.milvus_service import MilvusService
m = MilvusService('http://localhost:29901', 'test', 1024)
m.create_collection(1024, True)
print('Milvus OK')
"

# Test Embedding
python -c "
from infrastructure.embedding_service import EmbeddingServiceSync
e = EmbeddingServiceSync('http://127.0.0.1:65503/v1', 'qwen3-embedding-0.6b')
emb = e.encode(['test'])
print(f'Embedding shape: {emb.shape}')
"
```

## Rollback Plan

If issues arise, you can rollback to the old services:

```bash
# Restore old files from core/
mv core/index_service.py services/index_service.py
mv core/qa_service.py services/qa_service.py

# Or use git
git checkout HEAD~1 core/index_service.py core/qa_service.py
```

## Next Steps

1. Run tests and verify all components work
2. Update `run_update_cycle.py` to use new IndexService
3. Update `scripts/iris_query.py` to use new QAService
4. Remove UltraRAG configuration files (configs/ultrarag/)
5. Remove UltraRAG vLLM scripts (vllm_serve_qwen_embed.sh)

## Additional Notes

- **Backward Compatibility**: Old services are renamed with `_old.py` suffix
- **Gradual Migration**: You can use both old and new services during transition
- **Performance**: New services are faster (no subprocess overhead)
- **Specific Mode Fix**: Now properly filters by doc_id in Milvus
