# IRIS Search Project Structure

## Overview

IRIS Search 的代码已重新组织以提高可读性和可维护性。

## Directory Structure

```
IRIS_search/
├── services/                      # 高层业务服务（ArXiv, Email, 部署等）
│   ├── arxiv_service.py       # ArXiv API 服务
│   ├── deploy_service.py      # 部署服务
│   ├── email_service.py       # 邮件服务
│   ├── paper_service.py      # 论文管理服务 (SQLite)
│   └── __init__.py
│
├── core/                          # 新独立服务的上层逻辑
│   ├── __init__.py
│   ├── retriever.py            # 检索核心逻辑
│   ├── index_service.py        # 索引服务（使用 infrastructure）
│   └── qa_service.py           # 问答服务（使用 infrastructure）
│
├── infrastructure/               # 底层基础设施服务
│   ├── __init__.py
│   ├── milvus_service.py      # Milvus 向量数据库操作
│   ├── embedding_service.py     # vLLM Embedding API 调用
│   ├── document_processor.py  # PDF 解析和文本切分
│   └── reranker_service.py     # CrossEncoder 重排序
│
├── configs/                      # 配置文件
│   ├── config.yaml              # 主配置文件
│   └── ultrarag/              # UltraRAG 相关配置（已废弃）
│
├── scripts/                      # 运行脚本
│   ├── iris_query.py          # 问答查询脚本
│   ├── run_update_cycle.py    # 更新周期脚本
│   └── test_new_services.py    # 新服务测试脚本
│
├── data/                         # 数据目录
│   ├── pdfs/                 # PDF 文件存储
│   └── index/                # 索引文件和 chunks
│
└── logs/                         # 日志文件
```

## Service Dependencies

```
infrastructure/ (底层)
├── milvus_service.py       # 独立，无依赖
├── embedding_service.py     # 独立，无依赖
├── document_processor.py   # 独立，无依赖
└── reranker_service.py     # 独立，无依赖

core/ (上层)
├── retriever.py            # 依赖: infrastructure/*
├── index_service.py        # 依赖: infrastructure/*
└── qa_service.py           # 依赖: infrastructure/*
```

## Import Rules

- **infrastructure/** 服务**: 互相之间不导入，保持独立
- **core/** 服务**: 可以导入 infrastructure/* 服务
- **services/** 服务**: 可以导入 core/* 服务
- **禁止循环导入**: infrastructure 不应导入 core/ 或 services/

## 服务职责

| 服务 | 目录 | 职责 |
|------|--------|------|
| MilvusService | infrastructure/ | Milvus 连接、集合管理、向量检索 |
| EmbeddingService | infrastructure/ | vLLM Embedding API 调用 |
| DocumentProcessor | infrastructure/ | PDF 解析、文本切分 |
| RerankerService | infrastructure/ | CrossEncoder 重排序 |
| Retriever | core/ | 整合 Embedding、Milvus、Reranker 实现检索 |
| IndexService | core/ | 调用 DocumentProcessor 和 Milvus 实现索引 |
| QAService | core/ | 调用 Retriever 和 vLLM 实现问答 |

## 配置使用

配置文件 `configs/config.yaml` 包含以下新部分：

```yaml
milvus:
  uri: http://localhost:29901
  collection_name: iris_papers
  embedding_dim: 768

embedding:
  base_url: http://127.0.0.1:65503/v1
  model_name: qwen3-embedding-0.6b
  batch_size: 32

reranker:
  model_path: /path/to/bge-reranker-v2-m3
  batch_size: 16
  device: cpu
  enabled: false  # 默认关闭，需要时启用

retrieval:
  top_k: 5
  rerank_multiplier: 3

qa:
  system_prompt: "..."
  temperature: 0.7
  max_tokens: 2048
```

## 示例：使用新服务

### 索引

```python
from core.index_service import create_index_service_from_config
import yaml

# 从配置加载
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建索引服务
index_svc = create_index_service_from_config(config)

# 执行索引
await index_svc.chunk_and_index(pdf_dir="data/pdfs", output_dir="data/index")
```

### 问答

```python
from core.qa_service import create_qa_service_from_config
import yaml

# 从配置加载
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建问答服务
qa_svc = create_qa_service_from_config(config)

# 单次查询
answer = await qa_svc.query(
    question="论文的主要贡献是什么？",
    mode="specific",  # 或 "global"
    paper_id="2401.12345"
)

# 批量查询
results = await qa_svc.query_batch(
    questions=["问题1", "问题2"],
    mode="global"
)

# 多轮对话
session_id = qa_svc.create_conversation()
answer1 = await qa_svc.query_with_conversation(
    session_id=session_id,
    question="第一轮问题",
    mode="specific",
    paper_id="2401.12345"
)
answer2 = await qa_svc.query_with_conversation(
    session_id=session_id,
    question="第二轮问题",
    mode="specific",
    paper_id="2401.12345"
)
```

### 测试

```bash
# 测试所有新服务
python test_new_services.py
```

## 迁移注意事项

1. **依赖安装**: 新增了 `pymupdf`, `chonkie`, `pymilvus`, `openai`, `sentence-transformers`
2. **配置更新**: 需要更新 `configs/config.yaml` 添加新配置
3. **服务运行**: 确保以下服务正在运行：
   - Milvus: http://localhost:29901
   - vLLM Embedding: http://127.0.0.1:65503/v1
   - vLLM Generation: http://127.0.0.1:65504/v1
4. **向后兼容**: 旧文件已重命名为 `*_old.py`

## 文件移动

以下文件已从 `services/` 移动：

**移动到 infrastructure/**:**
- `services/milvus_service.py` → `infrastructure/milvus_service.py`
- `services/embedding_service.py` → `infrastructure/embedding_service.py`
- `services/document_processor.py` → `infrastructure/document_processor.py`
- `services/reranker_service.py` → `infrastructure/reranker_service.py`

**移动到 core/**:**
- `services/index_service.py` → `core/index_service.py`
- `services/qa_service.py` → `core/qa_service.py`

**保留在 services/**:**
- `arxiv_service.py` - ArXiv API 服务
- `deploy_service.py` - 部署服务
- `email_service.py` - 邮件服务
- `paper_service.py` - 论文管理服务

**备份文件**（已重命名）:**
- `services/index_service_old.py`
- `services/qa_service_old.py`
