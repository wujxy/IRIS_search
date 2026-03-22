# IRIS - 智能文献信息系统教程
# 独立于 UltraRAG 框架

---

## 快速开始

> 本项目现已支持 **uv 包管理器** 和 **一键部署脚本**，可在 5 分钟内完成部署。

### 快速部署

```bash
# 1. 克隆或进入项目目录
cd /path/to/IRIS_search

# 2. 运行一键部署脚本
./setup.sh

# 3. 配置系统
nano configs/config.yaml

# 4. 启动 Milvus (Docker)
bash build_milvus.sh

# 5. 运行 IRIS
./start.sh
```

**部署选项**：
```bash
./setup.sh              # 标准安装（仅核心依赖）
./setup.sh --gpu        # 安装 GPU 支持
./setup.sh --dev        # 安装开发工具
./setup.sh --all        # 安装所有可选依赖
./setup.sh --reinstall  # 强制重新安装
```

### 快速使用

| 命令 | 描述 |
|------|------|
| `./start.sh` | 运行单次更新周期 |
| `./start.sh daemon` | 守护进程模式（自动定期更新） |
| `./start.sh query` | 交互式查询模式 |
| `./start.sh web` | 启动 Web 界面 (http://127.0.0.1:8000) |
| `./start.sh help` | 显示帮助信息 |

### Web 界面

启动 Web 服务后，访问 **http://127.0.0.1:8000** 可以：
- 浏览所有论文
- 搜索和过滤论文
- 查看论文详情
- 使用 AI 问答功能（全局检索 + 单篇论文检索）

### 关于 uv

[uv](https://github.com/astral-sh/uv) 是一个极速的 Python 包管理器：
- **速度**: 比 pip 快 10-100 倍
- **便捷**: 自动管理虚拟环境
- **可靠**: 统一的依赖管理 (pyproject.toml)

详见 [UV_DEPLOYMENT.md](UV_DEPLOYMENT.md)

---

## 目录

0. [快速开始](#快速开始)
1. [项目介绍](#1-项目介绍)
2. [系统架构](#2-系统架构)
3. [环境准备](#3-环境准备)
4. [配置文件说明](#4-配置文件说明)
5. [安装与部署](#5-安装与部署)
6. [使用指南](#6-使用指南)
7. [实现细节说明](#7-实现细节说明)
8. [故障排除](#8-故障排除)

---

## 1. 项目介绍

### 1.1 IRIS 是什么？

IRIS (Intelligent Research Information System) 是一个自动化的人工智能文献监控和知识提取系统，**完全独立于 UltraRAG 框架**。它能够：

- **自动搜索 arXiv** - 根据关键词定期搜索最新学术论文
- **智能下载** - 自动下载新论文的 PDF 文件
- **自动索引** - 使用独立的文档处理服务将 PDF 切片并建立向量索引
- **AI 摘要** - 使用本地 LLM 模型自动生成论文摘要
- **知识提取** - 从论文中提取关键知识点
- **邮件通知** - 更新完成后发送邮件提醒
- **智能问答** - 支持对文献库进行自然语言查询
- **多轮对话** - 支持上下文保持的连续对话

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| 定期更新 | 可配置的更新间隔（默认 2 小时）|
| 重复检测 | 自动跳过已下载的论文 |
| 过滤评论文章 | 自动排除综述、调查类论文 |
| 本地部署 | 所有模型本地运行，数据安全 |
| 交互式查询 | 支持命令行交互式问答 |
| 批量处理 | 一次可处理多篇论文 |
| 知识日志 | 自动提取并保存关键知识点 |
| Specific 模式 | 使用 Milvus 元数据过滤，精确检索单篇论文 |
| 多轮对话 | 原生支持对话会话管理 |
| 异步架构 | 基于 async/await，性能更佳 |
| Web 界面 | 基于 FastAPI 的文献浏览界面 |
| 实时问答 | 浏览器内 AI 问答交互 |

### 1.3 技术栈

- **Python 3.10+** - 主要编程语言
- **arxiv** - arXiv API 客户端
- **独立服务架构** - 不依赖 UltraRAG，直接调用底层 API
- **Qwen3-Embedding-0.6B** - 嵌入模型（生成向量表示）
- **Llama-3.2-3B-Instruct** - 生成模型（用于摘要和问答）
- **llama3-3b-instruct** - vLLM 服务名（用于 API 调用）
- **Milvus** - 向量数据库（支持增量索引和元数据过滤）
- **Docker** - Milvus 容器部署
- **vLLM** - LLM 推理服务（OpenAI 兼容 API）
- **chonkie** - 文档切分库（支持语义切分）
- **pymupdf** - PDF 解析库
- **SQLite** - 论文数据库（存储元数据和 Q&A 结果）
- **FastAPI** - Web 框架
- **Uvicorn** - ASGI 服务器
- **Jinja2** - 模板引擎
- **MathJax** - LaTeX 公式渲染

### 1.4 架构优势

| 方面 | 旧版本 (UltraRAG) | 新版本 (独立) |
|------|------------------|-------------|
| 依赖 | 需要 UltraRAG 框架 | 完全独立，无框架依赖 |
| Specific 模式 | chunks 文件过滤（无效） | Milvus 元数据过滤（正确） |
| 性能 | subprocess 调用开销 | 原生 async/await，更快 |
| 多轮对话 | 不支持 | 原生支持 |
| 定制化 | 受限于框架 | 完全自由 |
| 代码结构 | UltraRAG + IRIS 两层 | IRIS 单层，更清晰 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IRIS 文献监控系统 (独立架构)            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │arXiv    │──▶│  索引    │──▶│  QA      │    │
│  │ 服务     │   │  服务     │   │  服务     │    │
│  └──────────┘   └──────────┘   └──────────┘    │
│       │               │                  │          │    │
│       ▼               ▼                  ▼          │    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │论文      │   │  向量    │   │  摘要    │    │
│  │数据库    │   │  数据库  │   │  与知识  │    │
│  │(SQLite)  │   │(Milvus)  │   │  日志     │    │
│  └──────────┘   └──────────┘   └──────────┘    │
│       │                             ▼       │
│       ▼                       ┌──────────┐    │
│  ┌──────────┐               │  用户查询 │    │
│  │邮件      │               │  接口    │    │
│  │通知服务  │               └──────────┘    │
│  └──────────┘                                  │
│                                                      │
└──────────────────────────────────────────────────────────────────┘

外部服务:
┌──────────┐   ┌──────────┐   ┌──────────┐
│  vLLM    │   │  vLLM    │   │  arXiv    │
│  索引服务  │   │  QA服务   │   │  API       │
│  (port 65503)│  (port 65504)│  └──────────┘
│──────────┤   └──────────┤
┌──────────┐
│  Milvus  │
│  Docker   │
└──────────┘
```

### 2.2 服务层级架构

```
IRIS_search/
├── src/
│   ├── infrastructure/          # 基础设施层
│   │   ├── milvus_service.py      # Milvus 向量数据库服务
│   │   ├── embedding_service.py   # vLLM Embedding 服务
│   │   ├── document_processor.py  # PDF 解析和文本切分
│   │   └── reranker_service.py   # CrossEncoder 重排序服务
│   ├── core/                   # 核心业务逻辑层
│   │   ├── retriever.py          # 检索核心逻辑
│   │   ├── index_service.py      # 索引服务（使用基础设施）
│   │   ├── qa_service.py         # QA 服务（使用 Retriever）
│   │   └── prompt_templates.py   # Jinja2 提示模板
│   ├── services/               # 业务服务层（独立服务）
│   │   ├── arxiv_service.py       # ArXiv 搜索服务
│   │   ├── paper_service.py       # SQLite 论文数据库
│   │   ├── email_service.py       # 邮件通知服务
│   │   └── deploy_service.py      # 基础设施部署服务
│   └── web/                    # Web 展示模块
│       ├── app.py                # FastAPI 主应用
│       ├── dependencies.py       # 依赖注入
│       ├── template_config.py    # 模板配置
│       ├── models.py             # Pydantic 数据模型
│       ├── routers/              # 路由模块
│       │   ├── web_routes.py     # 网页路由
│       │   ├── api_routes.py     # API 路由
│       │   └── qa_routes.py      # QA 路由
│       ├── templates/            # Jinja2 模板
│       └── static/               # 静态资源
├── scripts/                    # 运行脚本
│   ├── run_update_cycle.py     # 主编排脚本
│   ├── iris_query.py          # 查询接口
│   └── run_web.py             # Web 服务启动脚本
└── utils/                      # 工具函数
    └── helpers.py              # 辅助函数
```

### 2.3 数据流

1. **搜索阶段**: arXiv 服务搜索论文 → 获取元数据
2. **过滤阶段**: 过滤评论文章 → SQLite 数据库查重 → 下载 PDF（失败重试机制）
3. **新论文检测**: 检查是否有新论文
4. **基础设施启动**（仅在有新论文时）: 启动 Milvus → 启动索引模型 vLLM → 启动 QA 模型 vLLM
5. **索引阶段**:
   ```
   PDF → DocumentProcessor.parse_pdf()
       → DocumentProcessor.chunk_text()
       → EmbeddingService.encode() [async]
       → MilvusService.insert() [with metadata]
   ```
6. **QA 阶段**: Specific 模式检索 → Llama3B 模型生成答案 → 提取知识点
7. **数据库保存**: 新论文 + Q&A 结果保存到 SQLite 数据库
8. **通知阶段**: 邮件服务 → 发送更新通知
9. **基础设施停止**: 停止 vLLM 模型 → 停止 Milvus

### 2.4 检索流程

```
用户问题
    ↓
EmbeddingService.encode() [async]
    ↓ (生成查询向量)
MilvusService.search(filter_expr=...)
    ↓
Specific 模式: filter_expr = 'doc_id == "paper_id" or doc_id like "paper_id%"'
Global 模式: 无过滤
    ↓
RerankerService.rerank() [可选]
    ↓ (重排序结果)
QAService.generate() [async, vLLM]
    ↓
最终答案
```

---

## 3. 环境准备

### 3.1 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.10 或更高版本
- **内存**: 至少 16GB RAM
- **GPU**: NVIDIA GPU (推荐 12GB+ VRAM)
- **存储**: 至少 50GB 可用空间
- **Docker**: Docker Engine 20.10+（用于 Milvus）

### 3.2 必需软件

1. **Docker**（Milvus 依赖）
   ```bash
   # 安装 Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # 验证安装
   docker --version
   docker ps
   ```

2. **Python 虚拟环境**
   ```bash
   # 创建 IRIS 专用虚拟环境
   cd /home/NagaiYoru/LLM_tuning/IRIS_search
   python -m venv .venv
   source .venv/bin/activate
   python --version  # 确认 Python 3.10+
   ```

3. **模型文件**
   - 嵌入模型: `/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B`
   - 生成模型: `/home/NagaiYoru/LLM_model/llama-3-8B-Instruct`
   - 重排序模型: `/home/NagaiYoru/LLM_model/bge-reranker-v2-m3`

### 3.3 Python 依赖安装

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 或者手动安装核心依赖
pip install arxiv PyYAML requests
pip install pymupdf chonkie pymilvus openai sentence-transformers
pip install numpy jinja2 tqdm
```

---

## 4. 配置文件说明

### 4.1 config.yaml 结构

主配置文件位于 `configs/config.yaml`，包含以下部分：

#### 4.1.1 更新配置 (update)

```yaml
update:
  interval_hours: 2  # 更新间隔（小时）
```

#### 4.1.2 arXiv 配置 (arxiv)

```yaml
arxiv:
  keywords:              # 搜索关键词列表
    - neutrino
    - juno
    - dune
  max_results_per_keyword: 20  # 每个关键词最大结果数
  sort_by: SubmittedDate    # 排序方式
```

#### 4.1.3 存储配置 (storage)

```yaml
storage:
  database_root: /home/NagaiYoru/research/IRIS_papers  # 文献数据库路径
  paper_db_path: /home/NagaiYoru/research/IRIS_papers/iris_papers.db  # SQLite 数据库
```

**重要**: 文献数据库不存储在 IRIS 项目目录中，需要在配置中指定。

#### 4.1.4 Milvus 配置 (milvus)

```yaml
milvus:
  uri: http://localhost:29901
  collection_name: iris_papers
  master_collection: iris_master  # 主集合名称（增量索引）
  embedding_dim: 1024  # Qwen3-Embedding-0.6B 的维度
  enabled: true
```

**说明**：
- Milvus 以 Docker 容器方式运行
- `master_collection` 用于增量索引，每次更新都追加到这个集合
- `embedding_dim` 必须与嵌入模型匹配

#### 4.1.5 Embedding 配置 (embedding)

```yaml
embedding:
  base_url: http://127.0.0.1:65503/v1
  model_name: qwen3-embedding-0.6b
  batch_size: 32
  enabled: true
  max_model_len: 4096          # vLLM 模型最大长度
  gpu_memory_utilization: 0.15  # GPU 内存利用率
  tensor_parallel_size: 1       # 张量并行大小
  enforce_eager: true           # 使用 eager 执行
```

#### 4.1.6 Reranker 配置 (reranker)

```yaml
reranker:
  model_path: /home/NagaiYoru/LLM_model/bge-reranker-v2-m3
  batch_size: 16
  device: cpu  # 或 cuda:0
  enabled: false  # 可选，用于检索结果重排序
```

#### 4.1.7 模型配置 (models)

```yaml
models:
  embedding_model_path: /home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B
  reranker_model_path: /home/NagaiYoru/LLM_model/bge-reranker-v2-m3
  llm_model_path: /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct
```

**说明**：
- `llm_model_path`: QA 模型的实际文件路径
- `embedding_model_path`: 嵌入模型的实际文件路径
- IRIS 使用两个独立的 vLLM 服务：
  - 索引模型（端口 65503）：专门用于文档嵌入，占用较少 GPU 资源
  - QA 模型（端口 65504）：用于问答和摘要生成，占用较多 GPU 资源

#### 4.1.8 文档处理配置 (document)

```yaml
document:
  chunk_size: 512
  chunk_overlap: 50
  use_title: true
  use_semantic: false  # 可选：是否使用语义切分
  chunk_backend: sentence  # sentence 或 semantic
```

#### 4.1.9 检索配置 (retrieval)

```yaml
retrieval:
  top_k: 5
  rerank_multiplier: 3  # 检索更多用于重排序
```

#### 4.1.10 QA 配置 (qa)

```yaml
qa:
  model_name: llama3-3b-instruct  # vLLM API 调用的服务名
  question_set_path: ./configs/questions.txt
  system_prompt: "你是一个专业的文献库搜索总结智能助手。请基于提供的英文文献，忽略参考文献，使用中文回答问题，要求足够专业。"
  temperature: 0.7
  top_p: 0.8
  max_tokens: 2048
  # vLLM QA 模型配置
  base_url: http://127.0.0.1:65504/v1
  max_model_len: 8192
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 1
  enforce_eager: true
```

#### 4.1.11 过滤配置 (filtering)

```yaml
filtering:
  exclude_reviews: true  # 是否排除综述论文
  review_keywords:       # 综论文论文的关键词列表
    - review
    - survey
    - overview
```

#### 4.1.12 邮件配置 (email)

```yaml
email:
  enabled: false  # 设为 true 启用邮件通知
  sender: your_email@gmail.com
  smtp_server: smtp.gmail.com
  smtp_port: 587
  password: your_app_password
  receiver: your_email@gmail.com
```

**Gmail 用户注意**: 需要使用应用专用密码，而非账号密码。
生成地址: https://myaccount.google.com/apppasswords

#### 4.1.13 Web 配置 (web)

```yaml
web:
  host: 127.0.0.1   # Web 服务监听地址
  port: 8000        # Web 服务端口
  reload: true      # 开发模式自动重载
  log_level: info   # 日志级别
```

**Gmail 用户注意**: 需要使用应用专用密码，而非账号密码。
生成地址: https://myaccount.google.com/apppasswords

### 4.2 questions.txt 说明

标准问题集用于论文摘要生成，每篇论文都会回答这些问题：

```text
What is main problem this paper addresses?
What are key contributions of this paper?
What methods or techniques are used in this paper?
What are important concepts introduced in this paper?
What are possible research directions for future work based on this paper?
```

可以自定义这些问题以获得更针对性的摘要。

---

## 5. 安装与部署

### 5.1 首次部署

#### 步骤 1: 创建数据库目录

```bash
mkdir -p /home/NagaiYoru/research/IRIS_papers
```

#### 步骤 2: 创建虚拟环境

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
python -m venv .venv
source .venv/bin/activate
```

#### 步骤 3: 安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤 4: 配置 IRIS

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search

# 编辑配置文件
nano configs/config.yaml  # 或使用您喜欢的编辑器

# 根据您的环境修改：
# - 模型路径
# - 数据库路径
# - 邮件配置（如需要）
```

#### 步骤 5: 运行首次更新

```bash
# 方法 1: 使用 start_IRIS.sh 脚本（推荐）
./start_IRIS.sh

# 方法 2: 直接运行 Python 脚本
source .venv/bin/activate
python scripts/run_update_cycle.py
```

### 5.2 使用 start_IRIS.sh 脚本（推荐）

IRIS 提供了便捷的启动脚本 `start_IRIS.sh`：

```bash
# 单次更新
./start_IRIS.sh

# 守护进程模式（每 2 小时更新一次）
./start_IRIS.sh --daemon --interval 2

# 自定义配置文件
./start_IRIS.sh --config custom_config.yaml

# 查看帮助
./start_IRIS.sh --help
```

**start_IRIS.sh 功能**：
- 自动检测 Python 虚拟环境
- 支持守护进程模式（daemon mode）
- 可配置更新间隔
- 彩色日志输出
- 优雅的错误处理

### 5.3 其他运行方式

#### 使用 tmux/screen

如果需要手动管理后台进程：

```bash
# 创建新会话
tmux new -s iris
cd /home/NagaiYoru/LLM_tuning/IRIS_search
source .venv/bin/activate
python scripts/run_update_cycle.py
# Ctrl+B, D 分离会话
```

### 5.4 设置 Cron 任务（可选）

创建定时任务（每小时运行）：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每小时运行）
0 * * * * cd /home/NagaiYoru/LLM_tuning/IRIS_search && /home/NagaiYoru/LLM_tuning/IRIS_search/.venv/bin/python src/scripts/run_update_cycle.py >> /home/NagaiYoru/LLM_tuning/IRIS_search/logs/cron.log 2>&1
```

---

## 6. 使用指南

### 6.1 运行更新周期

#### 6.1.1 单次更新

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
source .venv/bin/activate
python src/scripts/run_update_cycle.py
```

输出示例：
```
============================================================
Starting IRIS Update Cycle
============================================================
[INFO] [Step 1] Creating update folder...
[INFO] Update folder: /home/NagaiYoru/research/IRIS_papers/update_2026_03_18_1200
[INFO] [Step 2] Loading existing paper entries...
[INFO] Found 127 existing papers in database
[INFO] [Step 3] Initializing arXiv service...
[INFO] [Step 4] Searching arXiv for papers...
[INFO] Found 15 papers from arXiv
[INFO] [Step 5] Filtering papers...
[INFO] Filtering results: 8 new, 3 duplicates, 4 review papers
[INFO] [Step 6] Downloading PDFs...
[INFO] Successfully downloaded 8/8 PDFs
[INFO] [Step 7] Indexing with new services...
[INFO] Embedding 45 chunks...
[INFO] Inserting into Milvus...
[INFO] QA processing...
[INFO] Generated summaries for 8 papers
============================================================
IRIS Update Cycle Completed Successfully
============================================================
```

#### 6.1.2 查看更新结果

每次更新创建一个带时间戳的文件夹：

```
IRIS_papers/
└── update_2026_03_18_1200/
    ├── pdfs/                          # 下载的 PDF 文件
    │   ├── 2603.09973v1.pdf
    │   └── ...
    ├── logs/                            # 摘要和知识日志
    │   ├── summary_log.md            # 论文摘要
    │   └── knowledge_log.md           # 知识点提取
    ├── metadata.json                    # 论文元数据
    └── update_log.md                   # 更新日志
```

### 6.2 查询文献库

#### 6.2.1 交互式查询模式

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
source .venv/bin/activate
python src/scripts/iris_query.py --interactive
```

交互示例：
```
========================================
IRIS Interactive Query Mode
========================================

Current mode: global
Commands:
  /mode <global|specific>  - Switch query mode
  /paper <id>             - Set paper for specific mode
  /list                    - List available papers
  /search <keyword>        - Search papers by keyword
  /quit                    - Exit

Enter your questions about literature.

IRIS[global]> What machine learning methods are used in neutrino experiments?
----------------------------------------
Question: What machine learning methods are used in neutrino experiments?
----------------------------------------

Answer:
根据搜索到的相关文献，中微子实验中使用的机器学习方法包括：

1. 卷积神经网络（CNN）：用于中微子探测器中的信号重建和模式识别...
（详细回答...）
```

#### 6.2.2 Specific 模式查询

```bash
# 列出可用论文
python src/scripts/iris_query.py --list-papers

# 特定论文查询
python src/scripts/iris_query.py --mode specific --paper-id 2401.12345 "What is the main method?"
```

#### 6.2.3 单次查询（Global 模式）

```bash
python src/scripts/iris_query.py "What are the key differences between liquid and scintillator detectors?"
```

#### 6.2.4 查看可用更新

```bash
python src/scripts/iris_query.py --list-updates
```

输出示例：
```
Available Updates:

  update_2026_03_18_1200
    Time: 2026-03-18T12:00:00
    New papers: 8

  update_2026_03_18_1400
    Time: 2026-03-18T14:00:00
    New papers: 5
```

### 6.3 命令行选项

```bash
# 查看帮助
python src/scripts/iris_query.py --help

# 使用自定义配置
python src/scripts/iris_query.py --config custom_config.yaml

# 指定数据库路径
python src/scripts/iris_query.py --database /path/to/database

# 启动基础设施
python src/scripts/iris_query.py --start-infra

# 交互模式
python src/scripts/iris_query.py -i

# Global 模式（默认）
python src/scripts/iris_query.py "问题"

# Specific 模式
python src/scripts/iris_query.py -m specific --paper-id 2401.12345 "问题"
```

### 6.4 Web 界面使用

#### 6.4.1 启动 Web 服务

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
source .venv/bin/activate

# 启动 Web 服务
python scripts/run_web.py
```

输出示例：
```
[INFO] Starting IRIS Web Server on 127.0.0.1:8000
[INFO] IRIS Web application starting up...
```

访问地址: `http://127.0.0.1:8000`

**注意**: 使用 QA 功能前需要确保 vLLM 服务和 Milvus 正在运行。

#### 6.4.2 文献列表页面

**功能**:
- 分页浏览（每页 10 篇）
- 分类过滤（hep-ph, hep-ex, astro-ph 等）
- 关键词搜索
- 排序选项（最新、最早）

**URL**: `http://127.0.0.1:8000/`

**操作**:
- 点击论文标题查看详情
- 使用搜索框查找论文
- 使用分类下拉框过滤

#### 6.4.3 论文详情页面

**功能**:
- 完整元数据展示（标题、作者、摘要、分类）
- PDF 下载链接（直接跳转 arXiv）
- DOI 链接
- LaTeX 公式渲染（MathJax）
- **嵌入式聊天框**（Specific 模式 QA）

**URL**: `http://127.0.0.1:8000/paper/{arxiv_id}`

#### 6.4.4 AI 问答功能

**Global 模式**（全局检索）:
- **位置**: 右下角浮动聊天按钮
- **功能**: 搜索整个文献数据库
- **使用**: 点击按钮展开聊天框，输入问题

**Specific 模式**（单篇论文检索）:
- **位置**: 论文详情页嵌入式聊天框
- **功能**: 仅针对当前论文内容检索
- **使用**: 在详情页底部聊天框输入问题

**多轮对话**:
- 自动保持会话上下文
- 支持追问和深入讨论
- 会话自动管理

#### 6.4.5 API 端点（程序化访问）

**文献 API**:
- `GET /api/papers` - 获取论文列表（分页）
- `GET /api/papers/{arxiv_id}` - 获取论文详情
- `GET /api/categories` - 获取分类列表
- `POST /api/search` - 搜索论文

**QA API**:
- `POST /api/qa/conversation` - 创建对话会话
- `POST /api/qa/conversation/{session_id}` - 发送问题
- `GET /api/qa/conversation/{session_id}` - 获取历史
- `DELETE /api/qa/conversation/{session_id}` - 删除会话
- `POST /api/qa/query` - 单次查询（无状态）

**API 文档**: `http://127.0.0.1:8000/docs` (FastAPI 自动生成)

**API 使用示例**:

```bash
# 获取论文列表
curl http://localhost:8000/api/papers?page=1&per_page=10

# 获取特定论文详情
curl http://localhost:8000/api/papers/2401.12345

# 搜索论文
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"keyword": "neutrino", "category": "hep-ph"}'

# 创建 QA 会话
curl -X POST http://localhost:8000/api/qa/conversation

# 在会话中提问
curl -X POST http://localhost:8000/api/qa/conversation/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main method?", "mode": "global", "top_k": 5}'
```

---

## 7. 实现细节说明

### 7.1 架构设计原则

IRIS 采用分层架构设计，各层职责明确：

```
┌─────────────────────────────────────────────────┐
│  scripts/  - 入口和编排逻辑            │
├─────────────────────────────────────────────────┤
│  services/  - 独立业务服务           │
│  （不依赖 UltraRAG）                │
├─────────────────────────────────────────────────┤
│  core/      - 核心检索和 QA 逻辑        │
│  （使用基础设施服务）                │
├─────────────────────────────────────────────────┤
│  infrastructure/ - 底层技术服务         │
│  （Milvus, Embedding 等）           │
└─────────────────────────────────────────────────┘
```

**设计原则**：
1. **基础设施层**：封装底层技术细节（Milvus、Embedding、Document Process）
2. **核心层**：实现业务逻辑（Retriever、QA、Index）
3. **服务层**：提供独立业务功能（ArXiv、Paper、Email）
4. **脚本层**：编排整个流程

### 7.2 基础设施服务 (src/infrastructure/)

#### 7.2.1 Milvus 服务 (`milvus_service.py`)

**核心功能**：
- 集合管理（创建、删除）
- 向量插入（批量处理）
- 向量检索（支持元数据过滤）
- 文档 ID 查询（通过 `doc_id` 字段）

**关键特性**：
- 使用 Milvus 元数据过滤实现 Specific 模式
- 过滤表达式：`doc_id == "paper_id" or doc_id like "paper_id%"`
- 支持增量索引（不覆盖现有数据）

**代码示例**：
```python
from infrastructure.milvus_service import MilvusService

milvus = MilvusService(
    uri="http://localhost:29901",
    collection_name="iris_papers",
    embedding_dim=1024
)

# 创建集合
milvus.create_collection(dim=1024, overwrite=False)

# 插入数据（带元数据）
milvus.insert(embeddings, chunks)

# Specific 模式检索（使用元数据过滤）
results = milvus.search(
    query_embedding,
    top_k=5,
    filter_expr='doc_id == "2401.12345" or doc_id like "2401.12345%"'
)
```

#### 7.2.2 Embedding 服务 (`embedding_service.py`)

**核心功能**：
- 异步批量编码
- vLLM OpenAI 兼容 API
- 错误处理和重试

**代码示例**：
```python
from infrastructure.embedding_service import EmbeddingService

embed_svc = EmbeddingService(
    base_url="http://127.0.0.1:65503/v1",
    model_name="qwen3-embedding-0.6b",
    batch_size=32
)

# 异步编码
embeddings = await embed_svc.encode(["文本1", "文本2"])

# 同步编码（用于非异步上下文）
embeddings = embed_svc.encode_sync(["文本1", "文本2"])
```

#### 7.2.3 文档处理服务 (`document_processor.py`)

**核心功能**：
- PDF 解析（使用 pymupdf）
- 文本切分（使用 chonkie 或简单字符切分）
- 元数据提取（doc_id, title, contents）

**代码示例**：
```python
from infrastructure.document_processor import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=512,
    chunk_overlap=50,
    use_semantic_chunking=False
)

# 解析 PDF
doc = processor.parse_pdf(Path("paper.pdf"))

# 切分文本
chunks = processor.chunk_text(
    text=doc['contents'],
    doc_id=doc['id'],
    title=doc['title'],
    chunk_size=512
)

# 或一步完成
chunks = processor.parse_and_chunk_pdf(Path("paper.pdf"))
```

#### 7.2.4 Reranker 服务 (`reranker_service.py`)

**核心功能**：
- CrossEncoder 重排序
- 批量处理
- 设备选择（CPU/GPU）

**代码示例**：
```python
from infrastructure.reranker_service import RerankerService

reranker = RerankerService(
    model_path="/path/to/bge-reranker-v2-m3",
    device="cpu"
)

# 重排序
reranked = reranker.rerank(query, passages, top_k=5)
```

### 7.3 核心服务 (src/core/)

#### 7.3.1 检索器 (`retriever.py`)

**核心功能**：
- 整合 Embedding、Milvus、Reranker
- 支持 global/specific 模式
- 异步检索

**实现逻辑**：
```python
class Retriever:
    async def retrieve(self, query, mode="global", paper_id=None, top_k=5):
        # 1. 生成查询 embedding
        query_emb = await self.embedding_service.encode([query])

        # 2. 构建 Milvus filter 表达式
        filter_expr = None
        if mode == "specific" and paper_id:
            filter_expr = f'doc_id == "{paper_id}" or doc_id like "{paper_id}%"'

        # 3. Milvus 检索（获取更多用于 rerank）
        results = self.milvus_service.search(
            query_emb[0],
            top_k=top_k * 3,
            filter_expr=filter_expr
        )

        # 4. Rerank
        if self.reranker_service and len(results) > top_k:
            passages = [r['contents'] for r in results]
            reranked = self.reranker_service.rerank(query, passages, top_k)
            results = [results[idx] for idx, _ in reranked]

        return results[:top_k]
```

#### 7.3.2 索引服务 (`index_service.py`)

**核心功能**：
- 批量处理 PDF
- 异步 embedding 和 Milvus 插入
- 保存 chunks.jsonl 文件

**实现逻辑**：
```python
class IndexService:
    async def chunk_and_index(self, pdf_dir, output_dir, overwrite=False):
        # 1. 处理所有 PDF
        all_chunks = []
        for pdf_file in Path(pdf_dir).glob("*.pdf"):
            chunks = self.doc_processor.parse_and_chunk_pdf(pdf_file)
            all_chunks.extend(chunks)

        # 2. 生成 embeddings
        texts = [c['contents'] for c in all_chunks]
        embeddings = await self.embedding_service.encode(texts)

        # 3. 存入 Milvus
        self.milvus_service.insert(embeddings, all_chunks)

        # 4. 保存 chunks.jsonl
        with open(chunks_file, 'w') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + '\n')
```

#### 7.3.3 QA 服务 (`qa_service.py`)

**核心功能**：
- RAG 问答
- Global/Specific 模式支持
- 多轮对话会话管理
- 批量问答

**实现逻辑**：
```python
class QAService:
    async def query(self, question, mode="global", paper_id=None, top_k=5):
        # 1. 检索相关 chunks
        retrieved = await self.retriever.retrieve(
            question,
            mode=mode,
            paper_id=paper_id,
            top_k=top_k
        )

        # 2. 构建上下文
        context = "\n\n".join([
            f"[文档 {i+1}]\n{r['contents']}"
            for i, r in enumerate(retrieved)
        ])

        # 3. 构建提示
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{question}"}
        ]

        # 4. 生成答案
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )

        return response.choices[0].message.content

    # 多轮对话支持
    def create_conversation(self) -> str:
        return str(uuid.uuid4())

    async def query_with_conversation(self, session_id, question, mode="global", paper_id=None):
        history = self._get_history(session_id)
        answer = await self.query(question, mode, paper_id, conversation_history=history)
        self._save_history(session_id, question, answer)
        return answer
```

### 7.4 独立业务服务 (src/services/)

这些服务不依赖 UltraRAG，保持独立实现：

#### 7.4.1 ArXiv 服务 (`arxiv_service.py`)

**核心功能**：
- 论文搜索（多关键词 OR 查询）
- 重复检测（基于 entry_id）
- 评论文章过滤
- PDF 下载（失败重试）

#### 7.4.2 Paper 服务 (`paper_service.py`)

**核心功能**：
- SQLite 数据库管理
- 论文增删查改
- 状态管理（new, duplicate, review）

#### 7.4.3 Email 服务 (`email_service.py`)

**核心功能**：
- SMTP/TLS 邮件发送
- HTML 格式支持
- 更新通知生成

#### 7.4.4 Deploy 服务 (`deploy_service.py`)

**核心功能**：
- Milvus Docker 容器管理
- vLLM 模型服务管理（索引 + QA 两个独立服务）
- 健康检查和自动启动
- **条件启动**: 仅在有新论文时启动基础设施
- **错误处理**: 启动失败时终止更新周期

**健康检查改进**：
- 检测超时: 10 秒（从 5 秒增加）
- 启动后验证: 60 秒等待模型完全初始化
- 详细的日志输出，显示检测 URL 和状态

**vLLM 服务参数**（从 config.yaml 读取）：

索引模型 (embedding):
- `max_model_len`: 4096
- `gpu_memory_utilization`: 0.15
- `tensor_parallel_size`: 1
- `enforce_eager`: true

QA 模型:
- `max_model_len`: 8192
- `gpu_memory_utilization`: 0.85
- `tensor_parallel_size`: 1
- `enforce_eager`: true

**代码示例**：
```python
from services.deploy_service import DeployService

deploy_server = DeployService(config)

# 启动基础设施（仅在有新论文时调用）
if deploy_server.start_infrastructure():
    # 执行索引和 QA 任务
    pass
else:
    # 启动失败，终止更新周期
    logger.error("Infrastructure startup failed")

# 停止基础设施
deploy_server.stop_infrastructure()
```

#### 手动启动 vLLM 服务（可选）

通常 DeployService 会自动管理 vLLM 服务，但也可以手动启动：

```bash
# 启动索引模型 (端口 65503)
bash vllm_serve_qwen_embed.sh

# 启动 QA 模型 (端口 65504)
bash vllm_serve_llama.sh
```

### 7.5 主编排器 (`scripts/run_update_cycle.py`)

**完整更新流程**：

```
开始
  ↓
[Step 1] 创建更新文件夹
  ↓
[Step 2] 初始化 PaperService，从 SQLite 加载现有论文
  ↓
[Step 3] 初始化 arXiv 服务
  ↓
[Step 4] 搜索 arXiv 并管理论文（过滤、下载）
  ↓
[Step 5] 保存元数据
  ↓
[Step 6] 生成更新日志
  ↓
[Step 7] 检查是否有新论文
  ├─ 无新论文 → 跳过索引、QA 和基础设施启动 → 完成
  └─ 有新论文 ↓
[Step 8] 启动基础设施（Milvus + vLLM 索引 + vLLM QA）
  ↓
[Step 9] 构建索引（增量更新）
  ↓
[Step 10] QA 处理（Specific 模式）
  ↓
[Step 11] 生成摘要和知识日志
  ↓
[Step 11.5] 保存新论文到 SQLite 数据库
  ↓
[Step 12] 发送邮件通知
  ↓
[Step 13] 停止基础设施
  ↓
完成
```

**关键特性**：
- **智能基础设施启动**: 仅在有新论文时启动 Milvus 和 vLLM 服务，节省资源
- **SQLite 数据库集成**: 自动保存新论文和 Q&A 结果到数据库
- **错误终止**: 如果基础设施或索引构建失败，立即停止并清理

---

## 8. 故障排除

### 8.1 常见问题

#### 问题 1: Milvus 容器无法启动

**症状**：
```
[ERROR] Failed to start Milvus
```

**解决方案**：

1. 检查 Docker 是否运行：
   ```bash
   docker ps
   docker --version
   ```

2. 检查端口是否被占用：
   ```bash
   netstat -tuln | grep 2990
   ```

3. 查看 Milvus 日志：
   ```bash
   docker logs milvus-ultrarag
   ```

#### 问题 2: vLLM 服务健康检查失败

**症状**：
```
[ERROR] vLLM service not ready after 30 seconds
```

**解决方案**：

1. 检查 vLLM 日志：
   ```bash
   tail -f logs/vllm_index_output.log
   tail -f logs/vllm_qa_output.log
   ```

2. 验证服务状态：
   ```bash
   curl http://127.0.0.1:65503/v1/models  # 索引模型
   curl http://127.0.0.1:65504/v1/models  # QA 模型
   ```

3. 检查 GPU 可用性：
   ```bash
   nvidia-smi
   ```

#### 问题 3: 模型文件不存在

**症状**：
```
[ERROR] Model file not found: /path/to/model
```

**解决方案**：

1. 确认模型文件路径：
   ```bash
   ls -la /home/NagaiYoru/LLM_model/
   ```

2. 检查配置文件中的模型路径：
   ```bash
   grep "model_path" configs/config.yaml
   ```

#### 问题 4: PDF 下载失败

**症状**：
```
[ERROR] Failed to download PDF for paper title...
```

**解决方案**：

1. 检查网络连接：
   ```bash
   ping arxiv.org
   curl -I https://arxiv.org
   ```

2. 检查磁盘空间：
   ```bash
   df -h /home/NagaiYoru/research/IRIS_papers
   ```

#### 问题 5: Specific 模式检索不正确

**症状**：
```
检索结果包含其他论文的内容
```

**解决方案**：

1. 检查 Milvus 元数据是否正确存储：
   ```python
   from infrastructure.milvus_service import MilvusService
   milvus = MilvusService(...)
   results = milvus.get_chunks_by_doc_id("2401.12345")
   print(f"Retrieved {len(results)} chunks for doc_id")
   ```

2. 确认 `doc_id` 字段在 chunks 中正确设置

#### 问题 6: 索引构建失败

**症状**：
```
[ERROR] Index build failed
```

**解决方案**：

1. 检查日志：
   ```bash
   tail -f logs/iris_*.log
   ```

2. 检查 Milvus 连接：
   ```bash
   curl http://localhost:29901/healthz
   ```

3. 手动测试服务：
   ```bash
   cd /home/NagaiYoru/LLM_tuning/IRIS_search
   source .venv/bin/activate
   python test_new_services.py
   ```

### 8.2 调试方法

#### 启用调试日志

```bash
# 运行时指定调试级别
python src/scripts/run_update_cycle.py --log-level DEBUG

# 或修改配置文件
# 编辑 configs/config.yaml
logging:
  level: DEBUG
```

#### 查看日志文件

```bash
# 查看最新的日志
ls -lt logs/

# 实时查看日志
tail -f logs/iris_$(date +%Y%m%d_%H%M%S).log

# 查看错误
cat logs/iris_*.log | grep ERROR
```

### 8.3 数据库维护

#### 清理旧更新

```bash
# 删除指定的更新
rm -rf /home/NagaiYoru/research/IRIS_papers/update_2026_03_15_1000

# 只保留最近 N 个更新
cd /home/NagaiYoru/research/IRIS_papers
ls -td update_* | tail -n +6 | xargs rm -rf
```

#### 重建索引

如果需要重建整个索引：

1. 备份当前数据库：
   ```bash
   cp -r /home/NagaiYoru/research/IRIS_papers/iris_papers.db iris_papers_backup.db
   ```

2. 删除 Milvus 集合：
   ```bash
   python -c "from infrastructure.milvus_service import MilvusService; m = MilvusService('http://localhost:29901', 'iris_papers'); m.drop_collection()"
   ```

3. 重新运行更新周期（将重建索引）

---

## 附录

### A.1 命令快速参考

| 命令 | 描述 |
|------|------|
| `source .venv/bin/activate` | 激活虚拟环境 |
| `./start_IRIS.sh` | 运行单次更新（推荐） |
| `./start_IRIS.sh --daemon` | 守护进程模式 |
| `./start_IRIS.sh --interval 4` | 设置更新间隔为 4 小时 |
| `python scripts/run_update_cycle.py` | 运行单次更新 |
| `python scripts/iris_query.py -i` | 交互式查询 |
| `python scripts/iris_query.py "问题"` | 单次查询 |
| `python scripts/iris_query.py -l` | 列出更新 |
| `python scripts/iris_query.py -m specific --paper-id XXXXX "问题"` | Specific 模式查询 |
| `python scripts/run_web.py` | 启动 Web 服务 |
| `bash vllm_serve_qwen_embed.sh` | 手动启动索引模型 |
| `bash vllm_serve_llama.sh` | 手动启动 QA 模型 |
| 访问 `http://localhost:8000/docs` | 查看 Web API 文档 |

### A.2 配置参数速查

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `update.interval_hours` | 1 | 更新间隔（小时） |
| `arxiv.max_results_per_keyword` | 20 | 每关键词最大结果 |
| `storage.database_root` | `~/research/IRIS_papers` | 数据库路径 |
| `storage.paper_db_path` | `~/research/IRIS_papers/iris_papers.db` | SQLite 数据库路径 |
| `milvus.uri` | `http://localhost:29901` | Milvus 服务地址 |
| `milvus.collection_name` | `iris_papers` | Milvus 集合名称 |
| `models.embedding_model_path` | Qwen3-Embedding-0.6B | 嵌入模型路径 |
| `models.llm_model_path` | Llama-3.2-3B-Instruct | 生成模型路径 |
| `embedding.base_url` | `http://127.0.0.1:65503/v1` | Embedding 服务地址 |
| `embedding.max_model_len` | 4096 | 索引模型最大长度 |
| `embedding.gpu_memory_utilization` | 0.15 | 索引模型 GPU 内存利用率 |
| `qa.base_url` | `http://127.0.0.1:65504/v1` | QA 服务地址 |
| `qa.model_name` | llama3-3b-instruct | QA 模型服务名 |
| `qa.max_model_len` | 8192 | QA 模型最大长度 |
| `qa.gpu_memory_utilization` | 0.85 | QA 模型 GPU 内存利用率 |
| `filtering.exclude_reviews` | true | 是否排除综述论文 |
| `web.host` | 127.0.0.1 | Web 服务监听地址 |
| `web.port` | 8000 | Web 服务端口 |
| `web.reload` | true | 开发模式自动重载 |
| `web.log_level` | info | Web 服务日志级别 |

### A.3 相关链接

- [arXiv](https://arxiv.org/)
- [Milvus 文档](https://milvus.io/docs)
- [vLLM 文档](https://docs.vllm.ai/)
- [Qwen 模型](https://huggingface.co/Qwen)
- [Llama 模型](https://huggingface.co/meta-llama)
- [Chonkie](https://github.com/bhavnicksm/chonkie)

---

## 总结

IRIS 提供了一个完整的自动化文献监控和知识管理解决方案，**完全独立于 UltraRAG 框架**。通过本教程，您应该能够：

1. ✓ 部署完整的 IRIS 系统
2. ✓ 配置所有必要的参数
3. ✓ 运行自动更新周期
4. ✓ 使用交互式查询接口
5. ✓ 理解系统的实现原理
6. ✓ 处理常见问题和故障
7. ✓ 了解新架构的优势和特性

如有问题，请查看日志文件或参考故障排除章节。
