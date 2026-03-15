# IRIS - 智能文献信息系统教程

## 目录

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

IRIS (Intelligent Research Information System) 是一个自动化的人工智能文献监控和知识提取系统。它能够：

- **自动搜索 arXiv** - 根据关键词定期搜索最新学术论文
- **智能下载** - 自动下载新论文的 PDF 文件
- **自动索引** - 使用 UltraRAG 将 PDF 切片并建立向量索引
- **AI 摘要** - 使用本地 LLM 模型自动生成论文摘要
- **知识提取** - 从论文中提取关键知识点
- **邮件通知** - 更新完成后发送邮件提醒
- **智能问答** - 支持对文献库进行自然语言查询

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

### 1.3 技术栈

- **Python 3.10+** - 主要编程语言
- **arxiv** - arXiv API 客户端
- **UltraRAG** - RAG 框架，用于文档处理和向量检索
- **Qwen3-Embedding-0.6B** - 嵌入模型（生成向量表示）
- **Llama-3.2-3B-Instruct** - 生成模型（用于摘要和问答）
- **FAISS** - 向量索引库
- **vLLM** - LLM 推理服务

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IRIS 文献监控系统                      │
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
│  │数据库    │   │  索引库  │   │  与知识  │    │
│  └──────────┘   └──────────┘   │  日志     │    │
│       │                             └──────────┘    │
│       ▼                                     ▼       │
│  ┌──────────┐                           ┌──────────┐    │
│  │邮件      │                           │  用户查询 │    │
│  │通知服务  │                           │  接口    │    │
│  └──────────┘                           └──────────┘    │
│                                                      │
└──────────────────────────────────────────────────────────────────┘

外部服务:
┌──────────┐   ┌──────────┐   ┌──────────┐
│  vLLM    │   │ UltraRAG  │   │  arXiv    │
│  服务     │   │ 框架     │   │  API       │
└──────────┘   └──────────┘   └──────────┘
```

### 2.2 数据流

1. **搜索阶段**: arXiv 服务搜索论文 → 获取元数据
2. **过滤阶段**: 过滤评论文章 → 检测重复 → 下载 PDF
3. **索引阶段**: UltraRAG 切片 PDF → Qwen0.3B 嵌入 → 生成向量索引
4. **摘要阶段**: Llama3B 模型 → 生成摘要 → 提取知识点
5. **通知阶段**: 邮件服务 → 发送更新通知

### 2.3 项目结构

```
IRIS_search/
├── IRIS_Tutorial.md              # 本教程文档
├── IRIS.md                     # 项目详细文档
├── README.md                   # 英文使用指南
├── start_IRIS.sh                # 启动脚本
├── configs/
│   ├── config.yaml             # 主配置文件
│   └── questions.txt           # 标准问题集
├── services/                   # 服务层
│   ├── arxiv_service.py        # arXiv 搜索服务
│   ├── index_service.py        # 索引服务
│   ├── qa_service.py           # 问答服务
│   └── email_service.py        # 邮件服务
├── scripts/                    # 脚本层
│   ├── run_update_cycle.py     # 主编排脚本
│   └── iris_query.py          # 查询接口
└── utils/                      # 工具层
    └── helpers.py              # 辅助函数
```

---

## 3. 环境准备

### 3.1 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.10 或更高版本
- **内存**: 至少 16GB RAM
- **GPU**: NVIDIA GPU (推荐 12GB+ VRAM)
- **存储**: 至少 50GB 可用空间

### 3.2 必需软件

1. **UltraRAG 框架**
   ```bash
   cd ~/LLM_tuning/UltraRAG
   # 确保 UltraRAG 已正确安装
   source .venv/bin/activate
   which ultrarag
   ```

2. **Python 虚拟环境**
   ```bash
   # 使用已有的 llm_env
   source /home/NagaiYoru/LLM_tuning/llm_env/bin/activate
   python --version  # 确认 Python 3.10+
   ```

3. **模型文件**
   - 嵌入模型: `/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B`
   - 生成模型: `/home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct`
   - 重排序模型: `/home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light`

### 3.3 Python 依赖安装

```bash
source /home/NagaiYoru/LLM_tuning/llm_env/bin/activate

# 安装核心依赖
pip install arxiv PyYAML requests

# UltraRAG、vLLM、sentence-transformers、faiss-gpu 等
# 这些应该在 UltraRAG 环境中已安装
# 如果需要，请参考 UltraRAG 的安装文档
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
```

**重要**: 文献数据库不存储在 IRIS 项目目录中，需要在配置中指定。

#### 4.1.4 UltraRAG 配置 (ultrarag)

```yaml
ultrarag:
  ultrarag_path: /home/NagaiYoru/LLM_tuning/UltraRAG
  index_backend: faiss      # 索引后端: faiss 或 milvus
  index_storage: /home/NagaiYoru/research/IRIS_papers/index_storage
```

#### 4.1.5 模型配置 (models)

```yaml
models:
  embedding_model_path: /home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B
  reranker_model_path: /home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light
  llm_model_path: /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct
  vllm:
    base_url: http://127.0.0.1:65504/v1
    served_model_name: llama3-3b-instruct
```

#### 4.1.6 QA 配置 (qa)

```yaml
qa:
  question_set_path: ./configs/questions.txt
  temperature: 0.7
  top_p: 0.8
  max_tokens: 2048
  system_prompt: "你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业"
```

#### 4.1.7 邮件配置 (email)

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

### 4.2 questions.txt 说明

标准问题集用于论文摘要生成，每篇论文都会回答这些问题：

```text
What is main problem this paper addresses?
What are the key contributions of this paper?
What methods or techniques are used in this paper?
What are the important concepts introduced in this paper?
What are the possible research directions for future work based on this paper?
```

可以自定义这些问题以获得更针对性的摘要。

---

## 5. 安装与部署

### 5.1 首次部署

#### 步骤 1: 创建数据库目录

```bash
mkdir -p /home/NagaiYoru/research/IRIS_papers
```

#### 步骤 2: 启动 vLLM 服务

**重要**: QA 功能需要 vLLM 服务运行。

```bash
cd /home/NagaiYoru/LLM_tuning/UltraRAG
source .venv/bin/activate

# 启动 Llama3B 模型服务
python -m vllm.entrypoints.openai.api_server \
    --model /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 65504 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size 1 \
    --enforce-eager
```

**验证服务启动**:

```bash
curl http://127.0.0.1:65504/v1/models
```

应该返回类似以下内容：
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3-3b-instruct",
      "object": "model",
      "created": 1710000000,
      ...
    }
  ]
}
```

#### 步骤 3: 配置 IRIS

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search

# 编辑配置文件
nano configs/config.yaml  # 或使用您喜欢的编辑器

# 根据您的环境修改：
# - 模型路径
# - UltraRAG 路径
# - 数据库路径
# - 邮件配置（如需要）
```

#### 步骤 4: 运行首次更新

```bash
# 方法 1: 使用启动脚本
./start_IRIS.sh

# 方法 2: 直接运行 Python 脚本
python scripts/run_update_cycle.py
```

### 5.2 守护进程模式

要让 IRIS 在后台定期运行：

```bash
./start_IRIS.sh --daemon --interval 2
```

这将：
1. 立即运行一次更新
2. 之后每 2 小时自动运行一次
3. 按 Ctrl+C 停止

### 5.3 设置 Cron 任务（可选）

创建定时任务（每小时运行）：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每小时运行）
0 * * * * cd /home/NagaiYoru/LLM_tuning/IRIS_search && /home/NagaiYoru/LLM_tuning/llm_env/bin/python scripts/run_update_cycle.py >> /home/NagaiYoru/LLM_tuning/IRIS_search/logs/cron.log 2>&1
```

---

## 6. 使用指南

### 6.1 运行更新周期

#### 6.1.1 单次更新

```bash
cd /home/NagaiYoru/LLM_tuning/IRIS_search
./start_IRIS.sh
```

输出示例：
```
[INFO] Starting IRIS literature monitoring service...
[INFO] Python environment: /home/NagaiYoru/LLM_tuning/llm_env/bin/python
[INFO] Configuration file: configs/config.yaml
============================================================
Starting IRIS Update Cycle
============================================================
[INFO] [Step 1] Creating update folder...
[INFO] Update folder: /home/NagaiYoru/research/IRIS_papers/update_2026_03_15_1200
[INFO] [Step 2] Loading existing paper entries...
[INFO] Found 127 existing papers in database
[INFO] [Step 3] Initializing arXiv service...
[INFO] [Step 4] Searching arXiv for papers...
[INFO] Found 15 papers from arXiv
[INFO] [Step 5] Filtering papers...
[INFO] Filtering results: 8 new, 3 duplicates, 4 review papers
[INFO] [Step 6] Downloading PDFs...
[INFO] Successfully downloaded 8/8 PDFs
...
============================================================
IRIS Update Cycle Completed Successfully
============================================================
```

#### 6.1.2 查看更新结果

每次更新创建一个带时间戳的文件夹：

```
IRIS_papers/
└── update_2026_03_15_1200/
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
python scripts/iris_query.py --interactive
```

交互示例：
```
========================================
IRIS Interactive Query Mode
========================================

Enter your questions about the literature.
Type 'quit' or 'exit' to exit.

IRIS> What machine learning methods are used in neutrino experiments?
----------------------------------------
Question: What machine learning methods are used in neutrino experiments?
----------------------------------------

Answer:
根据搜索到的相关文献，中微子实验中使用的机器学习方法包括：

1. 卷积神经网络（CNN）：用于中微子探测器中的信号重建和模式识别...
（详细回答...）

IRIS> What are the latest findings from JUNO?
----------------------------------------
Question: What are the latest findings from JUNO?
----------------------------------------
...
```

#### 6.2.2 单次查询

```bash
python scripts/iris_query.py "What are the key differences between liquid and scintillator detectors?"
```

#### 6.2.3 查看可用更新

```bash
python scripts/iris_query.py --list-updates
```

输出示例：
```
Available Updates:

  update_2026_03_15_1200
    Time: 2026-03-15T12:00:00
    New papers: 8

  update_2026_03_15_1400
    Time: 2026-03-15T14:00:00
    New papers: 5
```

#### 6.2.4 查询特定更新

```bash
python scripts/iris_query.py --update update_2026_03_15_1200 "What is the energy resolution threshold?"
```

### 6.3 命令行选项

```bash
# 查看帮助
python scripts/iris_query.py --help

# 使用自定义配置
python scripts/iris_query.py --config custom_config.yaml

# 指定数据库路径
python scripts/iris_query.py --database /path/to/database

# 指定更新目录
python scripts/iris_query.py --update update_2026_03_15_1200

# 显式指定索引文件
python scripts/iris_query.py --chunks /path/to/chunks.jsonl --index /path/to/index.index
```

---

## 7. 实现细节说明

### 7.1 arXiv 服务 (`services/arxiv_service.py`)

#### 核心功能

1. **论文搜索** (`search_papers`)
   - 使用 `arxiv` Python 库查询 arXiv API
   - 支持多关键词 OR 搜索
   - 按提交日期排序

2. **论文过滤** (`filter_papers`)
   - 重复检测：基于 `entry_id`
   - 评论文章过滤：检测标题和摘要中的关键词（review, survey, overview）

3. **PDF 下载** (`download_pdf`, `download_pdfs`)
   - 使用 `requests` 库下载 PDF
   - 支持批量下载
   - 错误处理和重试

#### 关键代码

```python
# 构建 OR 查询
query_parts = []
for item in self.keywords:
    if " " in item:
        query_parts.append('"' + item + '"')  # 多词关键词加引号
    else:
        query_parts.append(item)
query = " OR ".join(query_parts)

# 搜索 arXiv
search = arxiv.Search(
    query=query,
    max_results=self.max_results,
    sort_by=self.sort_by
)
```

### 7.2 索引服务 (`services/index_service.py`)

#### 核心功能

1. **调用 UltraRAG** (`chunk_and_index`)
   - 创建运行时参数文件
   - 执行 `ultrarag run` 命令
   - 处理 UltraRAG 输出

2. **参数模板替换**
   - 从模板文件创建运行时参数
   - 替换占位符（如 `__RAW_PDF_DIR__`）
   - 支持模型路径、输出路径等配置

#### UltraRAG 工作流

```
PDF 文件夹
    ↓
UltraRAG offline_build_index_watch.yaml
    ↓
[corpus.parse_file_path] → 文本提取
    ↓
[corpus.chunk_backend: sentence] → 句子切分
    ↓
[retriever.model_name_or_path] → Qwen3-Embedding-0.6B
    ↓
[retriever.index_backend_configs.faiss] → FAISS 索引构建
    ↓
chunks.jsonl + index.index
```

#### 关键代码

```python
# 创建运行时参数文件
replacements = {
    "RAW_PDF_DIR": str(pdf_dir.absolute()),
    "CHUNKS_OUTPUT_PATH": str(chunks_output.absolute()),
    "INDEX_OUTPUT_PATH": str(index_output.absolute()),
    "EMBEDDING_MODEL_PATH": self.embedding_model,
}

# 替换模板中的占位符
with open(template_path, "r") as f:
    content = f.read()
for key, value in replacements.items():
    content = content.replace(f"__{key}__", str(value))
```

### 7.3 QA 服务 (`services/qa_service.py`)

#### 核心功能

1. **知识库查询** (`query_knowledge_base`)
   - 支持单问题和批量问题
   - 通过 vLLM API 调用 Llama3B
   - 使用 UltraRAG 的检索管道

2. **vLLM 服务健康检查** (`check_vllm_service`)
   - 检查 vLLM 服务是否就绪
   - 支持超时设置

3. **答案提取** (`_extract_answers`)
   - 从 UltraRAG 输出 JSON 中提取答案
   - 映射问题和答案

#### RAG 检索增强生成流程

```
用户问题
    ↓
[retriever] → 向量相似度检索 (Qwen0.3B)
    ↓
[reranker] → 结果重排序 (MiniCPM-Reranker)
    ↓
Top-k 相关 chunks
    ↓
[generation] → Llama3B 生成回答
    ↓
最终答案
```

#### 关键代码

```python
# 检查 vLLM 服务状态
health_url = self.vllm_base_url.rstrip("/").replace("/v1", "") + "/v1/models"
response = requests.get(health_url, timeout=5)
if response.status_code == 200:
    return True

# 调用 UltraRAG QA 流水线
result = subprocess.run(
    [ultrarag_cmd, "run", str(pipeline_yaml)],
    cwd=self.ultrarag_path,
    capture_output=True,
    env=env
)
```

### 7.4 邮件服务 (`services/email_service.py`)

#### 核心功能

1. **邮件发送** (`send_notification`)
   - 支持 SMTP/TLS
   - 支持纯文本和 HTML 格式
   - 支持附件（可扩展）

2. **更新通知** (`send_update_notification`)
   - 自动生成更新摘要
   - 包含新论文列表
   - 包含摘要和知识日志
   - HTML 格式化

#### 邮件内容结构

```python
# MIME 多部分邮件
message = MIMEMultipart("alternative")
message["Subject"] = self.subject_prefix + subject
message["From"] = self.sender
message["To"] = self.receiver

# 添加纯文本部分
text_part = MIMEText(content, "plain", "utf-8")
message.attach(text_part)

# 添加 HTML 部分
html_part = MIMEText(html_content, "html", "utf-8")
message.attach(html_part)
```

### 7.5 主编排器 (`scripts/run_update_cycle.py`)

#### 完整更新流程

```
开始
  ↓
[Step 1] 创建更新文件夹
  ↓
[Step 2] 加载现有论文条目（重复检测）
  ↓
[Step 3] 初始化 arXiv 服务
  ↓
[Step 4] 搜索 arXiv
  ↓
[Step 5] 过滤论文（重复、评论）
  ↓
[Step 6] 下载 PDF
  ↓
[Step 7] 保存元数据
  ↓
[Step 8] 生成更新日志
  ↓
[Step 9] 构建 UltraRAG 索引（如果有新论文）
  ↓
[Step 10] 生成摘要（如果索引成功）
  ↓
[Step 11] 生成摘要和知识日志
  ↓
[Step 12] 发送邮件通知
  ↓
完成
```

#### 摘要生成格式

每篇论文生成以下结构：

```markdown
## [论文标题]

**Authors:** 作者列表
**arXiv ID:** 论文 ID

### Abstract Summary
摘要内容...

### Key Information

**Main Problem:**
问题描述的答案...

**Key Contributions:**
关键贡献的答案...

**Methods:**
方法技术的答案...

**Important Concepts:**
重要概念的答案...

**Research Directions:**
研究方向的答案...
```

---

## 8. 故障排除

### 8.1 常见问题

#### 问题 1: vLLM 服务不可用

**症状**：
```
[ERROR] vLLM service is not available
```

**解决方案**：

1. 确认 vLLM 服务正在运行：
   ```bash
   ps aux | grep vllm
   curl http://127.0.0.1:65504/v1/models
   ```

2. 重新启动 vLLM 服务：
   ```bash
   cd /home/NagaiYoru/LLM_tuning/UltraRAG
   source .venv/bin/activate
   python -m vllm.entrypoints.openai.api_server \
       --model /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct \
       --host 127.0.0.1 \
       --port 65504 \
       --max-model-len 8192
   ```

3. 检查配置文件中的 vLLM 地址是否正确

#### 问题 2: UltraRAG 命令找不到

**症状**：
```
[ERROR] UltraRAG run failed with code 127
ultrarag: command not found
```

**解决方案**：

1. 检查 UltraRAG 是否正确安装：
   ```bash
   cd /home/NagaiYoru/LLM_tuning/UltraRAG
   ls .venv/bin/ultrarag
   ```

2. 确认虚拟环境已激活：
   ```bash
   source /home/NagaiYoru/LLM_tuning/UltraRAG/.venv/bin/activate
   which ultrarag
   ```

3. 检查配置文件中的 UltraRAG 路径

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

3. 下载缺失的模型（如果需要）

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

3. 查看详细日志：
   ```bash
   tail -f logs/iris_*.log
   ```

#### 问题 5: 索引构建失败

**症状**：
```
[ERROR] UltraRAG run failed with code 1
```

**解决方案**：

1. 检查 UltraRAG 日志：
   ```bash
   tail -f ~/LLM_tuning/UltraRAG/logs/*.log
   ```

2. 手动运行 UltraRAG 进行调试：
   ```bash
   cd /home/NagaiYoru/LLM_tuning/UltraRAG
   source .venv/bin/activate
   ultrarag run pipelines/offline_build_index_watch.yaml
   ```

3. 检查 GPU 可用性：
   ```bash
   nvidia-smi
   ```

### 8.2 调试方法

#### 启用调试日志

```bash
# 运行时指定调试级别
python scripts/run_update_cycle.py --log-level DEBUG

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

# 查看完整日志
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

1. 备份当前索引：
   ```bash
   cp -r index_storage index_storage_backup
   ```

2. 删除旧索引：
   ```bash
   rm -rf index_storage
   ```

3. 重新运行更新周期（将重建索引）

---

## 附录

### A.1 命令快速参考

| 命令 | 描述 |
|------|------|
| `./start_IRIS.sh` | 运行单次更新 |
| `./start_IRIS.sh --daemon` | 守护进程模式 |
| `python scripts/run_update_cycle.py` | 直接运行更新脚本 |
| `python scripts/iris_query.py -i` | 交互式查询 |
| `python scripts/iris_query.py "问题"` | 单次查询 |
| `python scripts/iris_query.py -l` | 列出更新 |

### A.2 配置参数速查

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `update.interval_hours` | 2 | 更新间隔（小时） |
| `arxiv.max_results_per_keyword` | 20 | 每关键词最大结果 |
| `storage.database_root` | `~/research/IRIS_papers` | 数据库路径 |
| `models.embedding_model_path` | Qwen3-Embedding-0.6B | 嵌入模型 |
| `models.llm_model_path` | Llama-3.2-3B-Instruct | 生成模型 |
| `vllm.base_url` | `http://127.0.0.1:65504/v1` | vLLM 服务地址 |

### A.3 相关链接

- [arXiv](https://arxiv.org/)
- [UltraRAG 文档](https://github.com/OpenBMB/UltraRAG)
- [vLLM 文档](https://docs.vllm.ai/)
- [Qwen 模型](https://huggingface.co/Qwen)
- [Llama 模型](https://huggingface.co/meta-llama)

---

## 总结

IRIS 提供了一个完整的自动化文献监控和知识管理解决方案。通过本教程，您应该能够：

1. ✓ 部署完整的 IRIS 系统
2. ✓ 配置所有必要的参数
3. ✓ 运行自动更新周期
4. ✓ 使用交互式查询接口
5. ✓ 理解系统的实现原理
6. ✓ 处理常见问题和故障

如有问题，请查看日志文件或参考故障排除章节。
