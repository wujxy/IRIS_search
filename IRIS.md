# IRIS.md
# IRIS – Intelligent Research Information System

IRIS is an automated AI-powered literature monitoring and knowledge extraction system.

The system periodically searches **arXiv**, downloads new papers, processes them into a **RAG knowledge base**, generates **automatic summaries**, and sends **email updates** to the user.

All models are **locally deployed**.

---

# 1 System Overview

IRIS integrates the following components:

- arXiv literature search
- automatic PDF downloading
- document chunking
- vector embedding indexing
- retrieval-based question answering
- paper summarization
- knowledge update reports
- email notifications

Pipeline:
arXiv Search
↓
Paper Metadata
↓
PDF Download
↓
UltraRAG Chunk + Index
↓
Vector Knowledge Base
↓
LLM Summary
↓
Knowledge Log
↓
Email Notification

---

# 2 Existing Reference Implementations

The user already has some working scripts.  
**These paths are only references for how services are started and how models are called.**

Claude **should not depend strictly on these paths**, but can reuse the logic if helpful.

## 2.1 arXiv Search Script (Reference)

Example path:

```
~/LLM_tuning/IRIS_search/arxiv_search.py
```

Function:

- query arXiv with keywords
- retrieve paper metadata
- download PDF

Claude should implement a **service wrapper** such as:

```
services/arxiv_service.py
```

which internally calls similar logic.

The arXiv service should support:

```
search_papers(keywords, max_results)
download_pdf(pdf_url, save_path)
```

---

## 2.2 UltraRAG Indexing Script (Reference)

Example path:

```
~/LLM_tuning/AS_serve_ultrarag_based/auto_scripts/index_watch_service.py
```

Function:

```
PDF
 → chunk
 → embedding
 → vector index
```

Embedding model:

```
Qwen0.3B
```

Claude should implement a wrapper service:

```
services/index_service.py
```

This service should call UltraRAG to perform:

```
chunk_documents()
build_embedding_index()
update_index()
```

The exact UltraRAG start command may resemble:

```
python index_watch_service.py
```

but Claude should treat this as **implementation reference only**.

---

# 3 Configurable Parameters

All parameters must be stored in:

```
configs/config.yaml
```

Example:

```yaml
update:
  interval_hours: 2

arxiv:
  keywords:
    - neutrino
    - particle detector
    - machine learning physics
  max_results_per_keyword: 20

storage:

  # IMPORTANT
  # Literature database is NOT inside the IRIS project directory
  database_root: /path/to/user_defined_database

ultrarag:
  ultrarag_path: /path/to/ultrarag
  index_storage: /path/to/index_storage

models:
  embedding_model_path: /models/Qwen0.3B
  llm_model_path: /models/Llama3B

qa:
  question_set_path: ./configs/questions.txt

email:
  sender: example@gmail.com
  smtp_server: smtp.gmail.com
  smtp_port: 587
  password: your_password
  receiver: user@email.com
```

# 4 Literature Database Structure
The literature database is not stored in the IRIS project directory.

The user defines the database path manually.

Example:
/research_database/IRIS_paperss
Every update creates a timestamped folder.

Example:
IRIS_database/

├── update_2026_03_15_1200
│
│   ├── pdfs
│   │   ├── paper1.pdf
│   │   ├── paper2.pdf
│   │
│   ├── logs
│   │   ├── summary_log.md
│   │   ├── knowledge_log.md
│   │
│   ├── metadata.json
│   ├── update_log.md
│
├── update_2026_03_15_1400
│
│   ├── pdfs
│   ├── logs
│   ├── metadata.json
│   ├── update_log.md

# 5 Metadata Format
Each update folder must contain:
metadata.json
Metadata example:
```json
{
  "entry_id": "http://arxiv.org/abs/1234.5678v1",
  "updated": "2026-03-14T12:00:00Z",
  "published": "2026-03-13T18:30:00Z",
  "title": "Example Paper Title",
  "authors": ["Author A", "Author B"],
  "summary": "Paper abstract text...",
  "comment": "12 pages, 5 figures",
  "journal_ref": "PhysRevD.102.012345",
  "doi": "10.1234/example.doi",
  "primary_category": "hep-ex",
  "categories": [
    "hep-ex",
    "hep-ph"
  ],
  "pdf_url": "http://arxiv.org/pdf/1234.5678v1"
}
```
Multiple papers should be stored as a list:
```json
[
  {...},
  {...}
]
```

# 6 Paper Filtering Rules

IRIS should not include review papers.

Filtering rule:

If title or abstract contains:
```
review
survey
overview
```
then mark as:
```
paper_type = review
```
and skip downloading the PDF.

# 7 Duplicate Detection

Each update must check whether a paper already exists in previous updates.

Use identifier:
```
entry_id
```
Logic:
```
if entry_id already exists in database:

    mark as duplicate
    do NOT download pdf
    still record in update log
```

# 8 Update Cycle Workflow
Main orchestrator:

scripts/run_update_cycle.py

Pipeline:

1 search arxiv

2 filter review papers

3 detect duplicates

4 download new PDFs

5 generate metadata.json

6 call UltraRAG indexing

7 generate summaries

8 generate knowledge log

9 send email

# 9 Paper Summarization

For each new paper:
```
retrieve paper chunks
↓
generate summary
↓
extract key contributions
↓
extract important methods
```

Output example:
```
Title

Summary

Key Contributions

Methods

Important Concepts

Possible Research Directions
```
Saved in:

```
logs/summary_log.md
```

# 10 QA Query Interface

Script:

scripts/iris_query.py

Example usage:

python iris_query.py "What machine learning methods are used in neutrino experiments?"

Pipeline:

question
↓
retrieve chunks
↓
LLM generation
↓
answer
# 11 Email Notification

After each update cycle:

Send email containing:

new papers list
summaries
knowledge update

Module:

scripts/email_sender.py
# 12 Startup Script

Create startup script:

start_IRIS.sh

Example:

#!/bin/bash

echo "Starting IRIS literature monitoring service..."

source ~/.bashrc

cd /path/to/IRIS

python scripts/run_update_cycle.py

This script allows the user to manually start the IRIS update service.

Future upgrades may include:

cron job
systemd service
# 13 Expected Final Result

IRIS will function as a personal AI research assistant:

arXiv monitoring
↓
automatic literature collection
↓
AI indexing
↓
knowledge extraction
↓
literature QA
↓
email updates

The system continuously maintains a self-updating literature knowledge base.