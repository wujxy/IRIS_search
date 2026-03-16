#!/bin/bash

ULTRARAG="/home/NagaiYoru/LLM_tuning/UltraRAG"
source $ULTRARAG/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen3-embedding-0.6b \
    --model /home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 65503 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15 \
    --tensor-parallel-size 1 \
    --enforce-eager
