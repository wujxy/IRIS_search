#!/bin/bash

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --served-model-name llama3-3b-instruct \
    --model /home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 65504 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.84 \
    --tensor-parallel-size 1 \
    --enforce-eager \