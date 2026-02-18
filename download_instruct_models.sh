#!/bin/bash

# Instruct-tuned models for emotion/game-theory experiments
MODELS=(
  # Google Gemma 3
  "google/gemma-3-270m-it"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"

  # Microsoft Phi
  "microsoft/Phi-3.5-mini-instruct"
  "microsoft/Phi-4-mini-instruct"

  # Meta Llama 3.2
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"

  # DeepSeek
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

  # Qwen2.5 Instruct (<7B)
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"

  # Qwen3 (<7B, thinking/non-thinking via system prompt)
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
)

BASE_DIR="${USER_HOME:-/home/jjl7137}/huggingface_models"

for MODEL in "${MODELS[@]}"; do
  LOCAL_DIR="$BASE_DIR/$MODEL"
  echo "Downloading $MODEL to $LOCAL_DIR ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" --local-dir "$LOCAL_DIR"
done

echo "All models downloaded."
