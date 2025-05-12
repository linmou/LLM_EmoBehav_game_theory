#!/bin/bash

# List of Qwen2.5 models to download
MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)

# Base directory for saving models
BASE_DIR="$HOME/huggingface_models/Qwen"

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")
  LOCAL_DIR="$BASE_DIR/$MODEL_NAME"
  echo "Downloading $MODEL to $LOCAL_DIR ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" --local-dir "$LOCAL_DIR"
done

echo "All models downloaded." 