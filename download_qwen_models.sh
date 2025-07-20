#!/bin/bash

# List of Qwen3 AWQ models to download
MODELS=(
  "Qwen/Qwen3-4B-AWQ"
  "Qwen/Qwen3-8B-AWQ"
  "Qwen/Qwen3-14B-AWQ"
  "Qwen/Qwen3-32B-AWQ"
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