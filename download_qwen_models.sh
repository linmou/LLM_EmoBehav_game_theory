#!/bin/bash

# List of Gemma 3 instruct-tuned models to download
MODELS=(
  "google/gemma-3-270m-it"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
)
# Base directory for saving models
BASE_DIR="$HOME/huggingface_models/google"

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")
  LOCAL_DIR="$BASE_DIR/$MODEL_NAME"
  echo "Downloading $MODEL to $LOCAL_DIR ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" --local-dir "$LOCAL_DIR"
done

echo "All models downloaded." 