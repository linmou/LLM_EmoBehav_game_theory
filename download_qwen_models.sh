#!/bin/bash

# List of Gemma 3 instruct-tuned models to download
MODELS=(
  # Mamba models
  #"state-spaces/mamba-370m-hf"
  #"state-spaces/mamba-790m-hf"
  #"state-spaces/mamba-1.4b-hf"
  #"state-spaces/mamba-2.8b-hf"
  
  # Mamba2 models
  #"state-spaces/mamba2-370m"
  #"state-spaces/mamba2-780m"
  #"state-spaces/mamba2-1.3b"
  #"state-spaces/mamba2-2.7b"
  
  # Falcon Mamba
  #"tiiuae/falcon-mamba-7b"
  #"tiiuae/falcon-mamba-7b-instruct"
  
  # Zamba2 models
  #"Zyphra/Zamba2-1.2B-Instruct"
  #"Zyphra/Zamba2-2.7B-Instruct"
  #"Zyphra/Zamba2-7B-Instruct"
  
  # Bamba (IBM's latest SSM-transformer hybrid)
  #"ibm-ai-platform/Bamba-9B-v2"
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/deepseek-vl2-tiny"
  "HuggingFaceTB/SmolLM3-3B"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  "HuggingFaceTB/SmolLM2-360M-Instruct"
  "HuggingFaceTB/SmolLM2-135M-Instruct"
  "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
  "microsoft/Phi-4-mini-flash-reasoning"
  "microsoft/Phi-3.5-vision-instruct"
  "facebook/MobileLLM-R1-950M"
  "facebook/MobileLLM-R1-360M"
  "facebook/MobileLLM-R1-140M"
)
# Base directory for saving models
BASE_DIR="$HOME/huggingface_models"

for MODEL in "${MODELS[@]}"; do
  LOCAL_DIR="$BASE_DIR/$MODEL"
  echo "Downloading $MODEL to $LOCAL_DIR ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" --local-dir "$LOCAL_DIR"
done

echo "All models downloaded." 
