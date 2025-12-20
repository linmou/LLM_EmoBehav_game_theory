#!/bin/bash

MODELS=(
  # Mamba models
  "state-spaces/mamba-370m-hf"
  "state-spaces/mamba-790m-hf"
  "state-spaces/mamba-1.4b-hf"
  "state-spaces/mamba-2.8b-hf"
  
  # Mamba2 models
  "state-spaces/mamba2-370m"
  "state-spaces/mamba2-780m"
  "state-spaces/mamba2-1.3b"
  "state-spaces/mamba2-2.7b"
  
  # Falcon Mamba
  "tiiuae/falcon-mamba-7b"
  "tiiuae/falcon-mamba-7b-instruct"
  
  # Zamba2 models
  "Zyphra/Zamba2-1.2B-Instruct"
  "Zyphra/Zamba2-2.7B-Instruct"
  "Zyphra/Zamba2-7B-Instruct"
  
  # Bamba (IBM's latest SSM-transformer hybrid)
  "ibm-ai-platform/Bamba-9B-v2"

)
# Base directory for saving models
BASE_DIR="$HOME/huggingface_models"

for MODEL in "${MODELS[@]}"; do
  LOCAL_DIR="$BASE_DIR/$MODEL"
  echo " test $LOCAL_DIR ..."
  conda run -n llm_fresh_mobile python test_vllm_support.py $LOCAL_DIR  
done

echo "All models tested" 
