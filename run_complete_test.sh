#!/bin/bash
#
# Complete compatibility test runner script
#
# This script:
# 1. Activates the conda environment
# 2. Runs the complete compatibility test with server startup
# 3. Uses the same model path as in the init script
#
# Usage:
#     ./run_complete_test.sh
#     or
#     bash run_complete_test.sh
#

set -e  # Exit on any error

echo "🔍 Complete LangGraph & AG2 Compatibility Test Runner"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda command not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment: llm_fresh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_fresh

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'llm_fresh'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

echo "✅ Conda environment activated"

# Set model path (same as in init_openai_server.sh)
MODEL_PATH="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME="qwen2.5-0.5B-anger"
EMOTION="anger"

echo ""
echo "🚀 Starting Complete Compatibility Test"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Emotion: $EMOTION"
echo ""

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Warning: Model path does not exist: $MODEL_PATH"
    echo "Please update the MODEL_PATH variable in this script to point to your model."
    echo ""
    echo "Available models in /data/home/jjl7137/huggingface_models/:"
    ls -la /data/home/jjl7137/huggingface_models/ 2>/dev/null || echo "Directory not accessible"
    echo ""
    echo "Continuing anyway (model path might be valid for your system)..."
fi

# Install required dependencies if needed
echo "🔧 Checking dependencies..."
pip install psutil requests openai > /dev/null 2>&1 || echo "⚠️ Some dependencies may be missing"

# Run the complete test
python run_complete_compatibility_test.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --emotion "$EMOTION" \
    --output "compatibility_test_results.json"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Complete compatibility test finished successfully!"
    echo "📄 Results saved to: compatibility_test_results.json"
else
    echo "❌ Compatibility test failed with exit code: $exit_code"
    echo "📄 Partial results may be saved to: compatibility_test_results.json"
fi

echo ""
echo "📋 Summary:"
echo "  - Server startup: Automated"
echo "  - Basic OpenAI compatibility: Tested"
echo "  - LangGraph compatibility: Tested (with message format fixes)"
echo "  - AG2 (AutoGen) compatibility: Tested"
echo "  - Advanced features: Tested"

exit $exit_code 