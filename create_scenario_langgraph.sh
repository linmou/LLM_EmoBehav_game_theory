#!/bin/bash

# =============================================================================
# Simple Script for LangGraph Scenario Creation (CLI args only)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/data_creation/create_scenario_langgraph.py"

# Default parameters
# Path to the persona jobs file (input data for personas)
PERSONA_JOBS_FILE="${SCRIPT_DIR}/data_creation/persona_jobs_all.jsonl"
# Name of the game scenario to generate
GAME_NAME="Endowment_Effect"
# Number of personas to process (limit for batch run)
NUM_PERSONAS=999999
# Number of personas to process in each batch (parallel jobs)
BATCH_SIZE=15
# Timeout (in seconds) for each individual scenario creation task
TASK_TIMEOUT=300
# Timeout (in seconds) for each batch of scenario creation tasks
BATCH_TIMEOUT=1800
# Maximum number of retries for a failed scenario creation task, 0 means no retries, 1 means so sample will try twice (1 start try + 1 retry)
MAX_RETRIES=1
# Delay (in seconds) between retries for failed tasks
RETRY_DELAY=5
# Maximum number of refinement iterations for scenario creation
MAX_ITERATIONS=8
# LLM model name to use for scenario generation
LLM_MODEL="gpt-4.1"
# Temperature for LLM when proposing scenarios (higher = more creative)
LLM_TEMPERATURE_PROPOSE=0.7
# Temperature for LLM when verifying scenarios (lower = more deterministic)
LLM_TEMPERATURE_VERIFY=0.3
# Temperature for LLM when validating payoffs (lowest = most deterministic)
LLM_TEMPERATURE_PAYOFF=0.1
# Whether to use Azure OpenAI (true/false)
AZURE_MODE=true
# Enable debug mode for more detailed logs (true/false)
DEBUG_MODE=false
# Enable auto-debug mode (true/false)
AUTO_DEBUG=false
# Output directory for generated scenarios and histories
OUTPUT_BASE_DIR="${SCRIPT_DIR}/data_creation/scenario_creation/langgraph_creation"
# Resume from previous run if available (true/false)
RESUME=true
# Enable verbose output (true/false)
VERBOSE=false
# Which verification nodes to use in the graph.
# Options: "narrative", "preference_order", "pay_off"
VERIFICATION_NODES=("narrative" "pay_off")

validate_inputs() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo "Error: Python script not found at $PYTHON_SCRIPT"
        exit 1
    fi
    if [[ ! -f "$PERSONA_JOBS_FILE" ]]; then
        echo "Error: Persona jobs file not found at $PERSONA_JOBS_FILE"
        exit 1
    fi
    if ! [[ "$NUM_PERSONAS" =~ ^[0-9]+$ ]] || [[ "$NUM_PERSONAS" -le 0 ]]; then
        echo "Error: NUM_PERSONAS must be a positive integer"
        exit 1
    fi
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -le 0 ]]; then
        echo "Error: BATCH_SIZE must be a positive integer"
        exit 1
    fi
    if ! [[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$MAX_ITERATIONS" -le 0 ]]; then
        echo "Error: MAX_ITERATIONS must be a positive integer"
        exit 1
    fi
    if [[ "$AZURE_MODE" != "true" && "$AZURE_MODE" != "false" ]]; then
        echo "Error: AZURE_MODE must be 'true' or 'false'"
        exit 1
    fi
}


main() {
    echo "LangGraph Scenario Creation Script"
    echo "=================================="
    validate_inputs
    echo "Running with arguments:" 
    echo "  --persona-jobs-file $PERSONA_JOBS_FILE" 
    echo "  --game-name $GAME_NAME" 
    echo "  --num-personas $NUM_PERSONAS" 
    echo "  --batch-size $BATCH_SIZE" 
    echo "  --task-timeout $TASK_TIMEOUT" 
    echo "  --batch-timeout $BATCH_TIMEOUT" 
    echo "  --max-retries $MAX_RETRIES" 
    echo "  --retry-delay $RETRY_DELAY" 
    echo "  --max-iterations $MAX_ITERATIONS" 
    echo "  --llm-model $LLM_MODEL" 
    echo "  --llm-temp-propose $LLM_TEMPERATURE_PROPOSE" 
    echo "  --llm-temp-verify $LLM_TEMPERATURE_VERIFY" 
    echo "  --llm-temp-payoff $LLM_TEMPERATURE_PAYOFF" 
    echo "  --azure-mode $AZURE_MODE" 
    echo "  --output-dir $OUTPUT_BASE_DIR" 
    echo "  --verification-nodes ${VERIFICATION_NODES[*]}"
    [[ "$DEBUG_MODE" == "true" ]] && echo "  --debug"
    [[ "$RESUME" == "true" ]] && echo "  --resume"
    [[ "$VERBOSE" == "true" ]] && echo "  --verbose"
    python -m data_creation.create_scenario_langgraph \
        --persona-jobs-file "$PERSONA_JOBS_FILE" \
        --game-name "$GAME_NAME" \
        --num-personas "$NUM_PERSONAS" \
        --batch-size "$BATCH_SIZE" \
        --task-timeout "$TASK_TIMEOUT" \
        --batch-timeout "$BATCH_TIMEOUT" \
        --max-retries "$MAX_RETRIES" \
        --retry-delay "$RETRY_DELAY" \
        --max-iterations "$MAX_ITERATIONS" \
        --llm-model "$LLM_MODEL" \
        --llm-temp-propose "$LLM_TEMPERATURE_PROPOSE" \
        --llm-temp-verify "$LLM_TEMPERATURE_VERIFY" \
        --llm-temp-payoff "$LLM_TEMPERATURE_PAYOFF" \
        --azure-mode "$AZURE_MODE" \
        --output-dir "$OUTPUT_BASE_DIR" \
        --verification-nodes "${VERIFICATION_NODES[@]}" \
        $( [[ "$DEBUG_MODE" == "true" ]] && echo "--debug" ) \
        $( [[ "$AUTO_DEBUG" == "true" ]] && echo "--auto-debug" ) \
        $( [[ "$RESUME" == "true" ]] && echo "--resume" ) \
        $( [[ "$VERBOSE" == "true" ]] && echo "--verbose" )
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 