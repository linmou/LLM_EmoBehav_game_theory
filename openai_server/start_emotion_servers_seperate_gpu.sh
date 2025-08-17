#!/bin/bash
# Start Emotion Servers on Different CUDA Devices
# Usage: ./start_emotion_servers_single_gpu.sh [start|stop|status|restart]

# Configuration
MODEL_PATH="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
GPU_MEMORY_UTIL=0.5
BATCH_SIZE=32
MAX_NUM_SEQS=32

# Server configurations with specific CUDA devices (using fresh GPUs)
HAPPINESS_PORT=8000
HAPPINESS_GPU=2
ANGER_PORT=8001
ANGER_GPU=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="logs"
mkdir -p $LOG_DIR

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

check_server_health() {
    local port=$1
    local timeout=5
    
    if curl -s --max-time $timeout "http://localhost:$port/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

start_server() {
    local emotion=$1
    local port=$2
    local gpu_id=$3
    local log_file="$LOG_DIR/${emotion}_server.log"
    
    print_status "Starting $emotion server on GPU $gpu_id, port $port..."
    
    # Check if port is already in use
    if check_server_health $port; then
        print_warning "$emotion server already running on port $port"
        return 0
    fi
    
    # Change to project root directory
    cd /data/home/jjl7137/LLM_EmoBehav_game_theory
    
    # Start server with specific CUDA device and reduced memory footprint
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python -m openai_server \
        --model "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --emotion "$emotion" \
        --port $port \
        --gpu_memory_utilization $GPU_MEMORY_UTIL \
        --batch_size $BATCH_SIZE \
        --max_num_seqs $MAX_NUM_SEQS \
        --disable_batching \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo $pid > "$LOG_DIR/${emotion}_server.pid"
    
    print_status "Started $emotion server with PID $pid on GPU $gpu_id, waiting for initialization..."
    
    # Wait for server to be ready (up to 90 seconds)
    local wait_time=0
    local max_wait=90
    
    while [ $wait_time -lt $max_wait ]; do
        if check_server_health $port; then
            print_success "$emotion server ready on port $port, GPU $gpu_id (took ${wait_time}s)"
            return 0
        fi
        
        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            print_error "$emotion server process died during startup"
            echo "Last 20 lines of log:"
            tail -20 "$log_file"
            return 1
        fi
        
        sleep 2
        wait_time=$((wait_time + 2))
        
        if [ $((wait_time % 20)) -eq 0 ]; then
            print_status "Still waiting for $emotion server... (${wait_time}s elapsed)"
        fi
    done
    
    print_error "$emotion server failed to start within ${max_wait}s"
    echo "Last 20 lines of log:"
    tail -20 "$log_file"
    return 1
}

stop_server() {
    local emotion=$1
    local port=$2
    local pid_file="$LOG_DIR/${emotion}_server.pid"
    
    print_status "Stopping $emotion server..."
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 2
            
            # Force kill if still running
            if kill -0 $pid 2>/dev/null; then
                print_warning "Force killing $emotion server (PID: $pid)"
                kill -9 $pid
            fi
            
            print_success "Stopped $emotion server (PID: $pid)"
        else
            print_warning "$emotion server was not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        # Try to find and kill by port
        local pid=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$pid" ]; then
            print_status "Found process on port $port (PID: $pid), killing..."
            kill $pid
            sleep 2
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid
            fi
            print_success "Stopped process on port $port"
        else
            print_warning "No $emotion server found running on port $port"
        fi
    fi
}

show_gpu_status() {
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r gpu_id name mem_used mem_total gpu_util temp; do
        local mem_percent=$((mem_used * 100 / mem_total))
        echo "  GPU $gpu_id ($name): ${mem_used}MB/${mem_total}MB (${mem_percent}%) | ${gpu_util}% util | ${temp}°C"
    done
    echo
}

show_status() {
    echo
    echo "=== Emotion Server Status ==="
    echo
    
    # Check happiness server
    if check_server_health $HAPPINESS_PORT; then
        print_success "Happiness server: RUNNING (port $HAPPINESS_PORT, GPU $HAPPINESS_GPU)"
        local response=$(curl -s "http://localhost:$HAPPINESS_PORT/health" | jq -r '.current_emotion // "unknown"' 2>/dev/null)
        echo "  Emotion: $response"
        local pid_file="$LOG_DIR/happiness_server.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            echo "  PID: $pid"
        fi
    else
        print_error "Happiness server: NOT RUNNING (port $HAPPINESS_PORT, GPU $HAPPINESS_GPU)"
    fi
    
    # Check anger server
    if check_server_health $ANGER_PORT; then
        print_success "Anger server: RUNNING (port $ANGER_PORT, GPU $ANGER_GPU)"
        local response=$(curl -s "http://localhost:$ANGER_PORT/health" | jq -r '.current_emotion // "unknown"' 2>/dev/null)
        echo "  Emotion: $response"
        local pid_file="$LOG_DIR/anger_server.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            echo "  PID: $pid"
        fi
    else
        print_error "Anger server: NOT RUNNING (port $ANGER_PORT, GPU $ANGER_GPU)"
    fi
    
    echo
    show_gpu_status
}

start_all_servers() {
    print_status "Starting all emotion servers on separate GPUs..."
    echo "Model: $MODEL_PATH"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
    echo "Batch Size: $BATCH_SIZE"
    echo "Max Sequences: $MAX_NUM_SEQS"
    echo "Happiness GPU: $HAPPINESS_GPU, Anger GPU: $ANGER_GPU"
    echo
    
    # Start both servers simultaneously since they use different GPUs
    print_status "Starting both servers in parallel..."
    
    # Start happiness server in background
    (start_server "happiness" $HAPPINESS_PORT $HAPPINESS_GPU) &
    local happiness_job=$!
    
    # Start anger server in background  
    (start_server "anger" $ANGER_PORT $ANGER_GPU) &
    local anger_job=$!
    
    # Wait for both to complete
    local happiness_result=0
    local anger_result=0
    
    wait $happiness_job || happiness_result=$?
    wait $anger_job || anger_result=$?
    
    if [ $happiness_result -eq 0 ] && [ $anger_result -eq 0 ]; then
        echo
        print_success "All servers started successfully!"
        show_status
        return 0
    else
        echo
        if [ $happiness_result -ne 0 ]; then
            print_error "Happiness server failed to start"
        fi
        if [ $anger_result -ne 0 ]; then
            print_error "Anger server failed to start"
        fi
        return 1
    fi
}

stop_all_servers() {
    print_status "Stopping all emotion servers..."
    
    # Stop both servers in parallel
    (stop_server "happiness" $HAPPINESS_PORT) &
    (stop_server "anger" $ANGER_PORT) &
    
    # Wait for both to complete
    wait
    
    # Clean up any orphaned processes
    print_status "Cleaning up orphaned processes..."
    pkill -f "openai_server.*emotion" 2>/dev/null || true
    
    print_success "All servers stopped"
}

restart_all_servers() {
    print_status "Restarting all emotion servers..."
    stop_all_servers
    sleep 3
    start_all_servers
}

show_help() {
    echo "Emotion Server Manager (Single GPU per Server)"
    echo "=============================================="
    echo
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start     Start both happiness and anger servers on separate GPUs"
    echo "  stop      Stop all emotion servers"
    echo "  restart   Restart all emotion servers"
    echo "  status    Show server status and GPU usage"
    echo "  logs      Show recent logs from both servers"
    echo "  help      Show this help message"
    echo
    echo "Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  Happiness: GPU $HAPPINESS_GPU, Port $HAPPINESS_PORT"
    echo "  Anger: GPU $ANGER_GPU, Port $ANGER_PORT"
    echo "  GPU Memory Util: $GPU_MEMORY_UTIL"
    echo "  Batch Size: $BATCH_SIZE"
    echo
    echo "Requirements:"
    echo "  - At least 2 CUDA-capable GPUs"
    echo "  - Sufficient VRAM per GPU for model and emotion vectors"
    echo "  - conda environment 'llm_fresh' activated"
}

show_logs() {
    echo "=== Recent Happiness Server Logs (GPU $HAPPINESS_GPU) ==="
    if [ -f "$LOG_DIR/happiness_server.log" ]; then
        tail -20 "$LOG_DIR/happiness_server.log"
    else
        echo "No happiness server log found"
    fi
    
    echo
    echo "=== Recent Anger Server Logs (GPU $ANGER_GPU) ==="
    if [ -f "$LOG_DIR/anger_server.log" ]; then
        tail -20 "$LOG_DIR/anger_server.log"
    else
        echo "No anger server log found"
    fi
}

# Check prerequisites
check_prerequisites() {
    # Check if we're in the right directory
    if [ ! -f "constants.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check conda environment
    if [[ "$CONDA_DEFAULT_ENV" != "llm_fresh" ]]; then
        print_warning "Not in 'llm_fresh' conda environment"
        echo "Run: conda activate llm_fresh"
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA drivers may not be installed."
        exit 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpu_count" -lt 2 ]; then
        print_error "Need at least 2 GPUs for separate emotion servers"
        print_error "Found: $gpu_count GPUs"
        exit 1
    fi
    
    # Validate specific GPU IDs
    if [ "$HAPPINESS_GPU" -ge "$gpu_count" ] || [ "$ANGER_GPU" -ge "$gpu_count" ]; then
        print_error "Invalid GPU IDs. Available GPUs: 0-$((gpu_count-1))"
        exit 1
    fi
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model path not found: $MODEL_PATH"
        exit 1
    fi
    
    # Show GPU info
    echo "Available GPUs:"
    show_gpu_status
}

# Main script logic
main() {
    case "${1:-help}" in
        "start")
            check_prerequisites
            start_all_servers
            ;;
        "stop")
            stop_all_servers
            ;;
        "restart")
            check_prerequisites
            restart_all_servers
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"