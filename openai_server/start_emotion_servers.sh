#!/bin/bash
# Start Emotion Servers with Tensor Parallel Support
# Usage: ./start_emotion_servers.sh [start|stop|status|restart]

# Configuration
MODEL_PATH="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL=2
GPU_MEMORY_UTIL=0.7
BATCH_SIZE=32
MAX_NUM_SEQS=32

# Server configurations
HAPPINESS_PORT=8000
ANGER_PORT=8001

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
    local log_file="$LOG_DIR/${emotion}_server.log"
    
    print_status "Starting $emotion server on port $port..."
    
    # Check if port is already in use
    if check_server_health $port; then
        print_warning "$emotion server already running on port $port"
        return 0
    fi
    
    # Change to project root directory
    cd /data/home/jjl7137/LLM_EmoBehav_game_theory
    
    # Start server in background
    nohup python -m openai_server \
        --model "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --emotion "$emotion" \
        --port $port \
        --tensor_parallel_size $TENSOR_PARALLEL \
        --gpu_memory_utilization $GPU_MEMORY_UTIL \
        --batch_size $BATCH_SIZE \
        --max_num_seqs $MAX_NUM_SEQS \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo $pid > "$LOG_DIR/${emotion}_server.pid"
    
    print_status "Started $emotion server with PID $pid, waiting for initialization..."
    
    # Wait for server to be ready (up to 120 seconds)
    local wait_time=0
    local max_wait=120
    
    while [ $wait_time -lt $max_wait ]; do
        if check_server_health $port; then
            print_success "$emotion server ready on port $port (took ${wait_time}s)"
            return 0
        fi
        
        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            print_error "$emotion server process died during startup"
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

show_status() {
    echo
    echo "=== Emotion Server Status ==="
    echo
    
    # Check happiness server
    if check_server_health $HAPPINESS_PORT; then
        print_success "Happiness server: RUNNING (port $HAPPINESS_PORT)"
        local response=$(curl -s "http://localhost:$HAPPINESS_PORT/health" | jq -r '.current_emotion // "unknown"' 2>/dev/null)
        echo "  Emotion: $response"
    else
        print_error "Happiness server: NOT RUNNING (port $HAPPINESS_PORT)"
    fi
    
    # Check anger server
    if check_server_health $ANGER_PORT; then
        print_success "Anger server: RUNNING (port $ANGER_PORT)"
        local response=$(curl -s "http://localhost:$ANGER_PORT/health" | jq -r '.current_emotion // "unknown"' 2>/dev/null)
        echo "  Emotion: $response"
    else
        print_error "Anger server: NOT RUNNING (port $ANGER_PORT)"
    fi
    
    echo
    echo "=== GPU Memory Usage ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r gpu_id mem_used mem_total gpu_util; do
        echo "  GPU $gpu_id: ${mem_used}MB / ${mem_total}MB (${gpu_util}% util)"
    done
    echo
}

start_all_servers() {
    print_status "Starting all emotion servers with tensor_parallel=$TENSOR_PARALLEL..."
    echo "Model: $MODEL_PATH"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
    echo "Batch Size: $BATCH_SIZE"
    echo "Max Sequences: $MAX_NUM_SEQS"
    echo
    
    # Start happiness server first
    if start_server "happiness" $HAPPINESS_PORT; then
        print_success "Happiness server started successfully"
    else
        print_error "Failed to start happiness server"
        return 1
    fi
    
    # Wait a bit before starting second server
    sleep 5
    
    # Start anger server
    if start_server "anger" $ANGER_PORT; then
        print_success "Anger server started successfully"
    else
        print_error "Failed to start anger server"
        print_warning "Happiness server is still running"
        return 1
    fi
    
    echo
    print_success "All servers started successfully!"
    show_status
}

stop_all_servers() {
    print_status "Stopping all emotion servers..."
    
    stop_server "happiness" $HAPPINESS_PORT
    stop_server "anger" $ANGER_PORT
    
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
    echo "Emotion Server Manager"
    echo "====================="
    echo
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start     Start both happiness and anger servers"
    echo "  stop      Stop all emotion servers"
    echo "  restart   Restart all emotion servers"
    echo "  status    Show server status and GPU usage"
    echo "  logs      Show recent logs from both servers"
    echo "  help      Show this help message"
    echo
    echo "Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  Tensor Parallel: $TENSOR_PARALLEL"
    echo "  GPU Memory Util: $GPU_MEMORY_UTIL"
    echo "  Ports: Happiness=$HAPPINESS_PORT, Anger=$ANGER_PORT"
    echo
    echo "Requirements:"
    echo "  - CUDA-capable GPUs (at least 2 for tensor_parallel=2)"
    echo "  - Sufficient VRAM for model and emotion vectors"
    echo "  - conda environment 'llm_fresh' activated"
}

show_logs() {
    echo "=== Recent Happiness Server Logs ==="
    if [ -f "$LOG_DIR/happiness_server.log" ]; then
        tail -20 "$LOG_DIR/happiness_server.log"
    else
        echo "No happiness server log found"
    fi
    
    echo
    echo "=== Recent Anger Server Logs ==="
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
    
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
    if [ "$gpu_count" -lt "$TENSOR_PARALLEL" ]; then
        print_error "Need at least $TENSOR_PARALLEL GPUs for tensor_parallel=$TENSOR_PARALLEL"
        print_error "Found: $gpu_count GPUs"
        exit 1
    fi
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model path not found: $MODEL_PATH"
        exit 1
    fi
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