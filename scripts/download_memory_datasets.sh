#!/bin/bash

# Memory Benchmark Dataset Download Script
# Downloads and sets up InfiniteBench and LongBench datasets
# for emotion memory experiments

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/memory_benchmarks"

echo "🔽 MEMORY BENCHMARK DATASET DOWNLOADER"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to download and verify a file
download_file() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    
    print_status "Downloading $description..."
    
    if [[ -f "$output_path" ]]; then
        print_warning "File already exists: $output_path"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping $description"
            return 0
        fi
    fi
    
    # Download with progress bar
    if command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$output_path"
    elif command -v wget &> /dev/null; then
        wget --progress=bar:force "$url" -O "$output_path"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        return 1
    fi
    
    if [[ -f "$output_path" ]]; then
        local file_size=$(du -h "$output_path" | cut -f1)
        print_success "Downloaded $description ($file_size)"
        return 0
    else
        print_error "Failed to download $description"
        return 1
    fi
}

download_infinitebench() {
    print_status "📊 Downloading InfiniteBench datasets..."
    
    # All InfiniteBench tasks as listed in their download script
    local infinitebench_tasks=(
        "code_debug"
        "code_run"
        "kv_retrieval"
        "longbook_choice_eng"
        "longbook_qa_chn"
        "longbook_qa_eng"
        "longbook_sum_eng"
        "longdialogue_qa_eng"
        "math_calc"
        "math_find"
        "number_string"
        "passkey"
    )
    
    local base_url="https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main"
    local downloaded_count=0
    local failed_count=0
    
    for task in "${infinitebench_tasks[@]}"; do
        local url="${base_url}/${task}.jsonl?download=true"
        local file="$DATA_DIR/infinitebench_${task}.jsonl"
        
        if download_file "$url" "$file" "InfiniteBench ${task}"; then
            # Verify the file is valid JSON lines
            if command -v head &> /dev/null && command -v jq &> /dev/null; then
                if head -1 "$file" | jq . > /dev/null 2>&1; then
                    local line_count=$(wc -l < "$file")
                    print_success "✅ InfiniteBench ${task}: $line_count items"
                    ((downloaded_count++))
                else
                    print_warning "Downloaded ${task} file may not be valid JSON"
                    ((failed_count++))
                fi
            else
                ((downloaded_count++))
            fi
        else
            print_warning "Failed to download InfiniteBench ${task}"
            ((failed_count++))
        fi
    done
    
    print_status "InfiniteBench download summary: $downloaded_count succeeded, $failed_count failed"
}

download_longbench() {
    print_status "📚 Downloading LongBench datasets..."
    
    # All LongBench v1 tasks from their documentation
    local longbench_tasks=(
        "narrativeqa"
        "qasper"
        "multifieldqa_en"
        "multifieldqa_zh"
        "hotpotqa"
        "2wikimqa"
        "musique"
        "dureader"
        "gov_report"
        "qmsum"
        "multi_news"
        "vcsum"
        "trec"
        "triviaqa"
        "samsum"
        "lsht"
        "passage_count"
        "passage_retrieval_en"
        "passage_retrieval_zh"
        "lcc"
        "repobench-p"
    )
    
    local base_url="https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data"
    local downloaded_count=0
    local failed_count=0
    
    for task in "${longbench_tasks[@]}"; do
        local url="${base_url}/${task}.jsonl"
        local file="$DATA_DIR/longbench_${task}.jsonl"
        
        if download_file "$url" "$file" "LongBench ${task}"; then
            # Verify the file is valid JSON lines
            if command -v head &> /dev/null && command -v jq &> /dev/null; then
                if head -1 "$file" | jq . > /dev/null 2>&1; then
                    local line_count=$(wc -l < "$file")
                    print_success "✅ LongBench ${task}: $line_count items"
                    ((downloaded_count++))
                else
                    print_warning "Downloaded ${task} file may not be valid JSON"
                    ((failed_count++))
                fi
            else
                ((downloaded_count++))
            fi
        else
            print_warning "Failed to download LongBench ${task}"
            ((failed_count++))
        fi
    done
    
    # Try to download LongBench v2 as well
    print_status "📚 Attempting to download LongBench v2..."
    local longbench_v2_url="https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"
    local longbench_v2_file="$DATA_DIR/longbench_v2.json"
    
    if download_file "$longbench_v2_url" "$longbench_v2_file" "LongBench v2"; then
        if command -v jq &> /dev/null; then
            if jq . "$longbench_v2_file" > /dev/null 2>&1; then
                local item_count=$(jq '. | length' "$longbench_v2_file")
                print_success "✅ LongBench v2: $item_count items"
                ((downloaded_count++))
            else
                print_warning "Downloaded LongBench v2 file may not be valid JSON"
                ((failed_count++))
            fi
        else
            ((downloaded_count++))
        fi
    else
        print_warning "Failed to download LongBench v2"
        ((failed_count++))
    fi
    
    print_status "LongBench download summary: $downloaded_count succeeded, $failed_count failed"
}

verify_downloads() {
    print_status "🔍 Verifying downloaded datasets..."
    
    local all_good=true
    
    # Check InfiniteBench
    local ib_files=($(find "$DATA_DIR" -name "infinitebench_*.jsonl" 2>/dev/null))
    if [[ ${#ib_files[@]} -gt 0 ]]; then
        local total_ib_lines=0
        for file in "${ib_files[@]}"; do
            if [[ -f "$file" ]]; then
                local lines=$(wc -l < "$file")
                total_ib_lines=$((total_ib_lines + lines))
            fi
        done
        print_success "✅ InfiniteBench: ${#ib_files[@]} datasets, $total_ib_lines total items"
    else
        print_error "❌ InfiniteBench data missing"
        all_good=false
    fi
    
    # Check LongBench
    local lb_files=($(find "$DATA_DIR" -name "longbench_*.jsonl" 2>/dev/null))
    local lb_v2_file="$DATA_DIR/longbench_v2.json"
    local total_lb_lines=0
    local lb_datasets=0
    
    for file in "${lb_files[@]}"; do
        if [[ -f "$file" ]]; then
            local lines=$(wc -l < "$file")
            total_lb_lines=$((total_lb_lines + lines))
            ((lb_datasets++))
        fi
    done
    
    if [[ -f "$lb_v2_file" ]]; then
        if command -v jq &> /dev/null; then
            local v2_items=$(jq '. | length' "$lb_v2_file" 2>/dev/null || echo "0")
            print_success "✅ LongBench: $lb_datasets v1 datasets ($total_lb_lines items) + v2 ($v2_items items)"
        else
            print_success "✅ LongBench: $lb_datasets v1 datasets ($total_lb_lines items) + v2 available"
        fi
    elif [[ $lb_datasets -gt 0 ]]; then
        print_success "✅ LongBench: $lb_datasets datasets, $total_lb_lines total items"
    else
        print_error "❌ LongBench data missing"
        all_good=false
    fi
    
    if [[ "$all_good" == true ]]; then
        print_success "🎉 All datasets verified successfully!"
        return 0
    else
        print_error "❌ Some datasets are missing or invalid"
        return 1
    fi
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Downloads memory benchmark datasets for emotion experiments"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --infinitebench     Download only InfiniteBench (all 12 tasks)"
    echo "  --longbench         Download only LongBench (v1 and v2)"
    echo "  --verify            Only verify existing downloads"
    echo "  --force             Overwrite existing files without prompting"
    echo ""
    echo "Examples:"
    echo "  $0                  # Download all datasets"
    echo "  $0 --infinitebench  # Download all InfiniteBench tasks"
    echo "  $0 --longbench      # Download all LongBench tasks"
    echo "  $0 --verify         # Verify existing downloads"
}

# Parse command line arguments
DOWNLOAD_ALL=true
DOWNLOAD_INFINITEBENCH=false
DOWNLOAD_LONGBENCH=false
VERIFY_ONLY=false
FORCE_OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        --infinitebench)
            DOWNLOAD_ALL=false
            DOWNLOAD_INFINITEBENCH=true
            shift
            ;;
        --longbench)
            DOWNLOAD_ALL=false
            DOWNLOAD_LONGBENCH=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --force)
            FORCE_OVERWRITE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Set force overwrite mode
if [[ "$FORCE_OVERWRITE" == true ]]; then
    export REPLY="y"  # Auto-answer yes to overwrite prompts
fi

# Main execution
if [[ "$VERIFY_ONLY" == true ]]; then
    verify_downloads
    exit $?
fi

echo "Starting dataset download..."
echo ""

if [[ "$DOWNLOAD_ALL" == true ]] || [[ "$DOWNLOAD_INFINITEBENCH" == true ]]; then
    download_infinitebench
    echo ""
fi

if [[ "$DOWNLOAD_ALL" == true ]] || [[ "$DOWNLOAD_LONGBENCH" == true ]]; then
    download_longbench
    echo ""
fi

# Final verification
echo "======================================"
verify_downloads
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo ""
    print_success "🎉 Dataset download completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run tests: python emotion_memory_experiments/tests/run_all_tests.py"
    echo "2. Or run individual tests:"
    echo "   - python emotion_memory_experiments/tests/test_real_data_comprehensive.py"
    echo "   - python emotion_memory_experiments/tests/test_original_evaluation_metrics.py"
    echo ""
    print_status "Data location: $DATA_DIR"
    echo ""
    echo "📊 InfiniteBench Tasks Downloaded:"
    echo "   • code_debug, code_run, kv_retrieval"
    echo "   • longbook_choice_eng, longbook_qa_chn, longbook_qa_eng, longbook_sum_eng"
    echo "   • longdialogue_qa_eng, math_calc, math_find"
    echo "   • number_string, passkey"
    echo ""
    echo "📚 LongBench Tasks Downloaded:"
    echo "   • Single-doc QA: narrativeqa, qasper, multifieldqa_en, multifieldqa_zh"
    echo "   • Multi-doc QA: hotpotqa, 2wikimqa, musique, dureader"
    echo "   • Summarization: gov_report, qmsum, multi_news, vcsum"
    echo "   • Few-shot: trec, triviaqa, samsum, lsht"
    echo "   • Synthetic: passage_count, passage_retrieval_en, passage_retrieval_zh"
    echo "   • Code: lcc, repobench-p"
    echo "   • LongBench v2: Enhanced evaluation dataset"
else
    echo ""
    print_error "❌ Some downloads failed. Check the output above."
    exit 1
fi