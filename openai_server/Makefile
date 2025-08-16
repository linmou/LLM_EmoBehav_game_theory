# Makefile for OpenAI Server
# Run tests and quality checks after modifications

.PHONY: all test test-unit test-integration test-existing test-coverage \
        lint format type-check clean install-dev server docs help

# Python interpreter
PYTHON := python3
MODULE := openai_server
TEST_DIR := tests
COVERAGE_DIR := htmlcov

# Default target
all: lint type-check test

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	$(PYTHON) -m pip install pytest pytest-cov pytest-asyncio pytest-timeout
	$(PYTHON) -m pip install black isort flake8 mypy
	$(PYTHON) -m pip install openai requests

# Run all tests
test:
	@echo "Running all tests..."
	cd $(TEST_DIR) && $(PYTHON) run_all_tests.py

# Run only unit tests (fast)
test-unit:
	@echo "Running unit tests..."
	cd $(TEST_DIR) && $(PYTHON) run_all_tests.py --unit-only -v

# Run only integration tests (requires GPU)
test-integration:
	@echo "Running integration tests..."
	cd $(TEST_DIR) && $(PYTHON) run_all_tests.py --integration-only -v

# Run existing test files
test-existing:
	@echo "Running existing tests..."
	cd $(TEST_DIR) && $(PYTHON) run_all_tests.py --existing

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	cd $(TEST_DIR) && $(PYTHON) run_all_tests.py --coverage
	@echo "Coverage report available in $(TEST_DIR)/$(COVERAGE_DIR)/index.html"

# Quick test (unit tests only, no GPU required)
test-quick: test-unit

# Lint code
lint:
	@echo "Running linter..."
	$(PYTHON) -m flake8 . --max-line-length=100 --exclude=__pycache__,tests
	cd $(TEST_DIR) && $(PYTHON) -m flake8 . --max-line-length=100 --exclude=__pycache__ --ignore=F401,F541,E402,E501,F841,W503

# Format code
format:
	@echo "Formatting code..."
	$(PYTHON) -m black . --line-length=100 --exclude=tests
	$(PYTHON) -m isort . --profile black --skip tests
	cd $(TEST_DIR) && $(PYTHON) -m black . --line-length=100
	cd $(TEST_DIR) && $(PYTHON) -m isort . --profile black

# Type checking
type-check:
	@echo "Running type checker..."
	$(PYTHON) -m mypy . --ignore-missing-imports --exclude=tests || true

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf $(TEST_DIR)/$(COVERAGE_DIR)
	rm -rf $(TEST_DIR)/.coverage
	rm -rf $(TEST_DIR)/.pytest_cache
	rm -rf logs/server/*.log

# Start test server with optimized settings
server:
	@echo "Starting optimized test server..."
	VLLM_GPU_MEMORY_UTILIZATION=0.9 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model \
		--emotion happiness \
		--port 8000 \
		--gpu_memory_utilization 0.9 \
		--max_num_seqs 64 \
		--batch_size 8 \
		--batch_timeout 0.05

# Start server with specific emotion (optimized)
server-anger:
	@echo "Starting optimized server with anger emotion..."
	VLLM_GPU_MEMORY_UTILIZATION=0.9 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model-anger \
		--emotion anger \
		--port 8000 \
		--gpu_memory_utilization 0.9 \
		--max_num_seqs 64 \
		--batch_size 8 \
		--batch_timeout 0.05

# Start high-performance server with maximum optimization
server-fast:
	@echo "Starting high-performance server..."
	VLLM_GPU_MEMORY_UTILIZATION=0.95 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model-fast \
		--emotion happiness \
		--port 8000 \
		--gpu_memory_utilization 0.95 \
		--max_num_seqs 128 \
		--batch_size 16 \
		--batch_timeout 0.02 \
		--enable_chunked_prefill

# Start server without batching (for comparison)
server-no-batch:
	@echo "Starting server without batching..."
	VLLM_GPU_MEMORY_UTILIZATION=0.9 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model-nobatch \
		--emotion happiness \
		--port 8000 \
		--gpu_memory_utilization 0.9 \
		--max_num_seqs 64 \
		--disable_batching

# Start multi-GPU server (2 GPUs - balanced load)
server-multigpu:
	@echo "Starting multi-GPU balanced server..."
	VLLM_GPU_MEMORY_UTILIZATION=0.8 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model-multigpu \
		--emotion happiness \
		--port 8000 \
		--gpu_memory_utilization 0.8 \
		--max_num_seqs 64 \
		--batch_size 8 \
		--batch_timeout 0.05 \
		--tensor_parallel_size 2

# Start multi-GPU high performance server
server-multigpu-fast:
	@echo "Starting multi-GPU high-performance server..."
	VLLM_GPU_MEMORY_UTILIZATION=0.85 $(PYTHON) -m $(MODULE) \
		--model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
		--model_name test-model-multigpu-fast \
		--emotion happiness \
		--port 8000 \
		--gpu_memory_utilization 0.85 \
		--max_num_seqs 128 \
		--batch_size 16 \
		--batch_timeout 0.02 \
		--tensor_parallel_size 2

# Generate documentation
docs:
	@echo "Generating documentation..."
	$(PYTHON) -m pydoc -w $(MODULE).server
	mv *.html docs/ 2>/dev/null || true

# Watch for changes and run tests
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -e modify -r $(MODULE) $(TEST_DIR) 2>/dev/null || sleep 5; \
		clear; \
		make test-quick; \
	done

# CI/CD pipeline simulation
ci: clean install-dev lint type-check test-unit
	@echo "CI pipeline complete!"

# Help
help:
	@echo "OpenAI Server Makefile"
	@echo "====================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Run lint, type-check, and all tests (default)"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only (fast, no GPU)"
	@echo "  test-integration - Run integration tests (requires GPU)"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  test-quick       - Quick test (unit tests only)"
	@echo "  lint             - Run code linter"
	@echo "  format           - Format code with black/isort"
	@echo "  type-check       - Run type checker"
	@echo "  clean            - Clean generated files"
	@echo "  install-dev      - Install development dependencies"
	@echo "  server           - Start optimized test server (happiness)"
	@echo "  server-anger     - Start optimized test server (anger)"
	@echo "  server-fast      - Start high-performance server (max optimization)"
	@echo "  server-no-batch  - Start server without batching (for comparison)"
	@echo "  docs             - Generate documentation"
	@echo "  watch            - Watch for changes and run tests"
	@echo "  ci               - Run full CI pipeline"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test-quick  # Run quick tests after code changes"
	@echo "  make ci          # Run full test suite before commit"
	@echo "  make server      # Start server for manual testing"