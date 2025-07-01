# OpenAI Server Testing Guide

## Overview

The OpenAI server now has a comprehensive test suite with automated testing via Makefile. This ensures code quality and prevents regressions when modifying the server.

## Test Coverage

### 1. Unit Tests (`tests/test_unit_server.py`)
- Pydantic model validation
- Request/response serialization
- Streaming response generation
- Error handling
- Component isolation testing

### 2. Integration Tests (`tests/test_integration_server.py`)
- Full server startup/shutdown
- API endpoint testing
- Emotion switching
- Concurrent request handling
- Long context support

### 3. Existing Tests
- Server connectivity tests
- Server management tests
- Multi-server orchestration

## Quick Start

### After Making Changes

1. **Quick Test (no GPU required)**
   ```bash
   cd openai_server
   make test-quick
   ```

2. **Full Test Suite**
   ```bash
   make test
   ```

3. **Before Committing**
   ```bash
   make ci  # Runs lint, format, type-check, and all tests
   ```

## Makefile Commands

### Testing Commands
- `make test` - Run all tests
- `make test-unit` - Unit tests only (fast)
- `make test-integration` - Integration tests (requires GPU)
- `make test-coverage` - Generate coverage report
- `make test-quick` - Alias for test-unit

### Code Quality
- `make lint` - Check code style
- `make format` - Auto-format code
- `make type-check` - Run type checker

### Development
- `make server` - Start test server (happiness)
- `make server-anger` - Start server with anger
- `make clean` - Clean generated files
- `make watch` - Auto-run tests on file changes

### CI/CD
- `make ci` - Full CI pipeline simulation
- `make install-dev` - Install dev dependencies

## Test Organization

```
openai_server/
├── Makefile                    # Automated test commands
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── pytest.ini             # Pytest configuration
│   ├── run_all_tests.py       # Test runner script
│   ├── test_unit_server.py    # Unit tests
│   ├── test_integration_server.py  # Integration tests
│   └── README.md              # Detailed test documentation
└── TESTING_GUIDE.md           # This file
```

## Writing New Tests

### Add Unit Test
```python
# In test_unit_server.py
def test_new_feature(self):
    """Test my new feature"""
    result = my_new_function(input_data)
    assert result == expected_output
```

### Add Integration Test
```python
# In test_integration_server.py
@pytest.mark.gpu
def test_new_endpoint(self):
    """Test new API endpoint"""
    response = requests.post(f"{self.base_url}/new_endpoint", json=data)
    assert response.status_code == 200
```

## CI/CD Integration

### GitHub Actions
The project includes `.github/workflows/openai_server_tests.yml` which:
- Runs on pushes to openai_server/**
- Tests multiple Python versions
- Generates coverage reports
- Checks code formatting

### Local CI Simulation
```bash
make ci  # Runs the same checks as GitHub Actions
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /data/home/jjl7137/LLM_EmoBehav_game_theory
   make test
   ```

2. **GPU Memory Issues**
   ```bash
   export VLLM_GPU_MEMORY_UTILIZATION=0.3
   make test-integration
   ```

3. **Missing Dependencies**
   ```bash
   make install-dev
   ```

## Best Practices

1. **Always run tests before committing**
   ```bash
   make test-quick  # At minimum
   make ci         # Ideally
   ```

2. **Keep tests fast**
   - Unit tests should run in < 1 second each
   - Use mocks for external dependencies

3. **Write tests for new features**
   - Add unit tests for new functions
   - Add integration tests for new endpoints

4. **Maintain test coverage**
   - Target: 80%+ coverage
   - Check with: `make test-coverage`

## Next Steps

1. Run `make install-dev` to install dependencies
2. Run `make test-unit` to verify setup
3. Make changes to openai_server/
4. Run `make test-quick` after changes
5. Run `make ci` before committing