# OpenAI Server Module Documentation

This directory contains documentation for the `openai_server` module, which provides an OpenAI-compatible FastAPI server with integrated emotion control capabilities.

## Quick Navigation

- [Module Overview](#module-overview)
- [Setup Guide](setup.md) - Installation and configuration
- [API Reference](api_reference.md) - Detailed endpoint documentation  
- [Testing Guide](testing.md) - Running and writing tests
- [Integration Examples](integration_examples.md) - LangGraph, AutoGen, and other integrations
- [Deployment Guide](deployment.md) - Production deployment strategies
- [Migration Guide](migration.md) - Upgrading from legacy structure

## Module Overview

The `openai_server` module is a restructured and enhanced version of the original `init_openai_server.py` script. It provides:

### Key Improvements

1. **Modular Architecture**: Organized as a proper Python module with clear separation of concerns
2. **Enhanced Testing**: Comprehensive test suite with both unit and integration tests
3. **Better Documentation**: Detailed guides and API references
4. **Backward Compatibility**: Legacy scripts still work with deprecation warnings
5. **Production Ready**: Docker support, environment configuration, and monitoring

### Module Structure

```
openai_server/
├── __init__.py          # Module initialization
├── __main__.py          # Entry point (python -m openai_server)
├── server.py            # Main FastAPI server implementation
├── README.md            # Module documentation
└── tests/
    ├── __init__.py
    ├── test_openai_server.py
    └── test_integrated_openai_server.py
```

## Quick Start

### New Modular Usage (Recommended)

```bash
# Start server
python -m openai_server --model /path/to/model --model_name MyModel --emotion anger

# Run tests
python -m openai_server.tests.test_openai_server
python -m openai_server.tests.test_integrated_openai_server
```

### Legacy Usage (Deprecated but Supported)

```bash
# Start server (shows deprecation warning)
python init_openai_server.py --model /path/to/model --model_name MyModel --emotion anger

# Run tests (shows deprecation warning)
python test_openai_server.py
python test_integrated_openai_server.py
```

## Related Documentation

- **Main Module README**: [`/openai_server/README.md`](../../openai_server/README.md)
- **Legacy Documentation**: [`openai_server_setup.md`](../openai_server_setup.md) (archived)
- **Neural Manipulation**: [`/neuro_manipulation/README.md`](../neuro_manipulation/README.md)
- **Experiment Integration**: [`/docs/reference/experiment_series_README.md`](../../reference/experiment_series_README.md)