# Enhanced Scenario Generation with Restart Capabilities

This document provides a link to the comprehensive documentation for the enhanced scenario generation system with restart capabilities and timeout handling.

## 📍 Full Documentation

For complete documentation on the enhanced scenario generation system, please see:

**[Enhanced Scenario Generation README](../../data_creation/scenario_creation/langgraph_creation/README.md)**

## Quick Overview

The enhanced scenario generation system includes:

- **🔄 Restart Capability**: Automatic resume from where it left off
- **⏱️ Timeout Handling**: Configurable timeouts with retry logic
- **📊 Comprehensive Logging**: Detailed progress tracking and error reporting
- **🛡️ Error Recovery**: Multi-layered error handling and graceful degradation

## Key Features

### Automatic Restart
- Detects completed scenarios and skips them automatically
- No duplicate work when restarting after interruption
- Progress preservation through immediate file saving

### Timeout Management
- Individual task timeout: 5 minutes (configurable)
- Batch timeout: 30 minutes (configurable)
- Automatic retry with exponential backoff

### Robust Error Handling
- Individual task isolation prevents cascade failures
- Batch-level retry with graph reconstruction
- Process-level restart capability

## Quick Start

```bash
# Run the enhanced scenario generation
python data_creation/create_scenario_langgraph.py

# If interrupted, simply restart the same command
# The system will automatically continue from where it left off
```

## Configuration

Key configuration constants:
```python
TASK_TIMEOUT = 300      # 5 minutes per task
BATCH_TIMEOUT = 1800    # 30 minutes per batch
MAX_RETRIES = 3         # Maximum retry attempts
RETRY_DELAY = 5         # Seconds between retries
```

## Monitoring

- **Console**: Real-time progress updates
- **Log File**: `data_creation/scenario_generation.log`
- **File System**: Monitor scenario output directories

For detailed usage instructions, troubleshooting, and advanced configuration, please refer to the [full documentation](../../data_creation/scenario_creation/langgraph_creation/README.md).

## Related Documentation

- [LangGraph Scenario Creation](../../data_creation/scenario_creation/langgraph_creation/README.md)
- [Game Theory Implementation](../games/)
- [Project Documentation](../index.md) 