# Emotion Expression Analysis

This document provides information about the emotion expression analysis in rationales.


## Overview

The emotion expression analysis examines if the rationales in experiment results express the emotions they were assigned and investigates correlations with specific categories.

### Key Features

- Uses GPT-4o-mini for emotion detection in text
- **Parallel Processing** for significantly faster analysis of large datasets
- Provides statistical analysis of emotion expression rates by category
- Generates visualizations to compare expression rates
- Exports detailed results for further analysis

## Performance Optimization

The analysis script uses parallel processing to significantly improve performance:

- Processes multiple items concurrently using thread pooling
- Implements smart rate limiting to avoid API throttling
- Displays a progress bar for real-time feedback
- Configurable performance parameters (workers, rate limits)

## Data Size Control

For testing or limited analysis, the script can be configured to analyze only a subset of the data:

- By default, processes only the first 100 items
- Easily adjustable limit by modifying a single line of code
- Useful for preliminary analysis or when testing new features

## Running the Analysis

To run the analysis:

```bash
# Activate conda environment
conda activate llm

# Run the script
python -m result_analysis.rational_analysis
```

See the [detailed documentation](../result_analysis/README.md) for more information about performance tuning, data size control, requirements, outputs, and customization options. 