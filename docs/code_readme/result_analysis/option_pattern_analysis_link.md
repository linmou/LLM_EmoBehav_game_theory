# Option Pattern Analysis Documentation

This document provides a link to the comprehensive documentation for the Option Pattern Analysis functionality.

## Documentation Location

📖 **Main Documentation**: [Option Pattern Analysis README](../result_analysis/README_option_pattern_analysis.md)

## Quick Links

- **Main Script**: [option_pattern_analysis.py](../result_analysis/option_pattern_analysis.py)
- **Unit Tests**: [test_option_pattern_analysis.py](../result_analysis/test_option_pattern_analysis.py)
- **Documentation**: [README_option_pattern_analysis.md](../result_analysis/README_option_pattern_analysis.md)

## Overview

The Option Pattern Analysis script analyzes common text patterns in Option 1 and Option 2 descriptions for cases where the decision is Option 1. This analysis helps understand what linguistic and semantic features make Option 1 more appealing in game theory scenarios.

## Key Features

- **Text Pattern Analysis**: Word frequency, N-grams, sentence patterns
- **Comparative Analysis**: TF-IDF distinctive words, length comparison
- **Linguistic Features**: POS tagging, verb ratios, emotional words
- **Keyword Analysis**: Cooperation vs competition themes
- **Theme Clustering**: Unsupervised pattern discovery
- **Visualizations**: Word clouds, frequency charts, category comparisons

## Quick Start

```bash
# Activate environment
conda activate llm

# Run analysis
cd result_analysis
python option_pattern_analysis.py

# Run tests
python test_option_pattern_analysis.py
```

## Analysis Output

The script generates:
- Detailed JSON results file
- Multiple visualization plots
- Statistical summaries
- Pattern comparisons

For complete documentation, usage examples, and methodology details, see the [full README](../result_analysis/README_option_pattern_analysis.md). 