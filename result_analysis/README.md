# Result Analysis Directory

This directory contains all post-experiment analysis scripts and results for the LLM Emotional Behavior Game Theory experiments.

## Analysis Scripts

### Core Analysis Scripts
- `analyze_switches_detailed.py` - Analyzes switching patterns between activation_only and context_and_activation conditions
- `analyze_choice_patterns.py` - Comprehensive analysis of choice patterns across all conditions
- `analyze_choice_differences.py` - Finds cases where choices differ between conditions

### Debug/Utility Scripts
- `debug_data_structure.py` - Utility to inspect data structure and verify scenario matching

## Analysis Results

### Switching Pattern Results
- `detailed_switches_with_prompts.csv` - Complete switching data including input prompts and model outputs (370KB)
  - Contains 119 cases where choices switched between activation_only and context_and_activation
  - Includes full prompts and generated texts for detailed analysis
  
- `detailed_switches_ao_ca.csv` - Summary of switching patterns without full prompts (26KB)

## Key Findings

From the latest experiment (2025-07-03):
- **Total scenarios tested**: 1000 (same scenarios across all 4 conditions)
- **Switching cases**: 119 (11.9% of scenarios showed different choices)
  - 67 switches from cooperation to defection (1→2)
  - 52 switches from defection to cooperation (2→1)

### Defection Rates by Condition:
- Baseline: 5.9%
- Context only: 5.3%
- Activation only (anger): 6.7%
- Context + Activation (anger): 8.1%

## Usage

To run analysis on new experiment results:

```bash
# Update the file path in the script to point to new results
python analyze_switches_detailed.py

# For comprehensive pattern analysis
python analyze_choice_patterns.py
```

## Original Experiment Results Location

The raw experiment results are stored in:
```
results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250703_133259/
```