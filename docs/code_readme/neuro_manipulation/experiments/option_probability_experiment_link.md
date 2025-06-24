# Option Probability Experiment

[**📊 Option Probability Experiment Documentation →**](README_option_probability_experiment.md)

## Quick Links

- **[ContextManipulationDataset](../../../../neuro_manipulation/datasets/context_manipulation_dataset.py)** - Dataset with conditional context manipulation
- **[OptionProbabilityExperiment](../../../../neuro_manipulation/experiments/option_probability_experiment.py)** - Main experiment class
- **[Unit Tests - Dataset](../../../../neuro_manipulation/tests/test_context_manipulation_dataset.py)** - Dataset unit tests
- **[Unit Tests - Experiment](../../../../neuro_manipulation/tests/test_option_probability_experiment.py)** - Experiment unit tests

## Summary

The Option Probability Experiment measures how emotional states and contextual factors influence decision-making probabilities in game-theoretic scenarios using a 2x2 factorial design with `CombinedVLLMHook`.

## Key Features
- 🎯 **2x2 Factorial Design**: emotion (neutral/angry) × context (with/without description)
- 📈 **Sequence Probability Measurement**: Uses `CombinedVLLMHook` for accurate probability calculation
- 📊 **Comprehensive Analysis**: Automatic statistical testing and reporting  
- 🔧 **Modular Design**: Easy to extend with new conditions
- 📁 **Multiple Output Formats**: JSON, CSV, and human-readable reports 