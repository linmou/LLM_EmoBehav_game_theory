# Option Probability Experiment

For comprehensive documentation on the Option Probability Experiment framework, see:

**[Option Probability Experiment Documentation](README_option_probability_experiment.md)**

## Quick Links

- **[ContextManipulationDataset](../../../../neuro_manipulation/datasets/context_manipulation_dataset.py)** - Dataset with conditional context manipulation
- **[OptionProbabilityExperiment](../../../../neuro_manipulation/experiments/option_probability_experiment.py)** - Main experiment class
- **[Unit Tests - Dataset](../../../../neuro_manipulation/tests/test_context_manipulation_dataset.py)** - Dataset unit tests
- **[Unit Tests - Experiment](../../../../neuro_manipulation/tests/test_option_probability_experiment.py)** - Experiment unit tests

## Summary

The Option Probability Experiment framework implements a 2x2 factorial design for studying emotion-context interactions in behavioral decision-making. It measures probabilities of all available behavioral choices using SequenceProbVLLMHook and provides comprehensive statistical analysis.

**Key Features:**
- 2x2 factorial design (Emotion × Context)
- vLLM-based sequence probability measurement
- Automated statistical analysis with interaction effects
- Comprehensive unit testing
- Full documentation and examples 