# Demo: activation's impact on sequence probability

This directory contains a link to the anger activation demonstration that shows the impact of emotion manipulation on word probability.

## Demo Location

The anger activation demo is located at:
- **File**: `neuro_manipulation/repe/sequence_prob_demo.py`
- **Documentation**: `neuro_manipulation/repe/README_sequence_prob_demo.md`

## Quick Usage

```python
from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo

# Initialize and run demo
demo = AngerActivationDemo()
results = demo.run_probability_analysis()
plot_path = demo.create_visualization(results)
```

## Purpose

The demo validates that:
1. Anger emotion activation measurably affects word probabilities
2. The sequence probability measurement system works with vLLM models
3. Higher anger intensities correlate with increased "angry" word probability

## Features

- **Emotion Activation**: Uses pre-trained anger direction vectors
- **Probability Measurement**: Measures exact probabilities using vLLM hooks
- **Visualization**: Creates plots showing intensity vs. probability relationship
- **Multiple Intensities**: Tests baseline and various anger activation levels

## Related Files

- Main demo: [sequence_prob_demo.py](../../repe/sequence_prob_demo.py)
- Documentation: [README_anger_activation_demo.md](../../repe/README_anger_activation_demo.md)
- Unit tests: [test_anger_activation_demo.py](../../repe/tests/test_anger_activation_demo.py) 