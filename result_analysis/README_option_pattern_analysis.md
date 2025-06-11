# Option Pattern Analysis

## Overview

The Option Pattern Analysis script analyzes common text patterns in Option 1 and Option 2 descriptions for cases where the decision is Option 1. This analysis helps understand what linguistic and semantic features make Option 1 more appealing in game theory scenarios.

## Features

### Core Analysis Components

1. **Text Pattern Analysis**
   - Word frequency analysis
   - N-gram extraction (bigrams and trigrams)
   - Sentence structure patterns
   - Linguistic feature extraction

2. **Comparative Analysis**
   - TF-IDF based distinctive word identification
   - Option length comparison
   - Keyword category analysis

3. **Linguistic Features**
   - Part-of-speech tagging
   - Verb ratio analysis
   - Emotional word detection
   - Sentence complexity metrics

4. **Keyword Analysis**
   - Cooperation vs. competition themes
   - Risk vs. benefit language
   - Action vs. avoidance patterns

5. **Theme Clustering**
   - Unsupervised theme discovery
   - K-means clustering of text patterns

## Installation and Dependencies

### Required Python Packages

```bash
pip install nltk matplotlib seaborn pandas wordcloud numpy scikit-learn
```

### NLTK Data Downloads

The script automatically downloads required NLTK data:
- punkt (tokenization)
- stopwords
- wordnet (lemmatization)
- averaged_perceptron_tagger (POS tagging)
- maxent_ne_chunker (named entity recognition)
- words corpus

## Usage

### Basic Usage

```python
from option_pattern_analysis import OptionPatternAnalyzer

# Initialize analyzer
analyzer = OptionPatternAnalyzer()

# Load experiment data
data_files = ['path/to/exp_results.json']
all_data = analyzer.load_data(data_files)

# Filter for Option 1 decisions
option1_data = analyzer.filter_option1_decisions(all_data)

# Run analysis
results = analyzer.analyze_patterns(option1_data)

# Save results
analyzer.save_results(results, 'analysis_results.json')

# Generate visualizations
analyzer.generate_visualizations(results, 'visualizations/')
```

### Command Line Usage

```bash
conda activate llm
python option_pattern_analysis.py
```

## Data Format

### Input Data Structure

The script expects JSON files with the following structure:

```json
[
  {
    "emotion": "anger",
    "intensity": 1.5,
    "scenario": "Scenario Name",
    "description": "Scenario description",
    "input": "Option 1. First choice\\nOption 2. Second choice\\nresponse format",
    "rationale": "Decision reasoning",
    "decision": "Option 1. First choice",
    "category": 1
  }
]
```

### Key Fields

- **emotion**: The emotional condition (e.g., "anger", "neutral")
- **scenario**: Name of the game theory scenario
- **input**: Full prompt including both options
- **decision**: The chosen option
- **rationale**: Reasoning for the decision

## Analysis Output

### Results Structure

```json
{
  "option1_patterns": {
    "word_frequency": [["word", frequency], ...],
    "bigrams": [["word1 word2", frequency], ...],
    "trigrams": [["word1 word2 word3", frequency], ...],
    "sentence_starters": [["word", frequency], ...],
    "total_texts": 100,
    "avg_length": 8.5,
    "unique_words": 250
  },
  "option2_patterns": { /* Similar structure */ },
  "comparative_analysis": {
    "option1_distinctive_words": [["word", tfidf_score], ...],
    "option2_distinctive_words": [["word", tfidf_score], ...],
    "option1_avg_length": 8.2,
    "option2_avg_length": 7.8
  },
  "linguistic_features": {
    "option1_features": { /* POS tags, verb ratios, etc. */ },
    "option2_features": { /* Similar structure */ }
  },
  "keyword_analysis": {
    "option1_keywords": {
      "cooperation": 15,
      "competition": 3,
      "risk": 8,
      "benefit": 12,
      "action": 20,
      "avoidance": 2
    },
    "option2_keywords": { /* Similar structure */ }
  },
  "theme_analysis": {
    "themes": {
      "theme_0": ["community", "benefit", "together"],
      "theme_1": ["individual", "personal", "private"]
    },
    "cluster_assignments": [0, 1, 0, 1, ...],
    "n_clusters": 2
  }
}
```

## Visualizations

The script generates several visualizations:

1. **Word Frequency Comparison**: Bar charts comparing most common words in Option 1 vs Option 2
2. **Keyword Category Comparison**: Bar chart showing keyword category distributions
3. **Word Clouds**: Visual representation of word frequencies for each option

### Visualization Files

- `word_frequency_comparison.png`
- `keyword_category_comparison.png`
- `option1_wordcloud.png`
- `option2_wordcloud.png`

## Key Insights

### What the Analysis Reveals

1. **Cooperation vs. Competition Patterns**
   - Option 1 typically contains more cooperation-related language
   - Option 2 often emphasizes individual benefits

2. **Action vs. Avoidance Language**
   - Option 1 uses more action-oriented verbs
   - Option 2 may contain more avoidance or passive language

3. **Risk vs. Benefit Framing**
   - Different risk/benefit language patterns between options
   - Community vs. individual benefit emphasis

4. **Linguistic Complexity**
   - Sentence structure differences
   - Emotional language usage patterns

## Methodology

### Text Preprocessing

1. **Tokenization**: Split text into individual words
2. **Normalization**: Convert to lowercase, remove special characters
3. **Stop Word Removal**: Filter out common words (the, and, is, etc.)
4. **Lemmatization**: Reduce words to their base forms

### Pattern Extraction

1. **N-gram Analysis**: Extract common word sequences
2. **POS Tagging**: Identify grammatical roles of words
3. **TF-IDF Analysis**: Find distinctive words for each option
4. **Clustering**: Group similar texts to identify themes

### Statistical Analysis

- Frequency distributions
- Comparative statistics
- Clustering algorithms (K-means)
- Feature aggregation and normalization

## Limitations

1. **Language Dependency**: Currently optimized for English text
2. **Domain Specificity**: Designed for game theory scenarios
3. **Sample Size**: Requires sufficient data for meaningful patterns
4. **Preprocessing Assumptions**: May lose some semantic nuances

## Testing

### Running Tests

```bash
conda activate llm
python test_option_pattern_analysis.py
```

### Test Coverage

- Unit tests for all major functions
- Integration tests for complete workflow
- Mock testing for visualization components
- Edge case handling

## Future Enhancements

1. **Semantic Analysis**: Add word embeddings and semantic similarity
2. **Sentiment Analysis**: Incorporate emotion detection
3. **Cross-Language Support**: Extend to other languages
4. **Advanced Clustering**: Use more sophisticated clustering algorithms
5. **Interactive Visualizations**: Add dynamic plotting capabilities

## Contributing

When extending this analysis:

1. Follow the existing code structure
2. Add comprehensive unit tests
3. Update documentation
4. Consider backward compatibility
5. Test with various data formats

## File Structure

```
result_analysis/
├── option_pattern_analysis.py          # Main analysis script
├── test_option_pattern_analysis.py     # Unit tests
├── README_option_pattern_analysis.md   # This documentation
├── option_pattern_analysis_results.json # Analysis results
└── option_pattern_visualizations/      # Generated plots
    ├── word_frequency_comparison.png
    ├── keyword_category_comparison.png
    ├── option1_wordcloud.png
    └── option2_wordcloud.png
```

## Example Output

### Sample Analysis Summary

```
==================================================
OPTION PATTERN ANALYSIS SUMMARY
==================================================

Total cases analyzed: 1250
Option pairs extracted: 1180

Option 1 - Top 5 words:
  community: 145
  contribute: 132
  benefit: 98
  together: 87
  invest: 76

Option 2 - Top 5 words:
  personal: 156
  individual: 134
  private: 89
  focus: 82
  own: 71

Option 1 distinctive words:
  collaboration: 0.234
  collective: 0.198
  shared: 0.187
  mutual: 0.165
  cooperative: 0.143

Option 2 distinctive words:
  individual: 0.267
  private: 0.234
  personal: 0.221
  independent: 0.198
  separate: 0.176

Keyword analysis - Option 1:
  cooperation: 234
  competition: 45
  risk: 67
  benefit: 189
  action: 298
  avoidance: 23
```

This analysis provides valuable insights into the linguistic patterns that influence decision-making in game theory scenarios, particularly understanding what makes cooperative options (Option 1) more appealing in various emotional contexts. 