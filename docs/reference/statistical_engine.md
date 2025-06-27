# Statistical Engine

This module provides a unified framework for statistical analysis and visualization of behavioral data, particularly for experiments involving emotions, intensities, and categorical decisions (e.g., cooperation/defection in game theory). It supports both JSON and CSV data formats and produces both statistical summaries and publication-ready plots.

## Main Components

### 1. `BehaviorAnalyzer`
- **Purpose:** Central class for loading data, performing statistical analysis, and structuring results.
- **Key Methods:**
  - `load_data(file_path)`: Loads data from JSON or CSV.
  - `analyze_data(data_source, output_dir=None)`: Main entry point for analysis. Handles both CSV and JSON input, computes statistics, and generates plots.
  - `_analyze_emotion_effects(df)`: Analyzes behavioral differences across emotions.
  - `_analyze_intensity_effects(df)`: Analyzes intensity effects within each emotion.
  - `_analyze_multiple_emotions(emotion_files)`: Handles multiple JSON files (one per emotion).
  - `_analyze_conditions(condition_counts)`: Core logic for statistical tests and pairwise comparisons.
  - `format_full_results(analysis_results)`: Adds human-readable text descriptions to results.

### 2. `BehaviorVisualizer`
- **Purpose:** Handles all plotting and visualization of analysis results.
- **Key Methods:**
  - `update_category_labels(categories)`: Sets up color mapping for categories.
  - `plot_results(results, output_path, title=None)`: Main plotting function, creates bar plots and p-value heatmaps.
  - Internal methods for plotting rates, heatmaps, and raw counts.

### 3. Statistical Tests
- **Chi-square test** and **Fisher's exact test** are used to assess differences in categorical behavior across conditions (emotions/intensities).
- **Pairwise comparisons** are performed between all pairs of conditions.
- **Significance markers** (`*`, `**`) are added based on p-value thresholds.

### 4. Textual Summaries
- The module generates a detailed, human-readable summary of the statistical findings, including:
  - Per-condition behavior rates
  - Overall test results
  - Pairwise comparison results

## Example Usage

```python
from statistical_engine import BehaviorAnalyzer

analyzer = BehaviorAnalyzer()
results = analyzer.analyze_data('path/to/data.csv')
print(results['text_description'])
```

- For JSON input (multiple files):
```python
emotion_files = {
    'happy': 'data/happy.json',
    'sad': 'data/sad.json',
    # ...
}
results = analyzer.analyze_data(emotion_files)
```

## Output
- **Plots:** Saved to the specified output directory (default: sibling 'plots' folder).
- **Results:** Dictionary with detailed statistics and a text summary.

## Dependencies
- numpy, pandas, scipy, matplotlib, seaborn

## See Also
- [statistical_engine.py](../../statistical_engine.py) 