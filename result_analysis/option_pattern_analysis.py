#!/usr/bin/env python3
"""
Option Pattern Analysis Script

This script analyzes common text patterns in Option 1 and Option 2 descriptions
for cases where the decision is Option 1. It extracts linguistic patterns,
keywords, and themes to understand what makes Option 1 more appealing.
"""

import json
import os
import re
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class OptionPatternAnalyzer:
    """Analyzes text patterns in Option 1 and Option 2 for cases where Option 1 is chosen."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.option1_texts = []
        self.option2_texts = []
        self.scenarios = []
        self.emotions = []
        
    def extract_options_from_input(self, input_text):
        """Extract Option 1 and Option 2 text from the input."""
        try:
            # Find Option 1 and Option 2 patterns - fix the regex to handle actual newlines
            option1_pattern = r'Option 1\.\s*([^\n]+?)(?=\nOption 2|\\nOption 2|$)'
            option2_pattern = r'Option 2\.\s*([^\n]+?)(?=\nresponse|\\nresponse|$)'
            
            option1_match = re.search(option1_pattern, input_text, re.DOTALL)
            option2_match = re.search(option2_pattern, input_text, re.DOTALL)
            
            option1_text = option1_match.group(1).strip() if option1_match else ""
            option2_text = option2_match.group(1).strip() if option2_match else ""
            
            return option1_text, option2_text
        except Exception as e:
            logger.warning(f"Error extracting options: {e}")
            return "", ""
    
    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def extract_linguistic_features(self, text):
        """Extract linguistic features from text."""
        features = {}
        
        # Basic statistics
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in word_tokenize(text)])
        
        # POS tags
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        # Normalize POS counts
        total_words = len(tokens)
        for pos, count in pos_counts.items():
            features[f'pos_{pos}'] = count / total_words if total_words > 0 else 0
        
        # Action words (verbs)
        verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
        features['verb_count'] = len(verbs)
        features['verb_ratio'] = len(verbs) / total_words if total_words > 0 else 0
        
        # Emotional words (simple heuristic)
        emotional_words = ['benefit', 'gain', 'profit', 'advantage', 'positive', 'good', 'better',
                          'loss', 'cost', 'burden', 'negative', 'bad', 'worse', 'risk', 'threat']
        emotional_count = sum(1 for word in tokens if word.lower() in emotional_words)
        features['emotional_word_ratio'] = emotional_count / total_words if total_words > 0 else 0
        
        return features
    
    def load_data(self, file_paths):
        """Load data from experiment result files."""
        all_data = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    logger.info(f"Loaded {len(data)} entries from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return all_data
    
    def filter_option1_decisions(self, data):
        """Filter data to only include cases where Option 1 was chosen."""
        option1_cases = []
        
        for item in data:
            decision = item.get('decision', '')
            if 'Option 1' in decision or decision.startswith('Invest') or decision.startswith('Adhere') or decision.startswith('Voluntarily'):
                option1_cases.append(item)
        
        logger.info(f"Found {len(option1_cases)} cases where Option 1 was chosen out of {len(data)} total cases")
        return option1_cases
    
    def analyze_patterns(self, data):
        """Analyze patterns in Option 1 and Option 2 texts."""
        logger.info("Starting pattern analysis...")
        
        for item in data:
            input_text = item.get('input', '')
            option1_text, option2_text = self.extract_options_from_input(input_text)
            
            if option1_text and option2_text:
                self.option1_texts.append(option1_text)
                self.option2_texts.append(option2_text)
                self.scenarios.append(item.get('scenario', ''))
                self.emotions.append(item.get('emotion', ''))
        
        logger.info(f"Extracted {len(self.option1_texts)} option pairs for analysis")
        
        # Analyze patterns
        results = {
            'option1_patterns': self.analyze_text_patterns(self.option1_texts, "Option 1"),
            'option2_patterns': self.analyze_text_patterns(self.option2_texts, "Option 2"),
            'comparative_analysis': self.compare_options(),
            'linguistic_features': self.analyze_linguistic_features(),
            'keyword_analysis': self.analyze_keywords(),
            'theme_analysis': self.analyze_themes()
        }
        
        return results
    
    def analyze_text_patterns(self, texts, option_name):
        """Analyze patterns in a collection of texts."""
        logger.info(f"Analyzing patterns for {option_name}")
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Preprocess
        tokens = self.preprocess_text(combined_text)
        
        # Most common words
        word_freq = Counter(tokens)
        most_common = word_freq.most_common(20)
        
        # N-grams
        bigrams = []
        trigrams = []
        
        for text in texts:
            text_tokens = self.preprocess_text(text)
            if len(text_tokens) >= 2:
                bigrams.extend([' '.join(text_tokens[i:i+2]) for i in range(len(text_tokens)-1)])
            if len(text_tokens) >= 3:
                trigrams.extend([' '.join(text_tokens[i:i+3]) for i in range(len(text_tokens)-2)])
        
        bigram_freq = Counter(bigrams).most_common(10)
        trigram_freq = Counter(trigrams).most_common(10)
        
        # Sentence patterns
        sentence_starters = []
        for text in texts:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if words:
                    sentence_starters.append(words[0])
        
        starter_freq = Counter(sentence_starters).most_common(10)
        
        return {
            'word_frequency': most_common,
            'bigrams': bigram_freq,
            'trigrams': trigram_freq,
            'sentence_starters': starter_freq,
            'total_texts': len(texts),
            'avg_length': np.mean([len(text.split()) for text in texts]),
            'unique_words': len(set(tokens))
        }
    
    def compare_options(self):
        """Compare Option 1 and Option 2 patterns."""
        logger.info("Comparing Option 1 and Option 2 patterns")
        
        # TF-IDF analysis
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Combine texts for comparison
        option1_combined = ' '.join(self.option1_texts)
        option2_combined = ' '.join(self.option2_texts)
        
        tfidf_matrix = vectorizer.fit_transform([option1_combined, option2_combined])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get distinctive words for each option
        option1_scores = tfidf_matrix[0].toarray()[0]
        option2_scores = tfidf_matrix[1].toarray()[0]
        
        option1_distinctive = [(feature_names[i], option1_scores[i]) 
                              for i in np.argsort(option1_scores)[-10:]]
        option2_distinctive = [(feature_names[i], option2_scores[i]) 
                              for i in np.argsort(option2_scores)[-10:]]
        
        return {
            'option1_distinctive_words': option1_distinctive,
            'option2_distinctive_words': option2_distinctive,
            'option1_avg_length': np.mean([len(text.split()) for text in self.option1_texts]),
            'option2_avg_length': np.mean([len(text.split()) for text in self.option2_texts])
        }
    
    def analyze_linguistic_features(self):
        """Analyze linguistic features of both options."""
        logger.info("Analyzing linguistic features")
        
        option1_features = []
        option2_features = []
        
        for text in self.option1_texts:
            option1_features.append(self.extract_linguistic_features(text))
        
        for text in self.option2_texts:
            option2_features.append(self.extract_linguistic_features(text))
        
        # Aggregate features
        def aggregate_features(features_list):
            if not features_list:
                return {}
            
            aggregated = {}
            for key in features_list[0].keys():
                values = [f.get(key, 0) for f in features_list]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
            
            return aggregated
        
        return {
            'option1_features': aggregate_features(option1_features),
            'option2_features': aggregate_features(option2_features)
        }
    
    def analyze_keywords(self):
        """Analyze keywords and their contexts."""
        logger.info("Analyzing keywords")
        
        # Define keyword categories
        keyword_categories = {
            'cooperation': ['contribute', 'collaborate', 'together', 'joint', 'shared', 'community', 'collective'],
            'competition': ['compete', 'advantage', 'profit', 'gain', 'win', 'beat', 'outperform'],
            'risk': ['risk', 'threat', 'danger', 'loss', 'cost', 'burden', 'negative'],
            'benefit': ['benefit', 'advantage', 'gain', 'profit', 'positive', 'good', 'better'],
            'action': ['invest', 'contribute', 'participate', 'engage', 'act', 'do', 'make'],
            'avoidance': ['avoid', 'prevent', 'stop', 'refuse', 'decline', 'opt out', 'focus']
        }
        
        def count_keywords_in_texts(texts, categories):
            category_counts = defaultdict(int)
            for text in texts:
                text_lower = text.lower()
                for category, keywords in categories.items():
                    for keyword in keywords:
                        category_counts[category] += text_lower.count(keyword)
            return dict(category_counts)
        
        option1_keywords = count_keywords_in_texts(self.option1_texts, keyword_categories)
        option2_keywords = count_keywords_in_texts(self.option2_texts, keyword_categories)
        
        return {
            'option1_keywords': option1_keywords,
            'option2_keywords': option2_keywords,
            'keyword_categories': keyword_categories
        }
    
    def analyze_themes(self):
        """Analyze themes using clustering."""
        logger.info("Analyzing themes")
        
        # Combine all texts for theme analysis
        all_texts = self.option1_texts + self.option2_texts
        
        if len(all_texts) < 5:
            return {"error": "Not enough texts for theme analysis"}
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # K-means clustering
            n_clusters = min(5, len(all_texts) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Analyze clusters
            feature_names = vectorizer.get_feature_names_out()
            cluster_themes = {}
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_features = [feature_names[j] for j in cluster_center.argsort()[-5:]]
                cluster_themes[f'theme_{i}'] = top_features
            
            return {
                'themes': cluster_themes,
                'cluster_assignments': clusters.tolist(),
                'n_clusters': n_clusters
            }
        
        except Exception as e:
            logger.error(f"Error in theme analysis: {e}")
            return {"error": str(e)}
    
    def generate_visualizations(self, results, output_dir):
        """Generate visualizations for the analysis results."""
        logger.info("Generating visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Word frequency comparison
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        option1_words = [word for word, freq in results['option1_patterns']['word_frequency']]
        option1_freqs = [freq for word, freq in results['option1_patterns']['word_frequency']]
        plt.barh(option1_words, option1_freqs)
        plt.title('Option 1: Most Common Words')
        plt.xlabel('Frequency')
        
        plt.subplot(1, 2, 2)
        option2_words = [word for word, freq in results['option2_patterns']['word_frequency']]
        option2_freqs = [freq for word, freq in results['option2_patterns']['word_frequency']]
        plt.barh(option2_words, option2_freqs)
        plt.title('Option 2: Most Common Words')
        plt.xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_frequency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Keyword category comparison
        if 'keyword_analysis' in results:
            plt.figure(figsize=(12, 6))
            
            categories = list(results['keyword_analysis']['option1_keywords'].keys())
            option1_counts = [results['keyword_analysis']['option1_keywords'][cat] for cat in categories]
            option2_counts = [results['keyword_analysis']['option2_keywords'][cat] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, option1_counts, width, label='Option 1', alpha=0.8)
            plt.bar(x + width/2, option2_counts, width, label='Option 2', alpha=0.8)
            
            plt.xlabel('Keyword Categories')
            plt.ylabel('Count')
            plt.title('Keyword Category Comparison')
            plt.xticks(x, categories, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'keyword_category_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Word clouds
        if self.option1_texts:
            option1_combined = ' '.join(self.option1_texts)
            wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate(option1_combined)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud1, interpolation='bilinear')
            plt.axis('off')
            plt.title('Option 1 Word Cloud')
            plt.savefig(os.path.join(output_dir, 'option1_wordcloud.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        if self.option2_texts:
            option2_combined = ' '.join(self.option2_texts)
            wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate(option2_combined)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud2, interpolation='bilinear')
            plt.axis('off')
            plt.title('Option 2 Word Cloud')
            plt.savefig(os.path.join(output_dir, 'option2_wordcloud.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results, output_file):
        """Save analysis results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main function to run the option pattern analysis."""
    # Initialize analyzer
    analyzer = OptionPatternAnalyzer()
    
    # Find experiment result files
    # result_files = []
    # for root, dirs, files in os.walk('results'):
    #     for file in files:
    #         if file == 'exp_results.json':
    #             result_files.append(os.path.join(root, file))
    result_files = [
        "results/Qwen2.5_Series_Prisoners_Dilemma/Qwen2.5_Series_Prisoners_Dilemma_Prisoners_Dilemma_Qwen2.5-3B-Instruct_20250510_221101/exp_results.json"
    ]
    if not result_files:
        logger.error("No experiment result files found!")
        return
    
    logger.info(f"Found {len(result_files)} experiment result files")
    
    # Load data
    all_data = analyzer.load_data(result_files[:5])  # Limit to first 5 files for testing
    
    if not all_data:
        logger.error("No data loaded!")
        return
    
    # Filter for Option 1 decisions
    option1_data = analyzer.filter_option1_decisions(all_data)
    
    if not option1_data:
        logger.error("No Option 1 decisions found!")
        return
    
    # Analyze patterns
    results = analyzer.analyze_patterns(option1_data)
    
    # Save results
    output_dir = 'result_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer.save_results(results, os.path.join(output_dir, 'option_pattern_analysis_results.json'))
    
    # Generate visualizations
    viz_dir = os.path.join(output_dir, 'option_pattern_visualizations')
    analyzer.generate_visualizations(results, viz_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("OPTION PATTERN ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nTotal cases analyzed: {len(option1_data)}")
    print(f"Option pairs extracted: {len(analyzer.option1_texts)}")
    
    print(f"\nOption 1 - Top 5 words:")
    for word, freq in results['option1_patterns']['word_frequency'][:5]:
        print(f"  {word}: {freq}")
    
    print(f"\nOption 2 - Top 5 words:")
    for word, freq in results['option2_patterns']['word_frequency'][:5]:
        print(f"  {word}: {freq}")
    
    print(f"\nOption 1 distinctive words:")
    for word, score in results['comparative_analysis']['option1_distinctive_words'][-5:]:
        print(f"  {word}: {score:.3f}")
    
    print(f"\nOption 2 distinctive words:")
    for word, score in results['comparative_analysis']['option2_distinctive_words'][-5:]:
        print(f"  {word}: {score:.3f}")
    
    if 'keyword_analysis' in results:
        print(f"\nKeyword analysis - Option 1:")
        for category, count in results['keyword_analysis']['option1_keywords'].items():
            print(f"  {category}: {count}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main() 