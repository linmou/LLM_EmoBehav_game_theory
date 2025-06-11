#!/usr/bin/env python3
"""
Unit tests for Option Pattern Analysis Script

This script tests the functionality of the OptionPatternAnalyzer class
using Python's unittest framework.
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add the current directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from option_pattern_analysis import OptionPatternAnalyzer


class TestOptionPatternAnalyzer(unittest.TestCase):
    """Test cases for OptionPatternAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = OptionPatternAnalyzer()
        
        # Create sample test data
        self.sample_data = [
            {
                "emotion": "anger",
                "intensity": 1.5,
                "scenario": "Test Scenario 1",
                "description": "A test scenario description",
                "input": "Option 1. Invest in community project\\nOption 2. Focus on personal gains\\nresponse in json format",
                "output": "test output",
                "rationale": "Community investment benefits everyone",
                "decision": "Option 1. Invest in community project",
                "category": 1
            },
            {
                "emotion": "neutral",
                "intensity": 0.0,
                "scenario": "Test Scenario 2", 
                "description": "Another test scenario",
                "input": "Option 1. Collaborate with others\\nOption 2. Work independently\\nresponse in json format",
                "output": "test output 2",
                "rationale": "Collaboration leads to better outcomes",
                "decision": "Option 1. Collaborate with others",
                "category": 1
            },
            {
                "emotion": "anger",
                "intensity": 1.0,
                "scenario": "Test Scenario 3",
                "description": "Third test scenario",
                "input": "Option 1. Share resources\\nOption 2. Keep resources private\\nresponse in json format",
                "output": "test output 3",
                "rationale": "Private resources are more secure",
                "decision": "Option 2. Keep resources private",
                "category": 2
            }
        ]
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test OptionPatternAnalyzer initialization."""
        self.assertIsInstance(self.analyzer.stop_words, set)
        self.assertEqual(len(self.analyzer.option1_texts), 0)
        self.assertEqual(len(self.analyzer.option2_texts), 0)
        self.assertEqual(len(self.analyzer.scenarios), 0)
        self.assertEqual(len(self.analyzer.emotions), 0)
    
    def test_extract_options_from_input(self):
        """Test extraction of Option 1 and Option 2 from input text."""
        input_text = "Option 1. Invest in community project\\nOption 2. Focus on personal gains\\nresponse in json format"
        
        option1, option2 = self.analyzer.extract_options_from_input(input_text)
        
        self.assertEqual(option1, "Invest in community project")
        self.assertEqual(option2, "Focus on personal gains")
    
    def test_extract_options_from_input_empty(self):
        """Test extraction with empty or malformed input."""
        # Test with empty input
        option1, option2 = self.analyzer.extract_options_from_input("")
        self.assertEqual(option1, "")
        self.assertEqual(option2, "")
        
        # Test with malformed input
        option1, option2 = self.analyzer.extract_options_from_input("No options here")
        self.assertEqual(option1, "")
        self.assertEqual(option2, "")
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality."""
        text = "This is a TEST text with Numbers123 and Special@Characters!"
        
        tokens = self.analyzer.preprocess_text(text)
        
        # Check that tokens are lowercase and cleaned
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(token.islower() for token in tokens))
        self.assertTrue(all(token.isalpha() for token in tokens))
        # Should not contain stopwords or short words
        self.assertNotIn("is", tokens)  # stopword
        self.assertNotIn("a", tokens)   # short word
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction."""
        text = "This is a sample text for testing. It has multiple sentences!"
        
        features = self.analyzer.extract_linguistic_features(text)
        
        # Check that required features are present
        required_features = ['word_count', 'sentence_count', 'avg_word_length', 
                           'verb_count', 'verb_ratio', 'emotional_word_ratio']
        
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
        
        # Check reasonable values
        self.assertGreater(features['word_count'], 0)
        self.assertGreater(features['sentence_count'], 0)
        self.assertGreaterEqual(features['verb_ratio'], 0)
        self.assertLessEqual(features['verb_ratio'], 1)
    
    def test_load_data(self):
        """Test data loading from JSON files."""
        # Create a temporary JSON file
        test_file = os.path.join(self.temp_dir, "test_data.json")
        with open(test_file, 'w') as f:
            json.dump(self.sample_data, f)
        
        # Test loading
        loaded_data = self.analyzer.load_data([test_file])
        
        self.assertEqual(len(loaded_data), len(self.sample_data))
        self.assertEqual(loaded_data[0]['scenario'], 'Test Scenario 1')
    
    def test_load_data_nonexistent_file(self):
        """Test loading data from non-existent file."""
        loaded_data = self.analyzer.load_data(['nonexistent_file.json'])
        self.assertEqual(len(loaded_data), 0)
    
    def test_filter_option1_decisions(self):
        """Test filtering for Option 1 decisions."""
        option1_cases = self.analyzer.filter_option1_decisions(self.sample_data)
        
        # Should return 2 cases (first two have Option 1 decisions)
        self.assertEqual(len(option1_cases), 2)
        
        # Check that all returned cases have Option 1 decisions
        for case in option1_cases:
            decision = case['decision']
            self.assertTrue('Option 1' in decision or 
                          decision.startswith('Invest') or 
                          decision.startswith('Collaborate'))
    
    def test_analyze_text_patterns(self):
        """Test text pattern analysis."""
        test_texts = [
            "Invest in community project for better outcomes",
            "Collaborate with others to achieve goals",
            "Share resources for mutual benefit"
        ]
        
        patterns = self.analyzer.analyze_text_patterns(test_texts, "Test Option")
        
        # Check required keys
        required_keys = ['word_frequency', 'bigrams', 'trigrams', 'sentence_starters',
                        'total_texts', 'avg_length', 'unique_words']
        
        for key in required_keys:
            self.assertIn(key, patterns)
        
        # Check data types and reasonable values
        self.assertEqual(patterns['total_texts'], 3)
        self.assertGreater(patterns['avg_length'], 0)
        self.assertGreater(patterns['unique_words'], 0)
        self.assertIsInstance(patterns['word_frequency'], list)
        self.assertIsInstance(patterns['bigrams'], list)
    
    def test_analyze_keywords(self):
        """Test keyword analysis functionality."""
        # Set up test data
        self.analyzer.option1_texts = ["Invest in community collaboration"]
        self.analyzer.option2_texts = ["Focus on personal profit and gain"]
        
        keyword_analysis = self.analyzer.analyze_keywords()
        
        # Check structure
        self.assertIn('option1_keywords', keyword_analysis)
        self.assertIn('option2_keywords', keyword_analysis)
        self.assertIn('keyword_categories', keyword_analysis)
        
        # Check that keyword categories are present
        expected_categories = ['cooperation', 'competition', 'risk', 'benefit', 'action', 'avoidance']
        for category in expected_categories:
            self.assertIn(category, keyword_analysis['option1_keywords'])
            self.assertIn(category, keyword_analysis['option2_keywords'])
    
    def test_compare_options(self):
        """Test option comparison functionality."""
        # Set up test data
        self.analyzer.option1_texts = ["Community investment collaboration"]
        self.analyzer.option2_texts = ["Personal profit individual gain"]
        
        comparison = self.analyzer.compare_options()
        
        # Check required keys
        required_keys = ['option1_distinctive_words', 'option2_distinctive_words',
                        'option1_avg_length', 'option2_avg_length']
        
        for key in required_keys:
            self.assertIn(key, comparison)
        
        # Check data types
        self.assertIsInstance(comparison['option1_distinctive_words'], list)
        self.assertIsInstance(comparison['option2_distinctive_words'], list)
        self.assertIsInstance(comparison['option1_avg_length'], (int, float))
        self.assertIsInstance(comparison['option2_avg_length'], (int, float))
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_visualizations(self, mock_close, mock_savefig):
        """Test visualization generation."""
        # Set up test data
        self.analyzer.option1_texts = ["test text one"]
        self.analyzer.option2_texts = ["test text two"]
        
        # Create mock results
        mock_results = {
            'option1_patterns': {
                'word_frequency': [('test', 5), ('word', 3)]
            },
            'option2_patterns': {
                'word_frequency': [('another', 4), ('word', 2)]
            },
            'keyword_analysis': {
                'option1_keywords': {'cooperation': 1, 'competition': 0},
                'option2_keywords': {'cooperation': 0, 'competition': 1}
            }
        }
        
        # Test visualization generation
        viz_dir = os.path.join(self.temp_dir, 'viz')
        self.analyzer.generate_visualizations(mock_results, viz_dir)
        
        # Check that directory was created
        self.assertTrue(os.path.exists(viz_dir))
        
        # Check that savefig was called (indicating plots were generated)
        self.assertTrue(mock_savefig.called)
    
    def test_save_results(self):
        """Test saving results to JSON file."""
        test_results = {
            'test_key': 'test_value',
            'test_number': 42,
            'test_list': [1, 2, 3]
        }
        
        output_file = os.path.join(self.temp_dir, 'test_results.json')
        self.analyzer.save_results(test_results, output_file)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check that content is correct
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results, test_results)
    
    def test_analyze_patterns_integration(self):
        """Test the full analyze_patterns method integration."""
        # Filter for Option 1 decisions
        option1_data = self.analyzer.filter_option1_decisions(self.sample_data)
        
        # Run analysis
        results = self.analyzer.analyze_patterns(option1_data)
        
        # Check that all expected sections are present
        expected_sections = ['option1_patterns', 'option2_patterns', 'comparative_analysis',
                           'linguistic_features', 'keyword_analysis', 'theme_analysis']
        
        for section in expected_sections:
            self.assertIn(section, results)
        
        # Check that data was extracted
        self.assertGreater(len(self.analyzer.option1_texts), 0)
        self.assertGreater(len(self.analyzer.option2_texts), 0)
        self.assertEqual(len(self.analyzer.option1_texts), len(self.analyzer.option2_texts))


class TestOptionPatternAnalysisIntegration(unittest.TestCase):
    """Integration tests for the complete option pattern analysis workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create more comprehensive test data
        self.comprehensive_data = []
        scenarios = ["Solar Initiative", "Fishing Quota", "Community Garden"]
        
        for i, scenario in enumerate(scenarios):
            for emotion in ["anger", "neutral"]:
                for decision_type in ["Option 1", "Option 2"]:
                    item = {
                        "emotion": emotion,
                        "intensity": 1.0 if emotion == "anger" else 0.0,
                        "scenario": scenario,
                        "description": f"Test scenario {scenario}",
                        "input": f"Option 1. Contribute to {scenario.lower()}\\nOption 2. Focus on personal interests\\nresponse",
                        "rationale": f"Rationale for {decision_type} in {scenario}",
                        "decision": f"{decision_type}. {'Contribute' if decision_type == 'Option 1' else 'Focus'} on {scenario.lower()}",
                        "category": 1 if decision_type == "Option 1" else 2
                    }
                    self.comprehensive_data.append(item)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_analysis_workflow(self):
        """Test the complete analysis workflow from data loading to results."""
        # Create test data file
        test_file = os.path.join(self.temp_dir, "comprehensive_test.json")
        with open(test_file, 'w') as f:
            json.dump(self.comprehensive_data, f)
        
        # Initialize analyzer
        analyzer = OptionPatternAnalyzer()
        
        # Load data
        all_data = analyzer.load_data([test_file])
        self.assertGreater(len(all_data), 0)
        
        # Filter for Option 1 decisions
        option1_data = analyzer.filter_option1_decisions(all_data)
        self.assertGreater(len(option1_data), 0)
        
        # Run full analysis
        results = analyzer.analyze_patterns(option1_data)
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('option1_patterns', results)
        self.assertIn('option2_patterns', results)
        
        # Save results
        results_file = os.path.join(self.temp_dir, "test_results.json")
        analyzer.save_results(results, results_file)
        self.assertTrue(os.path.exists(results_file))
        
        # Generate visualizations (mock the plotting to avoid display issues)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            viz_dir = os.path.join(self.temp_dir, 'visualizations')
            analyzer.generate_visualizations(results, viz_dir)
            self.assertTrue(os.path.exists(viz_dir))


def run_tests():
    """Run all tests and return the results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestOptionPatternAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestOptionPatternAnalysisIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("Running Option Pattern Analysis Tests...")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll tests passed successfully! ✅")
    else:
        print("\nSome tests failed. ❌")
    
    exit(0 if result.wasSuccessful() else 1) 