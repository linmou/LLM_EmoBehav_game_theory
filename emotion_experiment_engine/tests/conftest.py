#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for emotion memory experiments.
"""

import json
import pickle
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig, BenchmarkItem


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
FIXTURES_DIR = TEST_DATA_DIR / "fixtures"
REALISTIC_DIR = TEST_DATA_DIR / "realistic"


@pytest.fixture(scope="session")
def test_data_dir():
    """Root test data directory"""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")  
def fixtures_dir():
    """Test fixtures directory"""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def realistic_dir():
    """Realistic test data directory"""
    return REALISTIC_DIR


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_benchmark_config():
    """Standard mock benchmark configuration"""
    return BenchmarkConfig(
        name="test_benchmark",
        task_type="test_task",
        data_path=None,
        base_data_dir="test_data",
        sample_limit=5,
        llm_eval_config={
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    )


@pytest.fixture
def mock_experiment_config(temp_dir):
    """Standard mock experiment configuration"""
    return ExperimentConfig(
        model_path="mock_model",
        emotions=["anger", "neutral"],
        intensities=[0.5, 1.0],
        benchmark=BenchmarkConfig(
            name="infinitebench",
            task_type="passkey", 
            data_path=temp_dir / "test.jsonl",
            sample_limit=3
        ),
        output_dir=temp_dir,
        batch_size=1
    )


@pytest.fixture
def minimal_passkey_data():
    """Minimal passkey test data"""
    return [
        {
            "id": 0,
            "input": "What is the passkey?",
            "answer": "12345",
            "context": "The passkey is 12345 hidden in this context.",
            "task_name": "passkey"
        },
        {
            "id": 1, 
            "input": "What is the passkey?",
            "answer": "67890",
            "context": "The passkey is 67890 hidden in this context.",
            "task_name": "passkey"
        }
    ]


@pytest.fixture
def minimal_qa_data():
    """Minimal QA test data"""
    return [
        {
            "id": "test_0",
            "input": "What is machine learning?",
            "answers": ["Machine learning is a subset of AI"],
            "context": "Machine learning is a subset of artificial intelligence.",
            "task_name": "qa"
        }
    ]


@pytest.fixture
def create_temp_jsonl_file():
    """Factory for creating temporary JSONL files"""
    def _create_file(data: List[Dict], temp_dir: Path) -> Path:
        file_path = temp_dir / "test_data.jsonl"
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return file_path
    return _create_file


@pytest.fixture
def create_temp_json_file():
    """Factory for creating temporary JSON files"""
    def _create_file(data: List[Dict], temp_dir: Path) -> Path:
        file_path = temp_dir / "test_data.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path
    return _create_file


@pytest.fixture
def mock_llm_evaluation():
    """Mock LLM evaluation response"""
    def _mock_response(response: str, ground_truth: Any, task: str) -> Dict[str, Any]:
        # Simple mock logic
        if isinstance(ground_truth, str):
            score = 1.0 if response.strip().lower() == ground_truth.lower() else 0.0
        else:
            score = 0.5  # Default for complex cases
        
        return {
            "emotion": "neutral",
            "confidence": score,
            "reasoning": "Mock evaluation"
        }
    return _mock_response


@pytest.fixture
def mock_neural_components():
    """Mock neural manipulation components"""
    with patch('emotion_experiment_engine.experiment.load_emotion_readers') as mock_readers, \
         patch('emotion_experiment_engine.experiment.setup_model_and_tokenizer') as mock_setup, \
         patch('emotion_experiment_engine.experiment.get_pipeline') as mock_pipeline:
        
        # Configure mocks
        mock_readers.return_value = {"anger": MagicMock(), "neutral": MagicMock()}
        mock_setup.return_value = (MagicMock(), MagicMock())
        mock_pipeline.return_value = MagicMock()
        
        yield {
            "readers": mock_readers,
            "setup": mock_setup, 
            "pipeline": mock_pipeline
        }


# Performance tracking fixture
class TestPerformanceTracker:
    """Track test execution times"""
    
    def __init__(self):
        self.results = {}
        
    def track(self, test_name: str, duration: float, category: str = "unknown"):
        """Track test performance"""
        if category not in self.results:
            self.results[category] = []
        
        self.results[category].append({
            "test": test_name,
            "duration": duration,
            "timestamp": time.time()
        })
        
    def get_stats(self, category: str = None) -> Dict:
        """Get performance statistics"""
        if category:
            data = self.results.get(category, [])
        else:
            data = [item for items in self.results.values() for item in items]
            
        if not data:
            return {}
            
        durations = [item["duration"] for item in data]
        return {
            "count": len(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
            "min": min(durations)
        }


@pytest.fixture(scope="session")
def performance_tracker():
    """Global performance tracker"""
    return TestPerformanceTracker()


@pytest.fixture(autouse=True)
def track_test_performance(request, performance_tracker):
    """Automatically track test performance"""
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    # Determine test category from path
    test_path = request.node.nodeid
    if "/unit/" in test_path:
        category = "unit"
    elif "/integration/" in test_path:
        category = "integration"
    elif "/regression/" in test_path:
        category = "regression"
    elif "/e2e/" in test_path:
        category = "e2e"
    elif "/priorities/" in test_path:
        category = "critical"
    elif "test_answer_wrapper_comprehensive.py" in test_path:
        category = "comprehensive"  # Special category for comprehensive test
    else:
        category = "other"
        
    performance_tracker.track(request.node.name, duration, category)
    
    # Warn on slow tests
    thresholds = {
        "unit": 1.0,
        "integration": 10.0,
        "regression": 30.0,
        "e2e": 60.0
    }
    
    if category in thresholds and duration > thresholds[category]:
        warnings.warn(
            f"Slow test: {request.node.name} took {duration:.2f}s "
            f"(threshold: {thresholds[category]}s)"
        )


# Pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration  
pytest.mark.regression = pytest.mark.regression
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.critical = pytest.mark.critical
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu

# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and ordering"""
    
    for item in items:
        # Auto-add markers based on test path
        test_path = str(item.fspath)
        
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/regression/" in test_path:
            item.add_marker(pytest.mark.regression)
        elif "/e2e/" in test_path:
            item.add_marker(pytest.mark.e2e)
            
        # Add critical marker for research-critical tests
        if "critical" in test_path or "golden" in test_path:
            item.add_marker(pytest.mark.critical)
            
        # Add slow marker for tests that might be slow
        if any(slow_keyword in item.name.lower() for slow_keyword in 
               ["e2e", "integration", "performance", "stress"]):
            item.add_marker(pytest.mark.slow)


# Session-scoped test data setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_data_directories():
    """Ensure test data directories exist"""
    for directory in [TEST_DATA_DIR, FIXTURES_DIR, REALISTIC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test data files if they don't exist
    minimal_passkey = FIXTURES_DIR / "minimal_passkey.jsonl"
    if not minimal_passkey.exists():
        data = [
            {"id": 0, "input": "What is the passkey?", "answer": "12345", "context": "The passkey is 12345"},
            {"id": 1, "input": "What is the passkey?", "answer": "67890", "context": "The passkey is 67890"}
        ]
        with open(minimal_passkey, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')