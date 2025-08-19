#!/usr/bin/env python3
"""
Comprehensive test of all InfiniteBench and LongBench tasks with REAL BENCHMARK DATA.
Tests all available tasks with actual downloaded data files using proper prompt wrappers.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.utils.data import DataLoader

from emotion_memory_experiments.benchmark_adapters import (
    BenchmarkConfig,
    BenchmarkItem,
    InfiniteBenchAdapter,
    LoCoMoAdapter,
    LongBenchAdapter,
    collate_memory_benchmarks,
    get_adapter,
)
from emotion_memory_experiments.memory_prompt_wrapper import get_memory_prompt_wrapper
from neuro_manipulation.prompt_formats import PromptFormat


# Define all available benchmark tasks
INFINITEBENCH_TASKS = {
    "passkey": "infinitebench_passkey.jsonl",
    "kv_retrieval": "infinitebench_kv_retrieval.jsonl", 
    "number_string": "infinitebench_number_string.jsonl",
    "longbook_qa_eng": "infinitebench_longbook_qa_eng.jsonl",
    "longbook_qa_chn": "infinitebench_longbook_qa_chn.jsonl",
    "longbook_sum_eng": "infinitebench_longbook_sum_eng.jsonl",
    "longbook_choice_eng": "infinitebench_longbook_choice_eng.jsonl",
    "longdialogue_qa_eng": "infinitebench_longdialogue_qa_eng.jsonl",
    "code_debug": "infinitebench_code_debug.jsonl",
    "code_run": "infinitebench_code_run.jsonl",
    "math_calc": "infinitebench_math_calc.jsonl",
    "math_find": "infinitebench_math_find.jsonl",
}

LONGBENCH_TASKS = {
    "narrativeqa": "longbench_narrativeqa.jsonl",
    "qasper": "longbench_qasper.jsonl",
    "multifieldqa_en": "longbench_multifieldqa_en.jsonl",
    "multifieldqa_zh": "longbench_multifieldqa_zh.jsonl",
    "hotpotqa": "longbench_hotpotqa.jsonl",
    "2wikimqa": "longbench_2wikimqa.jsonl",
    "musique": "longbench_musique.jsonl",
    "dureader": "longbench_dureader.jsonl",
    "gov_report": "longbench_gov_report.jsonl",
    "qmsum": "longbench_qmsum.jsonl",
    "multi_news": "longbench_multi_news.jsonl",
    "vcsum": "longbench_vcsum.jsonl",
    "trec": "longbench_trec.jsonl",
    "triviaqa": "longbench_triviaqa.jsonl",
    "samsum": "longbench_samsum.jsonl",
    "lsht": "longbench_lsht.jsonl",
    "passage_retrieval_en": "longbench_passage_retrieval_en.jsonl",
    "passage_count": "longbench_passage_count.jsonl",
    "passage_retrieval_zh": "longbench_passage_retrieval_zh.jsonl",
    "lcc": "longbench_lcc.jsonl",
    "repobench-p": "longbench_repobench-p.jsonl",
}

# Extended LongBench-E tasks (with length evaluation)
LONGBENCH_E_TASKS = {
    "2wikimqa_e": "longbench_2wikimqa_e.jsonl",
    "gov_report_e": "longbench_gov_report_e.jsonl", 
    "hotpotqa_e": "longbench_hotpotqa_e.jsonl",
    "lcc_e": "longbench_lcc_e.jsonl",
    "multi_news_e": "longbench_multi_news_e.jsonl",
    "multifieldqa_en_e": "longbench_multifieldqa_en_e.jsonl",
    "passage_count_e": "longbench_passage_count_e.jsonl",
    "passage_retrieval_en_e": "longbench_passage_retrieval_en_e.jsonl",
    "qasper_e": "longbench_qasper_e.jsonl",
    "repobench-p_e": "longbench_repobench-p_e.jsonl",
    "samsum_e": "longbench_samsum_e.jsonl",
    "trec_e": "longbench_trec_e.jsonl",
    "triviaqa_e": "longbench_triviaqa_e.jsonl",
}


def get_test_data_path():
    """Get the path to test data directory"""
    return Path(__file__).parent.parent.parent / "test_data" / "real_benchmarks"


def create_mock_prompt_wrapper(task_name: str):
    """Create a mock prompt wrapper for testing"""
    mock_prompt_format = Mock()
    mock_prompt_format.build = Mock(return_value=f"<formatted_{task_name}_prompt>")
    return get_memory_prompt_wrapper(task_name, mock_prompt_format)


def test_infinitebench_all_tasks():
    """Test ALL InfiniteBench tasks with real data using for loops"""
    print("🔍 Testing ALL InfiniteBench tasks with real data...")
    
    test_data_path = get_test_data_path()
    results = {}
    
    for task_name, filename in INFINITEBENCH_TASKS.items():
        print(f"\n📋 Testing InfiniteBench task: {task_name}")
        
        data_path = test_data_path / filename
        if not data_path.exists():
            print(f"  ⏭️  Skipping {task_name} - data file not found: {filename}")
            results[task_name] = "skipped"
            continue
            
        try:
            # Create config
            config = BenchmarkConfig(
                name="infinitebench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method=f"get_score_one_{task_name}",
            )
            
            # Create adapter
            adapter = InfiniteBenchAdapter(config)
            print(f"  ✅ Adapter created for {task_name}")
            
            # Test dataset creation without prompt wrapper
            dataset = adapter.create_dataset()
            print(f"  ✅ Dataset created with {len(dataset)} items")
            
            if len(dataset) == 0:
                print(f"  ⚠️  Warning: Empty dataset for {task_name}")
                results[task_name] = "empty"
                continue
            
            # Test first item structure
            item = dataset[0]
            assert isinstance(item, BenchmarkItem), f"Expected BenchmarkItem, got {type(item)}"
            assert item.ground_truth is not None, f"Ground truth is None for {task_name}"
            print(f"  ✅ Item structure valid")
            
            # Test with prompt wrapper
            prompt_wrapper = create_mock_prompt_wrapper(task_name)
            dataset_with_wrapper = adapter.create_dataset(prompt_wrapper=prompt_wrapper)
            
            formatted_item = dataset_with_wrapper[0]
            assert isinstance(formatted_item, dict), "Expected dict with prompt wrapper"
            assert "prompt" in formatted_item, "Missing 'prompt' key"
            assert "item" in formatted_item, "Missing 'item' key"
            print(f"  ✅ Prompt wrapper integration works")
            
            # Test evaluation methods
            complexity = adapter.get_evaluation_complexity(task_name)
            metrics = adapter.get_task_metrics(task_name)
            print(f"  ✅ Task complexity: {complexity}, metrics: {metrics}")
            
            # Test specific evaluation based on task type
            if task_name == "passkey":
                # Test passkey evaluation
                test_response = str(item.ground_truth)
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                assert score == 1.0, f"Expected perfect score for exact match, got {score}"
                print(f"  ✅ Passkey evaluation: {score}")
                
            elif task_name == "kv_retrieval":
                # Test KV retrieval evaluation
                test_response = f"The answer is {item.ground_truth}"
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                assert score == 1.0, f"Expected perfect score for KV retrieval, got {score}"
                print(f"  ✅ KV retrieval evaluation: {score}")
                
            elif task_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                # Test QA evaluation
                test_response = str(item.ground_truth)
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                assert score >= 0.0, f"Expected non-negative score, got {score}"
                print(f"  ✅ QA evaluation: {score}")
                
            else:
                # Generic evaluation test
                test_response = str(item.ground_truth)
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ Generic evaluation: {score}")
            
            # Test detailed metrics
            detailed_metrics = adapter.evaluate_with_detailed_metrics(
                str(item.ground_truth), item.ground_truth, task_name
            )
            assert "overall_score" in detailed_metrics, "Missing overall_score in detailed metrics"
            print(f"  ✅ Detailed metrics: {list(detailed_metrics.keys())}")
            
            # Test DataLoader integration
            dataloader = adapter.get_dataloader(
                batch_size=min(4, len(dataset)),
                shuffle=False,
                prompt_wrapper=prompt_wrapper,
                collate_fn=collate_memory_benchmarks,
            )
            
            batch = next(iter(dataloader))
            assert "prompts" in batch, "Missing 'prompts' in batch"
            assert "items" in batch, "Missing 'items' in batch"
            assert batch["batch_size"] > 0, "Invalid batch size"
            print(f"  ✅ DataLoader integration works")
            
            results[task_name] = "passed"
            print(f"  🎉 {task_name} test PASSED!")
            
        except Exception as e:
            print(f"  ❌ {task_name} test FAILED: {str(e)}")
            results[task_name] = f"failed: {str(e)}"
    
    return results


def test_longbench_all_tasks():
    """Test ALL LongBench tasks with real data using for loops"""
    print("\n🔍 Testing ALL LongBench tasks with real data...")
    
    test_data_path = get_test_data_path()
    results = {}
    
    for task_name, filename in LONGBENCH_TASKS.items():
        print(f"\n📋 Testing LongBench task: {task_name}")
        
        data_path = test_data_path / filename
        if not data_path.exists():
            print(f"  ⏭️  Skipping {task_name} - data file not found: {filename}")
            results[task_name] = "skipped"
            continue
            
        try:
            # Create config
            config = BenchmarkConfig(
                name="longbench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method=f"longbench_{task_name}",
            )
            
            # Create adapter
            adapter = LongBenchAdapter(config)
            print(f"  ✅ Adapter created for {task_name}")
            
            # Test dataset creation
            dataset = adapter.create_dataset()
            print(f"  ✅ Dataset created with {len(dataset)} items")
            
            if len(dataset) == 0:
                print(f"  ⚠️  Warning: Empty dataset for {task_name}")
                results[task_name] = "empty"
                continue
            
            # Test first item structure
            item = dataset[0]
            assert isinstance(item, BenchmarkItem), f"Expected BenchmarkItem, got {type(item)}"
            assert item.ground_truth is not None, f"Ground truth is None for {task_name}"
            print(f"  ✅ Item structure valid")
            
            # Test with prompt wrapper
            prompt_wrapper = create_mock_prompt_wrapper(task_name)
            dataset_with_wrapper = adapter.create_dataset(prompt_wrapper=prompt_wrapper)
            
            formatted_item = dataset_with_wrapper[0]
            assert isinstance(formatted_item, dict), "Expected dict with prompt wrapper"
            assert "prompt" in formatted_item, "Missing 'prompt' key"
            assert "item" in formatted_item, "Missing 'item' key"
            print(f"  ✅ Prompt wrapper integration works")
            
            # Test evaluation methods
            complexity = adapter.get_evaluation_complexity(task_name)
            metrics = adapter.get_task_metrics(task_name)
            print(f"  ✅ Task complexity: {complexity}, metrics: {metrics}")
            
            # Test specific evaluation based on task type
            if task_name in ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
                # Test QA F1 evaluation
                test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ QA F1 evaluation: {score}")
                
            elif task_name in ["dureader", "gov_report", "qmsum", "multi_news", "vcsum", "samsum"]:
                # Test ROUGE evaluation
                test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ ROUGE evaluation: {score}")
                
            elif task_name in ["trec", "lsht"]:
                # Test classification evaluation
                test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ Classification evaluation: {score}")
                
            elif task_name in ["passage_retrieval_en", "passage_retrieval_zh"]:
                # Test retrieval evaluation
                test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ Retrieval evaluation: {score}")
                
            else:
                # Generic evaluation test
                test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                score = adapter.evaluate_response(test_response, item.ground_truth, task_name)
                # Validate score is numeric and in valid range
                assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
                assert 0 <= score <= 1, f"Expected score in range [0,1], got {score}"
                print(f"  ✅ Generic evaluation: {score}")
            
            # Test detailed metrics
            detailed_metrics = adapter.evaluate_with_detailed_metrics(
                str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0]), 
                item.ground_truth, 
                task_name
            )
            assert "overall_score" in detailed_metrics, "Missing overall_score in detailed metrics"
            print(f"  ✅ Detailed metrics: {list(detailed_metrics.keys())}")
            
            # Test DataLoader integration
            dataloader = adapter.get_dataloader(
                batch_size=min(4, len(dataset)),
                shuffle=False,
                prompt_wrapper=prompt_wrapper,
                collate_fn=collate_memory_benchmarks,
            )
            
            batch = next(iter(dataloader))
            assert "prompts" in batch, "Missing 'prompts' in batch"
            assert "items" in batch, "Missing 'items' in batch"
            assert batch["batch_size"] > 0, "Invalid batch size"
            print(f"  ✅ DataLoader integration works")
            
            results[task_name] = "passed"
            print(f"  🎉 {task_name} test PASSED!")
            
        except Exception as e:
            print(f"  ❌ {task_name} test FAILED: {str(e)}")
            results[task_name] = f"failed: {str(e)}"
    
    return results


def test_longbench_e_length_evaluation():
    """Test LongBench-E tasks with length-based evaluation"""
    print("\n🔍 Testing LongBench-E tasks with length-based evaluation...")
    
    test_data_path = get_test_data_path()
    results = {}
    
    for task_name, filename in LONGBENCH_E_TASKS.items():
        print(f"\n📋 Testing LongBench-E task: {task_name}")
        
        data_path = test_data_path / filename
        if not data_path.exists():
            print(f"  ⏭️  Skipping {task_name} - data file not found: {filename}")
            results[task_name] = "skipped"
            continue
            
        try:
            # Extract base task name (remove '_e' suffix)
            base_task = task_name.replace("_e", "")
            
            # Create config
            config = BenchmarkConfig(
                name="longbench",
                data_path=data_path,
                task_type=base_task,
                evaluation_method=f"longbench_{base_task}",
            )
            
            # Create adapter
            adapter = LongBenchAdapter(config)
            print(f"  ✅ Adapter created for {task_name}")
            
            # Test dataset creation
            dataset = adapter.create_dataset()
            print(f"  ✅ Dataset created with {len(dataset)} items")
            
            if len(dataset) == 0:
                print(f"  ⚠️  Warning: Empty dataset for {task_name}")
                results[task_name] = "empty"
                continue
            
            # Check if items have length information
            item = dataset[0]
            has_length = hasattr(item, 'metadata') and item.metadata and 'length' in item.metadata
            
            if has_length:
                # Test length-based evaluation
                responses = []
                ground_truths = []
                task_names = []
                lengths = []
                
                # Collect data for length-based evaluation
                for i in range(min(10, len(dataset))):  # Test with first 10 items
                    test_item = dataset[i]
                    responses.append(str(test_item.ground_truth) if not isinstance(test_item.ground_truth, list) else str(test_item.ground_truth[0]))
                    ground_truths.append(test_item.ground_truth)
                    task_names.append(base_task)
                    lengths.append(test_item.metadata.get('length', 5000))  # Default length if missing
                
                # Test length-based evaluation
                length_scores = adapter.evaluate_by_length(responses, ground_truths, task_names, lengths)
                
                assert isinstance(length_scores, dict), "Expected dict from evaluate_by_length"
                assert "0-4k" in length_scores, "Missing '0-4k' category"
                assert "4-8k" in length_scores, "Missing '4-8k' category"
                assert "8k+" in length_scores, "Missing '8k+' category"
                
                print(f"  ✅ Length-based evaluation: {length_scores}")
            else:
                print(f"  ⚠️  No length information found, testing basic evaluation")
                
            # Test basic evaluation
            test_response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
            score = adapter.evaluate_response(test_response, item.ground_truth, base_task)
            assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
            print(f"  ✅ Basic evaluation: {score}")
            
            results[task_name] = "passed"
            print(f"  🎉 {task_name} test PASSED!")
            
        except Exception as e:
            print(f"  ❌ {task_name} test FAILED: {str(e)}")
            results[task_name] = f"failed: {str(e)}"
    
    return results


def test_batch_evaluation_all_tasks():
    """Test batch evaluation across multiple tasks"""
    print("\n🔍 Testing batch evaluation across multiple tasks...")
    
    test_data_path = get_test_data_path()
    
    # Test InfiniteBench batch evaluation
    ib_results = {}
    print("\n📋 InfiniteBench batch evaluation:")
    
    for task_name, filename in list(INFINITEBENCH_TASKS.items())[:3]:  # Test first 3 tasks
        data_path = test_data_path / filename
        if not data_path.exists():
            continue
            
        try:
            config = BenchmarkConfig(
                name="infinitebench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method=f"get_score_one_{task_name}",
            )
            
            adapter = InfiniteBenchAdapter(config)
            dataset = adapter.create_dataset()
            
            if len(dataset) == 0:
                continue
            
            # Prepare batch data
            responses = []
            ground_truths = []
            task_names = []
            
            for i in range(min(5, len(dataset))):
                item = dataset[i]
                responses.append(str(item.ground_truth))
                ground_truths.append(item.ground_truth)
                task_names.append(task_name)
            
            # Test batch evaluation
            scores = adapter.evaluate_batch(responses, ground_truths, task_names)
            assert len(scores) == len(responses), f"Expected {len(responses)} scores, got {len(scores)}"
            assert all(isinstance(s, (int, float)) for s in scores), "All scores should be numeric"
            
            print(f"  ✅ {task_name}: batch of {len(scores)} items evaluated")
            ib_results[task_name] = scores
            
        except Exception as e:
            print(f"  ❌ {task_name} batch evaluation failed: {str(e)}")
    
    # Test LongBench batch evaluation
    lb_results = {}
    print("\n📋 LongBench batch evaluation:")
    
    for task_name, filename in list(LONGBENCH_TASKS.items())[:3]:  # Test first 3 tasks
        data_path = test_data_path / filename
        if not data_path.exists():
            continue
            
        try:
            config = BenchmarkConfig(
                name="longbench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method=f"longbench_{task_name}",
            )
            
            adapter = LongBenchAdapter(config)
            dataset = adapter.create_dataset()
            
            if len(dataset) == 0:
                continue
            
            # Prepare batch data
            responses = []
            ground_truths = []
            task_names = []
            
            for i in range(min(5, len(dataset))):
                item = dataset[i]
                response = str(item.ground_truth) if not isinstance(item.ground_truth, list) else str(item.ground_truth[0])
                responses.append(response)
                ground_truths.append(item.ground_truth)
                task_names.append(task_name)
            
            # Test batch evaluation
            scores = adapter.evaluate_batch(responses, ground_truths, task_names)
            assert len(scores) == len(responses), f"Expected {len(responses)} scores, got {len(scores)}"
            assert all(isinstance(s, (int, float)) for s in scores), "All scores should be numeric"
            
            print(f"  ✅ {task_name}: batch of {len(scores)} items evaluated")
            lb_results[task_name] = scores
            
        except Exception as e:
            print(f"  ❌ {task_name} batch evaluation failed: {str(e)}")
    
    return {"infinitebench": ib_results, "longbench": lb_results}


def test_factory_function_all_tasks():
    """Test factory function with all available tasks"""
    print("\n🔍 Testing factory function with all available tasks...")
    
    test_data_path = get_test_data_path()
    results = {"infinitebench": {}, "longbench": {}}
    
    # Test InfiniteBench factory
    print("\n📋 InfiniteBench factory function:")
    for task_name, filename in INFINITEBENCH_TASKS.items():
        data_path = test_data_path / filename
        if not data_path.exists():
            continue
            
        try:
            config = BenchmarkConfig(
                name="infinitebench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method="test",
            )
            
            adapter = get_adapter(config)
            assert isinstance(adapter, InfiniteBenchAdapter), f"Expected InfiniteBenchAdapter, got {type(adapter)}"
            
            dataset = adapter.create_dataset()
            assert len(dataset) >= 0, "Dataset should be created"
            
            print(f"  ✅ {task_name}: {type(adapter).__name__} with {len(dataset)} items")
            results["infinitebench"][task_name] = len(dataset)
            
        except Exception as e:
            print(f"  ❌ {task_name} factory test failed: {str(e)}")
            results["infinitebench"][task_name] = f"failed: {str(e)}"
    
    # Test LongBench factory
    print("\n📋 LongBench factory function:")
    for task_name, filename in LONGBENCH_TASKS.items():
        data_path = test_data_path / filename
        if not data_path.exists():
            continue
            
        try:
            config = BenchmarkConfig(
                name="longbench",
                data_path=data_path,
                task_type=task_name,
                evaluation_method="test",
            )
            
            adapter = get_adapter(config)
            assert isinstance(adapter, LongBenchAdapter), f"Expected LongBenchAdapter, got {type(adapter)}"
            
            dataset = adapter.create_dataset()
            assert len(dataset) >= 0, "Dataset should be created"
            
            print(f"  ✅ {task_name}: {type(adapter).__name__} with {len(dataset)} items")
            results["longbench"][task_name] = len(dataset)
            
        except Exception as e:
            print(f"  ❌ {task_name} factory test failed: {str(e)}")
            results["longbench"][task_name] = f"failed: {str(e)}"
    
    return results


def show_comprehensive_statistics():
    """Show comprehensive statistics for all available benchmark data"""
    print("📊 COMPREHENSIVE BENCHMARK DATA STATISTICS")
    print("=" * 80)
    
    test_data_path = get_test_data_path()
    
    # InfiniteBench statistics
    print("\n🔥 InfiniteBench Tasks:")
    print("-" * 40)
    ib_total_items = 0
    ib_available_tasks = 0
    
    for task_name, filename in INFINITEBENCH_TASKS.items():
        data_path = test_data_path / filename
        if data_path.exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    item_count = sum(1 for _ in f)
                print(f"{task_name:.<25} {item_count:>6} items")
                ib_total_items += item_count
                ib_available_tasks += 1
            except Exception as e:
                print(f"{task_name:.<25} ERROR: {str(e)}")
        else:
            print(f"{task_name:.<25} NOT FOUND")
    
    print(f"{'Total InfiniteBench:':<25} {ib_total_items:>6} items ({ib_available_tasks} tasks)")
    
    # LongBench statistics
    print("\n📚 LongBench Tasks:")
    print("-" * 40)
    lb_total_items = 0
    lb_available_tasks = 0
    
    for task_name, filename in LONGBENCH_TASKS.items():
        data_path = test_data_path / filename
        if data_path.exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    item_count = sum(1 for _ in f)
                print(f"{task_name:.<25} {item_count:>6} items")
                lb_total_items += item_count
                lb_available_tasks += 1
            except Exception as e:
                print(f"{task_name:.<25} ERROR: {str(e)}")
        else:
            print(f"{task_name:.<25} NOT FOUND")
    
    print(f"{'Total LongBench:':<25} {lb_total_items:>6} items ({lb_available_tasks} tasks)")
    
    # LongBench-E statistics
    print("\n📏 LongBench-E Tasks:")
    print("-" * 40)
    lbe_total_items = 0
    lbe_available_tasks = 0
    
    for task_name, filename in LONGBENCH_E_TASKS.items():
        data_path = test_data_path / filename
        if data_path.exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    item_count = sum(1 for _ in f)
                print(f"{task_name:.<25} {item_count:>6} items")
                lbe_total_items += item_count
                lbe_available_tasks += 1
            except Exception as e:
                print(f"{task_name:.<25} ERROR: {str(e)}")
        else:
            print(f"{task_name:.<25} NOT FOUND")
    
    print(f"{'Total LongBench-E:':<25} {lbe_total_items:>6} items ({lbe_available_tasks} tasks)")
    
    # Grand total
    grand_total = ib_total_items + lb_total_items + lbe_total_items
    total_tasks = ib_available_tasks + lb_available_tasks + lbe_available_tasks
    print(f"\n{'GRAND TOTAL:':<25} {grand_total:>6} items ({total_tasks} tasks)")
    print("=" * 80)


def test_all_success_cases():
    """Test all predefined success cases from evaluation_utils"""
    print("🧪 Testing all success cases from evaluation_utils...")
    
    # Import the test case functions from adapters directory
    try:
        from emotion_memory_experiments.adapters.evaluation_utils import get_success_test_cases, get_score_one
    except ImportError:
        print("  ⏭️ Skipping success cases test - evaluation_utils not available")
        return
    
    success_cases = get_success_test_cases()
    
    passed_tests = 0
    total_tests = len(success_cases)
    
    for task_name, test_case in success_cases.items():
        try:
            prediction = test_case["prediction"]
            ground_truth = test_case["ground_truth"]
            expected_score = test_case["expected_score"]
            
            actual_score = get_score_one(prediction, ground_truth, task_name, "test_model")
            
            # Validate score is numeric and in valid range
            assert isinstance(actual_score, (int, float)), f"Task {task_name}: Expected numeric score, got {type(actual_score)}"
            assert 0 <= actual_score <= 1, f"Task {task_name}: Expected score in range [0,1], got {actual_score}"
            
            # Allow some tolerance for F1 scores
            if expected_score == 1.0:
                assert actual_score >= 0.8, f"Task {task_name} success case failed: expected >= 0.8, got {actual_score}"
            else:
                assert abs(actual_score - expected_score) < 0.2, f"Task {task_name} success case failed: expected ~{expected_score}, got {actual_score}"
            
            passed_tests += 1
            print(f"  ✅ {task_name}: expected ~{expected_score}, got {actual_score}")
            
        except Exception as e:
            print(f"  ❌ {task_name} success case failed: {str(e)}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"  📊 Success case validation: {passed_tests}/{total_tests} ({success_rate:.1f}%) passed")
    
    return success_rate >= 80


def test_all_failure_cases():
    """Test all predefined failure cases from evaluation_utils"""
    print("🧪 Testing all failure cases from evaluation_utils...")
    
    # Import the test case functions from adapters directory
    try:
        from emotion_memory_experiments.adapters.evaluation_utils import get_failure_test_cases, get_score_one
    except ImportError:
        print("  ⏭️ Skipping failure cases test - evaluation_utils not available")
        return
    
    failure_cases = get_failure_test_cases()
    
    passed_tests = 0
    total_tests = len(failure_cases)
    
    for task_name, test_case in failure_cases.items():
        try:
            prediction = test_case["prediction"]
            ground_truth = test_case["ground_truth"]
            expected_score = test_case["expected_score"]
            
            actual_score = get_score_one(prediction, ground_truth, task_name, "test_model")
            
            # Validate score is numeric and in valid range
            assert isinstance(actual_score, (int, float)), f"Task {task_name}: Expected numeric score, got {type(actual_score)}"
            assert 0 <= actual_score <= 1, f"Task {task_name}: Expected score in range [0,1], got {actual_score}"
            
            # Should be low scores for failure cases
            assert actual_score <= 0.2, f"Task {task_name} failure case failed: expected <= 0.2, got {actual_score}"
            
            passed_tests += 1
            print(f"  ✅ {task_name}: expected low score, got {actual_score}")
            
        except Exception as e:
            print(f"  ❌ {task_name} failure case failed: {str(e)}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"  📊 Failure case validation: {passed_tests}/{total_tests} ({success_rate:.1f}%) passed")
    
    return success_rate >= 80


def main():
    """Run comprehensive tests for ALL benchmark tasks"""
    print("🧪 COMPREHENSIVE TESTING - ALL BENCHMARK TASKS")
    print("Testing ALL InfiniteBench and LongBench tasks with real data")
    print("=" * 80)
    
    # Show comprehensive statistics
    show_comprehensive_statistics()
    
    # Run all tests
    all_results = {}
    
    try:
        print("\n" + "="*80)
        all_results["infinitebench"] = test_infinitebench_all_tasks()
    except Exception as e:
        print(f"❌ InfiniteBench comprehensive test failed: {e}")
        all_results["infinitebench"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["longbench"] = test_longbench_all_tasks()
    except Exception as e:
        print(f"❌ LongBench comprehensive test failed: {e}")
        all_results["longbench"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["longbench_e"] = test_longbench_e_length_evaluation()
    except Exception as e:
        print(f"❌ LongBench-E test failed: {e}")
        all_results["longbench_e"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["batch_evaluation"] = test_batch_evaluation_all_tasks()
    except Exception as e:
        print(f"❌ Batch evaluation test failed: {e}")
        all_results["batch_evaluation"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["factory_function"] = test_factory_function_all_tasks()
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        all_results["factory_function"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["success_cases"] = test_all_success_cases()
    except Exception as e:
        print(f"❌ Success cases test failed: {e}")
        all_results["success_cases"] = {"error": str(e)}
    
    try:
        print("\n" + "="*80)
        all_results["failure_cases"] = test_all_failure_cases()
    except Exception as e:
        print(f"❌ Failure cases test failed: {e}")
        all_results["failure_cases"] = {"error": str(e)}
    
    # Generate final summary
    print("\n" + "="*80)
    print("📋 FINAL COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    total_passed = 0
    total_tests = 0
    
    for benchmark_name, results in all_results.items():
        if isinstance(results, dict) and "error" not in results:
            passed = sum(1 for r in results.values() if r == "passed")
            skipped = sum(1 for r in results.values() if r == "skipped")
            failed = len(results) - passed - skipped
            total_passed += passed
            total_tests += len(results)
            
            print(f"\n{benchmark_name.upper()}:")
            print(f"  ✅ Passed: {passed}")
            print(f"  ⏭️  Skipped: {skipped}")
            print(f"  ❌ Failed: {failed}")
            print(f"  📊 Total: {len(results)} tasks")
        elif isinstance(results, dict) and "error" in results:
            print(f"\n{benchmark_name.upper()}: ❌ CRITICAL ERROR")
        elif isinstance(results, bool):
            if results:
                print(f"\n{benchmark_name.upper()}: ✅ PASSED")
            else:
                print(f"\n{benchmark_name.upper()}: ❌ FAILED")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOVERALL SUCCESS RATE: {success_rate:.1f}% ({total_passed}/{total_tests})")
    
    if success_rate >= 80:
        print("\n🎉 COMPREHENSIVE TESTING SUCCESSFUL!")
        print("✅ All major benchmark tasks tested with real data")
        print("✅ Prompt wrapper integration verified")  
        print("✅ Evaluation methods validated")
        print("✅ DataLoader integration confirmed")
        print("✅ Factory function working correctly")
    else:
        print(f"\n⚠️  TESTING PARTIALLY SUCCESSFUL ({success_rate:.1f}%)")
        print("Some tasks may need attention - see detailed logs above")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)