#!/usr/bin/env python3
"""
Comprehensive Multi-Model Response Time Experiment

This script conducts a comprehensive experiment across multiple Qwen2.5 model sizes
to measure response times for different output lengths and establish optimal timeout values.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import signal

@dataclass
class ModelConfig:
    """Configuration for a model to test"""
    name: str
    size: str  # e.g., "0.5B", "1.5B", etc.
    path: str
    min_vram_gb: float
    expected_startup_time: int  # seconds
    port: int = 8000

@dataclass
class TestConfig:
    """Configuration for response time tests"""
    output_lengths: List[int]
    runs_per_test: int
    timeout_per_test: int
    warmup_requests: int

@dataclass
class ExperimentResult:
    """Result from a single test run"""
    model_name: str
    model_size: str
    max_tokens: int
    run_number: int
    success: bool
    response_time: float
    actual_tokens: int
    tokens_per_second: float
    error: Optional[str] = None

class ServerManager:
    """Manages OpenAI server lifecycle for different models"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.current_process = None
        self.current_model = None
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available Qwen2.5 models"""
        
        # Define model configurations
        # Note: Paths should be adjusted based on your actual model locations
        models = [
            ModelConfig(
                name="Qwen2.5-0.5B-Instruct",
                size="0.5B", 
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
                min_vram_gb=2.0,
                expected_startup_time=60  # Increased from 30s
            ),
            ModelConfig(
                name="Qwen2.5-1.5B-Instruct", 
                size="1.5B",
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct",
                min_vram_gb=4.0,
                expected_startup_time=75  # Increased from 45s
            ),
            ModelConfig(
                name="Qwen2.5-3B-Instruct",
                size="3B", 
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct",
                min_vram_gb=8.0,
                expected_startup_time=90  # Increased from 60s
            ),
            ModelConfig(
                name="Qwen2.5-7B-Instruct",
                size="7B",
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-7B-Instruct", 
                min_vram_gb=16.0,
                expected_startup_time=150  # Longer for tensor parallel
            ),
            ModelConfig(
                name="Qwen2.5-14B-Instruct",
                size="14B",
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-14B-Instruct",
                min_vram_gb=32.0,
                expected_startup_time=200  # Longer for tensor parallel
            ),
            ModelConfig(
                name="Qwen2.5-32B-Instruct", 
                size="32B",
                path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-32B-Instruct",
                min_vram_gb=64.0,
                expected_startup_time=300  # Much longer for tensor parallel
            )
        ]
        
        # Filter to only models that exist
        available_models = []
        for model in models:
            if Path(model.path).exists():
                available_models.append(model)
                print(f"✅ Found model: {model.name} ({model.size}) at {model.path}")
            else:
                print(f"❌ Model not found: {model.name} at {model.path}")
        
        return available_models
    
    def get_tensor_parallel_size(self, model: ModelConfig) -> int:
        """Determine optimal tensor parallel size based on model size"""
        size_to_tp = {
            "0.5B": 1,  # Single GPU
            "1.5B": 1,  # Single GPU  
            "3B": 1,    # Single GPU
            "7B": 2,    # 2 GPUs for 7B models
            "14B": 4,   # 4 GPUs for 14B models
            "32B": 8,   # 8 GPUs for 32B models (if available)
        }
        
        tp_size = size_to_tp.get(model.size, 1)
        print(f"   Using tensor-parallel-size={tp_size} for {model.size} model")
        return tp_size
    
    def stop_current_server(self):
        """Stop the currently running server"""
        
        if self.current_process:
            print("🛑 Stopping current server...")
            try:
                # Get process group ID for proper cleanup
                pgid = os.getpgid(self.current_process.pid)
                print(f"   Process PID: {self.current_process.pid}, Group: {pgid}")
                
                # First try graceful shutdown
                print("   Attempting graceful shutdown...")
                self.current_process.terminate()
                
                try:
                    self.current_process.wait(timeout=10)
                    print("   ✅ Graceful shutdown successful")
                except subprocess.TimeoutExpired:
                    # Force kill the entire process group
                    print("   ⚡ Force killing process group...")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        print(f"   Killed process group {pgid}")
                    except ProcessLookupError:
                        print(f"   Process group {pgid} already dead")
                    except Exception as e:
                        print(f"   Warning: Could not kill process group: {e}")
                        # Fallback to killing just the main process
                        self.current_process.kill()
                    
                    # Wait for cleanup
                    try:
                        self.current_process.wait(timeout=5)
                    except:
                        pass
                
            except ProcessLookupError:
                print("   Process already dead")
            except Exception as e:
                print(f"   Error during shutdown: {e}")
            
            self.current_process = None
            self.current_model = None
            print("✅ Server stopped")
        
        # Multi-layered orphan cleanup strategy
        self._cleanup_orphaned_processes()
    
    def _cleanup_orphaned_processes(self):
        """Multi-layered approach to clean up orphaned OpenAI server processes"""
        
        print("   🧹 Multi-layered orphan cleanup...")
        
        # Layer 1: Use manage_servers.py (preferred method)
        orphans_found = self._cleanup_with_manage_servers()
        
        # Layer 2: Direct process cleanup if manage_servers.py fails
        if orphans_found is None:  # manage_servers.py failed
            print("   🔧 Fallback to direct process cleanup...")
            self._cleanup_with_direct_kill()
        
        # Layer 3: Port-based cleanup (nuclear option)
        if self._check_port_occupied(8000):
            print("   ⚡ Port still occupied, using nuclear cleanup...")
            self._cleanup_port_processes(8000)
    
    def _cleanup_with_manage_servers(self) -> int:
        """Try to clean up using manage_servers.py. Returns number of processes found, or None if failed."""
        try:
            print("   📋 Using manage_servers.py...")
            result = subprocess.run([
                "python", "openai_server/manage_servers.py"
            ], input="a\ny\n", text=True, capture_output=True, cwd=self.base_path, timeout=30)
            
            if "killed" in result.stdout.lower() or "Killed" in result.stdout:
                # Parse number of killed servers
                lines = result.stdout.split('\n')
                killed_line = [line for line in lines if 'Killed' in line and ('server' in line.lower() or 'process' in line.lower())]
                if killed_line:
                    print(f"   ✅ manage_servers.py: {killed_line[0].strip()}")
                else:
                    print("   ✅ manage_servers.py: Cleaned up orphaned processes")
                return 0  # Success
            elif "No OpenAI servers found" in result.stdout:
                print("   ✅ No orphaned processes found")
                return 0  # Success, no orphans
            else:
                # Extract server count from output
                lines = result.stdout.split('\n')
                server_lines = [line for line in lines if 'PID:' in line and 'Port:' in line]
                if server_lines:
                    print(f"   ⚠️  manage_servers.py found {len(server_lines)} servers but didn't kill them")
                    return len(server_lines)
                return 0
                
        except subprocess.TimeoutExpired:
            print("   ⚠️  manage_servers.py timeout")
            return None  # Failed
        except Exception as e:
            print(f"   ⚠️  manage_servers.py error: {e}")
            return None  # Failed
    
    def _cleanup_with_direct_kill(self):
        """Direct process cleanup using ps and kill commands"""
        try:
            # Find OpenAI server processes
            result = subprocess.run([
                "ps", "aux"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                # Look for openai_server processes
                server_processes = []
                for line in lines:
                    if ('openai_server' in line or 'python -m openai_server' in line) and 'grep' not in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                server_processes.append(pid)
                            except ValueError:
                                continue
                
                if server_processes:
                    print(f"   🔫 Killing {len(server_processes)} orphaned processes directly...")
                    for pid in server_processes:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            print(f"   ✅ Killed PID {pid}")
                        except ProcessLookupError:
                            print(f"   ℹ️  PID {pid} already dead")
                        except Exception as e:
                            print(f"   ⚠️  Could not kill PID {pid}: {e}")
                else:
                    print("   ✅ No OpenAI server processes found")
            else:
                print(f"   ⚠️  ps command failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ⚠️  Direct cleanup failed: {e}")
    
    def _check_port_occupied(self, port: int) -> bool:
        """Check if a port is still occupied"""
        try:
            result = subprocess.run([
                "lsof", "-i", f":{port}"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines:
                    print(f"   ⚠️  Port {port} still occupied by {len(lines)} processes")
                    return True
            return False
            
        except Exception as e:
            print(f"   ⚠️  Could not check port {port}: {e}")
            return False
    
    def _cleanup_port_processes(self, port: int):
        """Kill all processes using a specific port (nuclear option)"""
        try:
            result = subprocess.run([
                "lsof", "-ti", f":{port}"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"   ⚡ Nuclear cleanup: killing {len(pids)} processes on port {port}")
                
                for pid_str in pids:
                    try:
                        pid = int(pid_str.strip())
                        os.kill(pid, signal.SIGKILL)
                        print(f"   💀 Nuclear killed PID {pid}")
                    except ValueError:
                        continue
                    except ProcessLookupError:
                        print(f"   ℹ️  PID {pid} already dead")
                    except Exception as e:
                        print(f"   ⚠️  Could not nuclear kill PID {pid}: {e}")
                        
                # Verify port is now free
                time.sleep(1)
                if not self._check_port_occupied(port):
                    print(f"   ✅ Port {port} successfully freed")
                else:
                    print(f"   ❌ Port {port} still occupied after nuclear cleanup")
            else:
                print(f"   ✅ No processes found on port {port}")
                
        except Exception as e:
            print(f"   ⚠️  Nuclear cleanup failed: {e}")
    
    def start_server(self, model: ModelConfig) -> bool:
        """Start server with specified model"""
        
        print(f"🚀 Starting server with {model.name} ({model.size})...")
        
        # Determine tensor parallel size based on model size
        tp_size = self.get_tensor_parallel_size(model)
        
        # Build command with conda environment activation and extended timeout
        cmd = f"""
        source /usr/local/anaconda3/etc/profile.d/conda.sh && 
        conda activate llm_fresh && 
        python -m openai_server \
            --model {model.path} \
            --model_name {model.name}-anger \
            --emotion anger \
            --port {model.port} \
            --gpu_memory_utilization 0.85 \
            --tensor_parallel_size {tp_size} \
            --request_timeout 600
        """
        
        # Start process
        try:
            self.current_process = subprocess.Popen(
                cmd,
                shell=True,  # Required for conda activation
                executable='/bin/bash',  # Use bash instead of sh
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_path,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            # Wait for server to start
            print(f"⏳ Waiting for server startup (up to {model.expected_startup_time}s)...")
            
            for i in range(model.expected_startup_time):
                try:
                    response = requests.get(f"http://localhost:{model.port}/health", timeout=5)
                    if response.status_code == 200:
                        self.current_model = model
                        print(f"✅ Server ready after {i+1}s")
                        return True
                except Exception as e:
                    if i == 0:  # Show first error for debugging
                        print(f"   Debug: Health check error: {e}")
                    pass
                
                if i % 10 == 0 and i > 0:
                    print(f"   Still waiting... ({i}s elapsed)")
                    # Check if process is still running
                    if self.current_process.poll() is not None:
                        print(f"   ⚠️  Process exited with code: {self.current_process.returncode}")
                        try:
                            stdout, stderr = self.current_process.communicate(timeout=5)
                            print(f"   STDOUT: {stdout.decode()[:200]}...")
                            print(f"   STDERR: {stderr.decode()[:200]}...")
                        except subprocess.TimeoutExpired:
                            print(f"   Could not read process output (timeout)")
                        except Exception as e:
                            print(f"   Could not read process output: {e}")
                        break
                
                time.sleep(1)
            
            print(f"❌ Server failed to start within {model.expected_startup_time}s")
            self.stop_current_server()
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            # Ensure cleanup happens even if there's an exception
            try:
                self.stop_current_server()
            except:
                pass
            return False
    
    def health_check(self) -> bool:
        """Check if current server is healthy"""
        
        if not self.current_model:
            return False
        
        try:
            response = requests.get(f"http://localhost:{self.current_model.port}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def restart_with_model(self, model: ModelConfig) -> bool:
        """Stop current server and start with new model"""
        
        if self.current_model and self.current_model.name == model.name:
            if self.health_check():
                print(f"✅ Server already running with {model.name}")
                return True
        
        self.stop_current_server()
        time.sleep(5)  # Wait for cleanup
        return self.start_server(model)


class ResponseTimeMeasurer:
    """Measures response times for different output lengths with hard validation"""
    
    def __init__(self, server_url: str = "http://localhost:8000", min_completion_rate: float = 0.85, max_retries: int = 3):
        self.server_url = server_url
        self.min_completion_rate = min_completion_rate  # Minimum 85% of target tokens
        self.max_retries = max_retries
    
    def generate_prompt_for_length(self, target_tokens: int, retry_attempt: int = 0) -> str:
        """Generate prompt designed to force exactly target_tokens output"""
        
        base_prompt = f"""CRITICAL: Your previous response was too short. You MUST generate exactly {target_tokens} words this time. This is mandatory for testing accuracy.

Continue writing a detailed technical analysis about artificial intelligence systems. You need to produce EXACTLY {target_tokens} words. Here are additional topics to ensure you reach the target:

1. Deep learning architectures (CNNs, RNNs, Transformers, GANs)
2. Optimization algorithms (SGD, Adam, RMSprop, learning rate scheduling)
3. Regularization techniques (dropout, batch normalization, weight decay)
4. Computer vision applications (object detection, segmentation, face recognition)
5. Natural language processing (tokenization, embeddings, attention mechanisms)
6. Reinforcement learning (Q-learning, policy gradients, actor-critic methods)
7. MLOps and deployment strategies (containerization, monitoring, scaling)
8. Data preprocessing and feature engineering techniques
9. Model evaluation metrics and cross-validation strategies
10. Hardware considerations (GPUs, TPUs, distributed training)
11. Specific case studies and real-world implementations
12. Mathematical foundations and statistical concepts
13. Code examples and implementation details
14. Performance benchmarking and optimization techniques
15. Ethical AI considerations and bias mitigation

IMPORTANT: Keep writing until you reach EXACTLY {target_tokens} words. Add more technical details, examples, mathematical formulations, code snippets, or any relevant content to reach the exact target. Do not stop until you have {target_tokens} words. Count carefully as you write."""

        if retry_attempt == 0:
            return base_prompt
        else:            
            return base_prompt + f"In the previous response, you did not generate exactly {target_tokens} words. You MUST generate exactly {target_tokens} words this time. This is mandatory for testing accuracy. Do not stop until you have {target_tokens} words."
    
    def validate_output_length(self, content: str, target_tokens: int) -> tuple[bool, int, float]:
        """Validate if output meets minimum length requirements"""
        actual_tokens = len(content.split())
        completion_rate = actual_tokens / target_tokens if target_tokens > 0 else 0
        is_valid = completion_rate >= self.min_completion_rate
        
        return is_valid, actual_tokens, completion_rate
    
    def measure_single_request(self, model_name: str, max_tokens: int, port: int = 8000) -> ExperimentResult:
        """Measure response time for a single request with hard validation and retry logic"""
        
        total_start_time = time.time()
        cumulative_response_time = 0.0
        best_result = None
        attempts_log = []
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            print(f"    Attempt {attempt + 1}/{self.max_retries + 1}...", end=" ")
            
            prompt = self.generate_prompt_for_length(max_tokens, attempt)
            attempt_start_time = time.time()
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        'model': model_name,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': max_tokens,
                        'temperature': 0.0,
                        'top_p': 1.0,
                        'frequency_penalty': 0.0,
                        'presence_penalty': 0.0,
                        'stop': None  # Don't stop early
                    },
                    timeout=600  # 10 minute timeout for measurement
                )
                
                attempt_end_time = time.time()
                attempt_response_time = attempt_end_time - attempt_start_time
                cumulative_response_time += attempt_response_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Validate output length
                    is_valid, actual_tokens, completion_rate = self.validate_output_length(content, max_tokens)
                    tokens_per_second = actual_tokens / attempt_response_time if attempt_response_time > 0 else 0
                    
                    attempts_log.append({
                        'attempt': attempt + 1,
                        'tokens': actual_tokens,
                        'completion_rate': completion_rate,
                        'time': attempt_response_time,
                        'valid': is_valid
                    })
                    
                    print(f"{actual_tokens}/{max_tokens} tokens ({completion_rate:.1%}) - {'✅ VALID' if is_valid else '❌ TOO SHORT'}")
                    
                    # Create result for this attempt
                    current_result = ExperimentResult(
                        model_name=model_name,
                        model_size="",  # Will be filled by caller
                        max_tokens=max_tokens,
                        run_number=0,  # Will be filled by caller
                        success=True,
                        response_time=attempt_response_time,
                        actual_tokens=actual_tokens,
                        tokens_per_second=tokens_per_second
                    )
                    
                    # Keep track of best result so far (highest token count)
                    if best_result is None or actual_tokens > best_result.actual_tokens:
                        best_result = current_result
                    
                    # If validation passed, return immediately
                    if is_valid:
                        best_result.error = f"Succeeded on attempt {attempt + 1} (validation passed)"
                        return best_result
                    
                    # If this is the last attempt and we still don't have valid output
                    if attempt == self.max_retries:
                        break
                    
                    # Brief pause before retry
                    time.sleep(1)
                    
                else:
                    attempts_log.append({
                        'attempt': attempt + 1,
                        'tokens': 0,
                        'completion_rate': 0,
                        'time': attempt_response_time,
                        'valid': False,
                        'error': f"HTTP {response.status_code}"
                    })
                    print(f"❌ HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                attempt_response_time = 600.0
                cumulative_response_time += attempt_response_time
                attempts_log.append({
                    'attempt': attempt + 1,
                    'tokens': 0,
                    'completion_rate': 0,
                    'time': attempt_response_time,
                    'valid': False,
                    'error': 'Timeout'
                })
                print("❌ TIMEOUT (600s)")
                
                return ExperimentResult(
                    model_name=model_name,
                    model_size="",
                    max_tokens=max_tokens,
                    run_number=0,
                    success=False,
                    response_time=600.0,
                    actual_tokens=0,
                    tokens_per_second=0,
                    error="Request timeout (600s) - would definitely exceed any reasonable server timeout"
                )
                
            except Exception as e:
                attempt_response_time = time.time() - attempt_start_time
                cumulative_response_time += attempt_response_time
                attempts_log.append({
                    'attempt': attempt + 1,
                    'tokens': 0,
                    'completion_rate': 0,
                    'time': attempt_response_time,
                    'valid': False,
                    'error': str(e)
                })
                print(f"❌ ERROR: {str(e)[:50]}...")
        
        # If we get here, all attempts failed validation
        if best_result:
            # Use the best attempt (highest token count) but mark as failed validation
            best_result.success = False
            best_result.error = f"Failed validation after {self.max_retries + 1} attempts. Best: {best_result.actual_tokens}/{max_tokens} tokens ({best_result.actual_tokens/max_tokens:.1%})"
            return best_result
        else:
            # No successful HTTP responses at all
            return ExperimentResult(
                model_name=model_name,
                model_size="",
                max_tokens=max_tokens,
                run_number=0,
                success=False,
                response_time=cumulative_response_time,
                actual_tokens=0,
                tokens_per_second=0,
                error=f"All {self.max_retries + 1} attempts failed"
            )


class ComprehensiveExperiment:
    """Main experiment controller"""
    
    def __init__(self, base_path: str, output_dir: str = "experiment_results"):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.server_manager = ServerManager(base_path)
        self.measurer = ResponseTimeMeasurer()
        self.results = []
        
        # Experiment timestamp
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = self.output_dir / f"progress_{self.experiment_id}.json"
    
    def save_progress(self):
        """Save current progress to file"""
        progress_data = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'results_count': len(self.results),
            'results': [asdict(r) for r in self.results]
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def run_experiment(self, test_config: TestConfig, models_to_test: Optional[List[str]] = None):
        """Run the comprehensive experiment"""
        
        print(f"🚀 COMPREHENSIVE MULTI-MODEL RESPONSE TIME EXPERIMENT")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Started: {datetime.now()}")
        print("=" * 80)
        
        # Get available models
        available_models = self.server_manager.get_available_models()
        
        if models_to_test:
            available_models = [m for m in available_models if m.size in models_to_test]
        
        if not available_models:
            print("❌ No models available for testing")
            return
        
        total_tests = len(available_models) * len(test_config.output_lengths) * test_config.runs_per_test
        test_count = 0
        
        print(f"📊 Experiment Plan:")
        print(f"  Models: {[m.size for m in available_models]}")
        print(f"  Output lengths: {test_config.output_lengths}")
        print(f"  Runs per test: {test_config.runs_per_test}")
        print(f"  Total tests: {total_tests}")
        print(f"  Estimated time: {total_tests * 30 / 60:.1f} minutes")
        print()
        
        try:
            for model in available_models:
                print(f"🔬 TESTING MODEL: {model.name} ({model.size})")
                print("-" * 60)
                
                # Start server with this model
                if not self.server_manager.restart_with_model(model):
                    print(f"❌ Failed to start server for {model.name}, skipping...")
                    continue
                
                # Warmup requests
                print(f"🔥 Warmup requests...")
                for i in range(test_config.warmup_requests):
                    try:
                        self.measurer.measure_single_request(
                            f"{model.name}-anger", 100, model.port
                        )
                    except:
                        pass
                
                # Test each output length
                for output_length in test_config.output_lengths:
                    print(f"\\n📏 Testing {output_length} tokens:")
                    
                    for run in range(test_config.runs_per_test):
                        test_count += 1
                        progress = test_count / total_tests * 100
                        
                        print(f"  Run {run+1}/{test_config.runs_per_test} [{progress:.1f}%]...", end=" ")
                        
                        result = self.measurer.measure_single_request(
                            f"{model.name}-anger", output_length, model.port
                        )
                        
                        # Fill in missing fields
                        result.model_size = model.size
                        result.run_number = run + 1
                        
                        self.results.append(result)
                        
                        if result.success:
                            print(f"✅ {result.response_time:.2f}s ({result.tokens_per_second:.1f} tok/s)")
                        else:
                            print(f"❌ {result.error}")
                        
                        # Save progress periodically
                        if test_count % 5 == 0:
                            self.save_progress()
                        
                        # Small delay between tests
                        time.sleep(2)
                
                print(f"✅ Completed testing {model.name}")
        
        except KeyboardInterrupt:
            print("\\n⚠️  Experiment interrupted by user")
        except Exception as e:
            print(f"\\n❌ Experiment failed: {e}")
        finally:
            # Stop server and save final results
            self.server_manager.stop_current_server()
            self.save_progress()
            self.generate_analysis_report()
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print(f"\\n📊 GENERATING ANALYSIS REPORT...")
        
        # Group results by model and output length
        model_stats = {}
        successful_results = [r for r in self.results if r.success]
        
        for result in successful_results:
            model_key = f"{result.model_size}"
            if model_key not in model_stats:
                model_stats[model_key] = {}
            
            length_key = result.max_tokens
            if length_key not in model_stats[model_key]:
                model_stats[model_key][length_key] = []
            
            model_stats[model_key][length_key].append(result)
        
        # Generate report
        report = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(self.results) if self.results else 0
            },
            'model_performance': {},
            'timeout_recommendations': {},
            'scaling_analysis': {}
        }
        
        print(f"\\n📈 PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Model':<8} {'Tokens':<8} {'Avg Time':<10} {'Tokens/Sec':<12} {'P95 Time':<10} {'Recommended Timeout'}")
        print("-" * 80)
        
        for model_size in sorted(model_stats.keys()):
            model_data = {}
            timeout_data = {}
            
            for output_length in sorted(model_stats[model_size].keys()):
                results = model_stats[model_size][output_length]
                times = [r.response_time for r in results]
                throughputs = [r.tokens_per_second for r in results]
                
                avg_time = statistics.mean(times)
                p95_time = statistics.quantiles(times, n=20)[18] if len(times) >= 5 else max(times)
                avg_throughput = statistics.mean(throughputs)
                recommended_timeout = p95_time * 2  # 2x safety margin
                
                print(f"{model_size:<8} {output_length:<8} {avg_time:<10.2f} {avg_throughput:<12.1f} {p95_time:<10.2f} {recommended_timeout:<10.0f}s")
                
                model_data[output_length] = {
                    'avg_response_time': avg_time,
                    'p95_response_time': p95_time,
                    'avg_tokens_per_second': avg_throughput,
                    'num_runs': len(results)
                }
                
                timeout_data[output_length] = recommended_timeout
            
            report['model_performance'][model_size] = model_data
            report['timeout_recommendations'][model_size] = timeout_data
        
        # Save report
        report_file = self.output_dir / f"comprehensive_analysis_{self.experiment_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n💾 Analysis report saved: {report_file}")
        print(f"💾 Raw results saved: {self.progress_file}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive multi-model response time experiment')
    parser.add_argument('--base-path', default='/data/home/jjl7137/LLM_EmoBehav_game_theory',
                       help='Base path to project directory')
    parser.add_argument('--models', nargs='+', 
                       choices=['0.5B', '1.5B', '3B', '7B', '14B', '32B', '72B'],
                       help='Model sizes to test (default: all available)')
    parser.add_argument('--output-lengths', nargs='+', type=int,
                       default=[50, 100, 200, 500, 1000, 2000],
                       help='Output lengths to test')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per test')
    parser.add_argument('--warmup', type=int, default=2,
                       help='Number of warmup requests per model')
    parser.add_argument('--output-dir', default='experiment_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create test configuration
    test_config = TestConfig(
        output_lengths=args.output_lengths,
        runs_per_test=args.runs,
        timeout_per_test=300,
        warmup_requests=args.warmup
    )
    
    # Run experiment
    experiment = ComprehensiveExperiment(args.base_path, args.output_dir)
    experiment.run_experiment(test_config, args.models)


if __name__ == "__main__":
    main()