"""
Simple Emotion Memory Experiment

Following the same pattern as emotion_game_experiment.py but for memory benchmarks.
Just like game scenarios, memory tasks are input-output pairs with emotion manipulation.
"""
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
from functools import partial
from openai import AzureOpenAI, OpenAI
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
import yaml

from api_configs import AZURE_OPENAI_CONFIG, OAI_CONFIG
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
from neuro_manipulation.repe.pipelines import get_pipeline


class MemoryDataset:
    """Simple dataset that loads memory benchmark tasks as input-output pairs"""
    
    def __init__(self, benchmark_name, data_path, sample_num=None):
        self.benchmark_name = benchmark_name
        self.data = self._load_data(data_path)
        
        if sample_num is not None:
            self.data = self.data[:sample_num]
    
    def _load_data(self, data_path):
        """Load benchmark data - each item should have 'input' and 'expected_output'"""
        with open(data_path, 'r') as f:
            if data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        # Convert to standard format if needed
        formatted_data = []
        for item in data:
            # Adapt different benchmark formats to standard input/output
            if self.benchmark_name == 'infinitebench':
                formatted_data.append({
                    'input': item['input'],  # The long context + question
                    'expected_output': item['answer'],
                    'task_type': item.get('task_type', 'unknown'),
                    'id': item.get('id', len(formatted_data))
                })
            elif self.benchmark_name == 'locomo':
                formatted_data.append({
                    'input': item['question'],  # Question about conversation
                    'expected_output': item['answer'],
                    'context': item.get('conversation', ''),  # Long conversation
                    'task_type': 'conversational_qa',
                    'id': item.get('sample_id', len(formatted_data))
                })
            else:
                # Generic format
                formatted_data.append(item)
        
        return formatted_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class EmotionMemoryExperiment:
    """
    Simple experiment class following the same pattern as EmotionGameExperiment
    but for memory benchmarks instead of game theory scenarios.
    """
    
    def __init__(self, repe_eng_config, exp_config, memory_config, batch_size, sample_num=None, repeat=1):
        # Setup logging (same as game experiment)
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/emotion_memory_experiment_{timestamp}.log"
        
        if not self.logger.handlers:
            Path("logs").mkdir(parents=True, exist_ok=True)
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
        
        self.logger.info(f"Initializing memory experiment with model: {repe_eng_config['model_name_or_path']}")
        
        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.memory_config = memory_config  # Contains benchmark name, data path, etc.
        self.generation_config = exp_config["experiment"]["llm"]["generation_config"]
        
        self.repeat = repeat
        self.sample_num = sample_num
        self.batch_size = batch_size
        
        # Setup model and emotion readers (same as game experiment)
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=False
        )
        num_hidden_layers = ModelLayerDetector.num_layers(self.model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        
        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config, self.model, self.tokenizer, self.hidden_layers
        )
        del self.model
        
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=True
        )
        
        self.intensities = self.exp_config["experiment"].get(
            "intensity", self.repe_eng_config["coeffs"]
        )
        
        self.rep_control_pipeline = get_pipeline(
            "rep-control-vllm",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.hidden_layers[len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3],
            block_name=self.repe_eng_config["block_name"],
            control_method=self.repe_eng_config["control_method"],
        )
        
        self.cur_emotion = None
        self.cur_coeff = None
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.exp_config['experiment']['output']['base_dir']}/memory_{self.memory_config['benchmark_name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def build_dataloader(self):
        """Build dataloader for memory benchmark"""
        dataset = MemoryDataset(
            self.memory_config['benchmark_name'],
            self.memory_config['data_path'],
            sample_num=self.sample_num
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_memory_items
        )
    
    def _collate_memory_items(self, batch):
        """Collate memory items into batches"""
        return {
            'input': [item['input'] for item in batch],
            'expected_output': [item['expected_output'] for item in batch],
            'task_type': [item.get('task_type', 'unknown') for item in batch],
            'id': [item.get('id', i) for i, item in enumerate(batch)]
        }
    
    def run_experiment(self):
        """Run the complete emotion memory experiment"""
        self.logger.info("Starting memory experiment")
        results = []
        
        # Test each emotion
        for emotion in self.exp_config["experiment"]["emotions"]:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion
            
            data_loader = self.build_dataloader()
            
            # Test each intensity
            for coeff in self.intensities:
                self.logger.info(f"Processing coefficient: {coeff}")
                self.cur_coeff = coeff
                results.extend(self._infer_with_activation(rep_reader, data_loader))
        
        # Test neutral condition
        self.cur_emotion = "Neutral"
        self.cur_coeff = 0
        self.logger.info("Processing Neutral condition")
        data_loader = self.build_dataloader()
        results.extend(self._infer_with_activation(rep_reader, data_loader))
        
        return self._save_results(results)
    
    def _infer_with_activation(self, rep_reader, data_loader):
        """Run inference with emotion activation"""
        self.logger.info(f"Setting up activations for emotion {self.cur_emotion}, coefficient {self.cur_coeff}")
        
        device = torch.device("cpu")  # For vLLM
        
        activations = {
            layer: torch.tensor(
                self.cur_coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
            ).to(device).half()
            for layer in self.hidden_layers
        }
        
        return self._forward_dataloader(data_loader, activations)
    
    def _forward_dataloader(self, data_loader, activations):
        """Process all batches with current emotion activation"""
        results = []
        
        for batch_idx, batch in enumerate(data_loader):
            # Repeat prompts for multiple runs
            repeat_batch = {
                key: [item for item in value for _ in range(self.repeat)]
                for key, value in batch.items()
            }
            
            # Generate responses with emotion control
            start_time = time.time()
            control_outputs = self.rep_control_pipeline(
                repeat_batch["input"],
                activations=activations,
                batch_size=self.batch_size * self.repeat,
                temperature=self.generation_config["temperature"],
                max_new_tokens=self.generation_config["max_new_tokens"],
                do_sample=self.generation_config["do_sample"],
                top_p=self.generation_config["top_p"],
            )
            end_time = time.time()
            
            self.logger.info(f"Batch {batch_idx} completed in {end_time - start_time:.2f}s")
            
            # Process results
            batch_results = self._process_batch_results(repeat_batch, control_outputs)
            results.extend(batch_results)
        
        return results
    
    def _process_batch_results(self, batch, control_outputs):
        """Process results from a single batch"""
        results = []
        batch_size = int(len(batch["input"]) / self.repeat)
        
        for i, output in enumerate(control_outputs):
            generated_text = output.outputs[0].text.replace(batch["input"][i], "")
            
            original_idx = i % batch_size
            repeat_num = i // batch_size
            
            results.append({
                "emotion": self.cur_emotion,
                "intensity": self.cur_coeff,
                "task_id": batch["id"][original_idx],
                "task_type": batch["task_type"][original_idx],
                "input": batch["input"][i],
                "output": generated_text,
                "expected_output": batch["expected_output"][original_idx],
                "repeat_num": repeat_num,
            })
        
        return results
    
    def _save_results(self, results):
        """Save experimental results"""
        self.logger.info("Saving experiment results")
        
        # Save JSON
        json_filename = f"{self.output_dir}/memory_results.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(results)
        csv_filename = f"{self.output_dir}/memory_results.csv"
        df.to_csv(csv_filename, index=False)
        
        self.logger.info(f"Results saved to {json_filename} and {csv_filename}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Example configuration (similar to game experiment configs)
    repe_eng_config = {
        "model_name_or_path": "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        "emotions": ["anger", "happiness", "sadness"],
        "coeffs": [0.5, 1.0, 1.5],
        "block_name": "transformer.h.{}.mlp",
        "control_method": "linear_comb"
    }
    
    exp_config = {
        "experiment": {
            "name": "memory_emotion_test",
            "emotions": ["anger", "happiness"],
            "intensity": [1.0],
            "llm": {
                "generation_config": {
                    "temperature": 0.7,
                    "max_new_tokens": 100,
                    "do_sample": True,
                    "top_p": 0.9
                }
            },
            "output": {
                "base_dir": "results"
            }
        }
    }
    
    memory_config = {
        "benchmark_name": "infinitebench",
        "data_path": "/data/home/jjl7137/memory_benchmarks/InfiniteBench/data/passkey.jsonl"
    }
    
    experiment = EmotionMemoryExperiment(
        repe_eng_config=repe_eng_config,
        exp_config=exp_config,
        memory_config=memory_config,
        batch_size=2,
        sample_num=10,  # Small test
        repeat=1
    )
    
    results_df = experiment.run_experiment()
    print(f"Experiment completed. Results shape: {results_df.shape}")