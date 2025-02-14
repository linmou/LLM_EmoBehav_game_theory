from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from transformers import pipeline
import pandas as pd
from functools import partial
from openai import OpenAI
from pydantic import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
import re
from pathlib import Path

import yaml

from neuro_manipulation.datasets.game_scenario_dataset import GameScenarioDataset
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
from neuro_manipulation.model_utils import setup_model_and_tokenizer, load_emotion_readers
from neuro_manipulation.utils import oai_response
from api_configs import OAI_CONFIG
from statistical_engine import analyze_emotion_and_intensity_effects

class ExtractedResult(BaseModel):
    option_id: int
    rationale: str
    decision: str
        

class EmotionGameExperiment:
    def __init__(self, repe_eng_config, exp_config, game_config, batch_size, sample_num=None,  repeat=1):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing experiment with model: {repe_eng_config['model_name_or_path']}")
        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.generation_config = exp_config['experiment']['llm']['generation_config']
        self.game_config = game_config
        
        self.repeat = repeat
        self.sample_num = sample_num
            
        self.batch_size = batch_size
        
        self.model, self.tokenizer, self.prompt_format, self.user_tag, self.assistant_tag = setup_model_and_tokenizer(repe_eng_config)
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        
        self.repe_eng_config.update({
            'user_tag': self.user_tag,
            'assistant_tag': self.assistant_tag
        })
        
        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config, 
            self.model, 
            self.tokenizer, 
            self.hidden_layers
        )
        
        self.intensities = self.exp_config['experiment'].get('intensity', self.repe_eng_config['coeffs'])
        
        self.rep_control_pipeline = pipeline(
            "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.repe_eng_config['control_layer_id'],
            block_name=self.repe_eng_config['block_name'],
            control_method=self.repe_eng_config['control_method']
        )
        
        self.reaction_prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, 
            response_format=self.game_config['decision_class']
        )
        
        self.cur_emotion = None
        self.cur_coeff = None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        self.output_dir = f"{self.exp_config['experiment']['output']['base_dir']}/{self.exp_config['experiment']['name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(self.output_dir + '/exp_config.yaml', 'w') as f:
            yaml.dump(self.exp_config, f)
         
        self.llm_client = OpenAI(**OAI_CONFIG)

    def run_experiment(self):
        self.logger.info("Starting experiment")
        results = []
        
        for emotion in self.repe_eng_config['emotions']:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion
            
            self.logger.info(f"Creating dataset with sample_num={self.sample_num if self.sample_num is not None else 'all'}")
            emo_dataset = GameScenarioDataset(
                self.game_config,
                partial(self.reaction_prompt_wrapper.__call__, 
                       user_messages=self.exp_config['experiment']['system_message_template'].format(emotion=emotion)),
                sample_num=self.sample_num,
            )
            
            data_loader = DataLoader(emo_dataset, batch_size=self.batch_size, shuffle=False)
            
            for coeff in self.intensities:
                self.logger.info(f"Processing coefficient: {coeff}")
                self.cur_coeff = coeff
                results.extend(self._infer_with_activation(rep_reader, data_loader))
        
        self.cur_emotion = 'Neutral'
        self.cur_coeff = 0
        self.logger.info(f"Processing Null Emotion")
        results.extend(self._infer_with_activation(rep_reader, data_loader))
         
        return self._save_results(results)

    def _infer_with_activation(self, rep_reader, data_loader):
        self.logger.info(f"Setting up activations for coefficient {self.cur_coeff}")
        activations = {
            layer: torch.tensor(self.cur_coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer])
            .to(self.model.device).half()
            for layer in self.hidden_layers
        }
       
        return self._forward_dataloader(data_loader, activations)
    
    def _forward_dataloader(self, data_loader, activations):
        batch_results = []
        pipeline_queue = Queue(maxsize=2)  # Control memory usage
         
        def pipeline_worker():
            for batch in data_loader:
                # Repeat each list in the batch dictionary
                repeat_batch = {
                    key: [item for item in value for _ in range(self.repeat)]
                    for key, value in batch.items()
                }
                control_outputs = self.rep_control_pipeline(
                    repeat_batch['prompt'],
                    activations=activations,
                    batch_size=self.batch_size,
                    max_new_tokens=self.generation_config['max_new_tokens'],
                    do_sample=self.generation_config['do_sample'],
                    top_p=self.generation_config['top_p']
                )
                pipeline_queue.put((batch, control_outputs))
            pipeline_queue.put(None)
        
        # Start pipeline worker thread
        worker = Thread(target=pipeline_worker)
        worker.start()
        
        # Process results while next batch is being generated
        with ThreadPoolExecutor(max_workers=8) as post_proc_executor:
            while True:
                item = pipeline_queue.get()
                if item is None:
                    break
                batch, control_outputs = item
                
                # Submit post-processing to executor
                future = post_proc_executor.submit(
                    self._post_process_batch,
                    batch,
                    control_outputs
                )
                batch_results.extend(future.result())
        
        worker.join()
        return batch_results

    def _post_process_batch(self, batch, control_outputs):
        results = []
        output_data = []
        
        # Prepare data for parallel processing
        batch_size = len(batch['prompt'])
        for i, p in enumerate(control_outputs):
            generated_text = p[0]['generated_text'].replace(batch['prompt'][i % batch_size], "")
            options = [opts[i % batch_size] for opts in batch['options']]
            output_data.append((generated_text, options))
            repeat_idx = i // batch_size
        
        # Process outputs in parallel
        with ThreadPoolExecutor(max_workers=min(len(output_data), 8)) as executor:
            extracted_reses = list(executor.map(self._post_process_single_output, output_data))
        
        # Combine results
        for i, (extracted_res, generated_text) in enumerate(zip(extracted_reses, output_data)):
            results.append({
                'emotion': self.cur_emotion,
                'intensity': self.cur_coeff,
                'scenario': batch['scenario'][i % batch_size],
                'description': batch['description'][i % batch_size],
                # 'options': batch['options'][i % batch_size],
                'input': batch['prompt'][i % batch_size],
                'output': generated_text,
                'rationale': extracted_res.rationale,
                'decision': extracted_res.decision,
                'category': extracted_res.option_id,
                'repeat_num': i // batch_size
            })
        
        return results

    def _post_process_single_output(self, output_data):
        generated_text, options = output_data
        
        try:
            option_id = [j+1 for j, o in enumerate(options) if o.lower() in generated_text.lower()][0]
            # Extract rationale and decision using regex
            rationale = re.search(r'"rational"\s*:\s*"([^"]*)"', generated_text)
            decision = re.search(r'"decision"\s*:\s*"([^"]*)"', generated_text)
            
            if not rationale or not decision:
                raise ValueError("Failed to extract rationale or decision")
                
            rationale = rationale.group(1)
            decision = decision.group(1)
            
            extracted_res = ExtractedResult(
                option_id=option_id,
                rationale=rationale,
                decision=decision
            )
            
            self.logger.info(f"Found option_id {option_id} directly from text")
        
        except Exception as e:
            self.logger.info(f"Extraction failed: {str(e)}. Using LLM to determine option_id")
            extracted_res = self._get_option_id_from_llm(generated_text, options)
            self.logger.info(f"LLM extracted res sucessfully")
            
        return extracted_res

    def _get_option_id_from_llm(self, generated_text, options):
        prompt = f"Given the options {[ 'option ' + str(oid+1) + ' : ' + opt for oid, opt in enumerate(options)]}. "
        prompt += f"Extract the decision, the rationale, and the option id for the following generated text: {generated_text}. "
        prompt += f"If the generated text is not in the options, return -1. Option ranges from 1 to {len(options)}."
        prompt += f"Response in the following format. {{'decision': <decision>, 'rationale': <rationale>, 'option_id': <option_id>}}"
        
        for _ in range(3):
            try:
                res = oai_response(prompt, model="gpt-4o-mini", client=self.llm_client, response_format=ExtractedResult)
                return ExtractedResult(**json.loads(res))
            except Exception as e:
                self.logger.info(f"Error getting post-processed response from LLM: {e}")
                prompt = f"In previous run, there is an error: {e}. Please try again.\n\n{prompt}"
        
        return ExtractedResult(option_id=-1, rationale=generated_text, decision="")

    def _save_results(self, results):
        self.logger.info("Saving experiment results")
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        json_filename = f"{self.output_dir}/exp_results.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {json_filename}")
        
        df = pd.DataFrame(results)
        csv_filename = f"{self.output_dir}/exp_results.csv"
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Results saved to {csv_filename}")
        
        stats_results = self._run_statistical_analysis(csv_filename)
        stats_filename = f"{self.output_dir}/stats_analysis.json"
        with open(stats_filename, "w") as f:
            json.dump(stats_results, f, indent=2)
        self.logger.info(f"Stats results saved to {stats_filename}")
        
        return df, stats_results


    def _run_statistical_analysis(self, csv_file_path):
        self.logger.info("Running statistical analysis")
        
        results = analyze_emotion_and_intensity_effects(csv_file_path)
        
        # Save analysis results
        return results
