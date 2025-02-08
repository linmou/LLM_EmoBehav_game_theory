from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import pipeline
import pandas as pd
from functools import partial
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

from neuro_manipulation.datasets.prisoner_delimma_dataset import PrisonerDelimmaDataset
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
from neuro_manipulation.model_utils import setup_model_and_tokenizer, load_emotion_readers
from neuro_manipulation.utils import oai_response
from api_configs import OAI_CONFIG

class EmotionGameExperiment:
    def __init__(self, repe_eng_config, model_config, game_config, sample_num=None, batch_size=None, repeat=1):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing experiment with model: {repe_eng_config['model_name_or_path']}")
        self.repe_eng_config = repe_eng_config
        self.model_config = model_config
        self.game_config = game_config
        
        self.repeat = repeat
        self.sample_num = sample_num
            
        if batch_size is not None:
            self.repe_eng_config['batch_size'] = batch_size
        
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
        
        self.rep_control_pipeline = pipeline(
            "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.model_config['control_layer_id'],
            block_name=self.repe_eng_config['block_name'],
            control_method=self.repe_eng_config['control_method']
        )
        
        self.reaction_prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, 
            response_format=self.game_config['decision_class']
        )
        
        self.cur_emotion = None
        self.cur_coeff = None
        
        self.llm_client = OpenAI(**OAI_CONFIG)

    def run_experiment(self):
        self.logger.info("Starting experiment")
        results = []
        
        for emotion in self.repe_eng_config['emotions']:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion
            
            self.logger.info(f"Creating dataset with sample_num={self.sample_num}")
            emo_dataset = PrisonerDelimmaDataset(
                'groupchat/scenarios/Prisoners_Dilemma_all_data_samples.json',
                partial(self.reaction_prompt_wrapper.__call__, 
                       user_messages="Remember you are Alice, an average American. What is your option? Choose one option shown above."),
                sample_num=self.sample_num,
                payoff_matrix=self.game_config['payoff_matrix']
            )
            
            data_loader = DataLoader(emo_dataset, batch_size=self.repe_eng_config['batch_size'], shuffle=False)
            
            for coeff in self.repe_eng_config['coeffs']:
                self.logger.info(f"Processing coefficient: {coeff}")
                self.cur_coeff = coeff
                results.extend(self._infer_with_activation(rep_reader, data_loader))
        
        self.cur_emotion = 'None'
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
                # Process current batch and put in queue
                control_outputs = self.rep_control_pipeline(
                    batch['prompt'] * self.repeat,
                    activations=activations,
                    batch_size=self.repe_eng_config['batch_size'],
                    max_new_tokens=self.repe_eng_config['max_new_tokens'],
                    do_sample=True,
                    top_p=0.95
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
            option_ids = list(executor.map(self._post_process_single_output, output_data))
        
        # Combine results
        for i, (option_id, (generated_text, options)) in enumerate(zip(option_ids, output_data)):
            results.append({
                'emotion': self.cur_emotion,
                'coefficient': self.cur_coeff,
                'input': batch['prompt'][i % batch_size],
                'options': options,
                'output': generated_text,
                'option_id': option_id,
                'repeat_idx': i // batch_size
            })
        
        return results

    def _post_process_single_output(self, output_data):
        generated_text, options = output_data
        self.logger.info(f"Generated text: {generated_text}")
        
        try:
            option_id = [j+1 for j, o in enumerate(options) if o.lower() in generated_text.lower()][0]
            self.logger.info(f"Found option_id {option_id} directly from text")
        except:
            self.logger.info("Using LLM to determine option_id")
            option_id = self._get_option_id_from_llm(generated_text, options)
            self.logger.info(f"Got option_id from LLM: {option_id}")
            
        return option_id

    def _get_option_id_from_llm(self, generated_text, options):
        prompt = f"Given the options {[ 'option ' + str(oid+1) + ' : ' + opt for oid, opt in enumerate(options)]}. "
        prompt += f"What is the option id for the following generated text: {generated_text}. "
        prompt += f"Only return the option id number, no other text. If the generated text is not in the options, return -1. Option ranges from 1 to {len(options)}."
        
        for _ in range(3):
            try:
                option_id = oai_response(prompt, model="gpt-4o-mini", client=self.llm_client)
                return int(option_id)
            except Exception as e:
                self.logger.info(f"Error getting option_id from LLM: {e}")
                prompt = f"In previous run, there is an error: {e}. Please try again.\n\n{prompt}"
                option_id = oai_response(prompt, model="gpt-4o-mini", client=self.llm_client)
                return int(option_id)
        
        raise ValueError("Failed to get option_id from LLM")

    def _save_results(self, results):
        self.logger.info("Saving experiment results")
        df = pd.DataFrame(results)
        correlation = df['option_id'].corr(df['coefficient'])
        self.logger.info(f"Correlation coefficient between option_id and coefficient: {correlation}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prisoner_delimma_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_exp_results_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Results saved to {filename}")
        return df 