import base64
from dataclasses import dataclass
import hashlib
from typing import Dict, List
from pathlib import Path
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, MistralForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import random
import os
import json

from vllm import LLM


from neuro_manipulation.repe import repe_pipeline_registry
import pickle
repe_pipeline_registry()


@dataclass
class AnswerProbabilities:
    # query: str
    # ans: str
    ans_probabilities: List[float]
    ans_ids: List[int]
    input_ids: List[int]
    input_text: str
    emotion: str
    emotion_activation: Dict[int, float]
    
    @property
    def emotion_activation_mean_last_layer(self):
        return np.mean(list(self.emotion_activation[-1].values()))

def primary_emotions_concept_dataset(data_dir, model_name=None, tokenizer=None, system_prompt=None, seed=0):
    """
    Create dataset of emotion scenarios using proper model-specific prompt formatting.
    
    Args:
        data_dir: Directory containing emotion scenario JSON files
        model_name: Name of the model to format prompts for
        tokenizer: Optional tokenizer for more accurate prompt formatting
        system_prompt: Optional system prompt, if None no system prompt will be used
        seed: Random seed for shuffling
    
    Returns:
        Dictionary of formatted emotion datasets
    """
    from neuro_manipulation.prompt_formats import PromptFormat, ManualPromptFormat
    from transformers import AutoTokenizer
    import random
    import numpy as np
    import json
    import os
    
    random.seed(seed)
    
    # Setup prompt format
    if tokenizer is None and model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer is not None:
        prompt_format = PromptFormat(tokenizer)
    elif model_name is not None:
        # Use ManualPromptFormat as fallback
        format_cls = ManualPromptFormat.get(model_name)
        user_tag = format_cls.user_tag
        assistant_tag = format_cls.assistant_tag
    else:
        # Default to empty tags if no model info provided
        user_tag = ''
        assistant_tag = ''
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json'), 'r', encoding='utf-8') as file:
            raw_data[emotion] = list(set(json.load(file)))
    
    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)
        
        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        # Format the data using prompt format
        emotion_test_data = []
        emotion_train_data = []
        
        for scenario in data_:
            user_message = f"Consider the emotion of the following scenario:\nScenario: {scenario}\nAnswer:"
            if tokenizer is not None:
                formatted_prompt = prompt_format.build(system_prompt, [user_message], [])
                emotion_test_data.append(formatted_prompt)
            else:
                if system_prompt:
                    formatted_prompt = f"{user_tag} {system_prompt}\n\n{user_message} {assistant_tag} "
                else:
                    formatted_prompt = f"{user_tag} {user_message} {assistant_tag} "
                emotion_test_data.append(formatted_prompt)
        
        for scenario in data:
            user_message = f"Consider the emotion of the following scenario:\nScenario: {scenario}\nAnswer:"
            if tokenizer is not None:
                formatted_prompt = prompt_format.build(system_prompt, [user_message], [])
                emotion_train_data.append(formatted_prompt)
            else:
                if system_prompt:
                    formatted_prompt = f"{user_tag} {system_prompt}\n\n{user_message} {assistant_tag} "
                else:
                    formatted_prompt = f"{user_tag} {user_message} {assistant_tag} "
                emotion_train_data.append(formatted_prompt)
        
        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0] * len(emotion_test_data)]}
        }
    
    return formatted_data


def test_direction(hidden_layers, rep_reading_pipeline, rep_reader, test_data, rep_token=-1):
    H_tests = rep_reading_pipeline(
        test_data['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        rep_reader=rep_reader,
        batch_size=32)
    
    results = {layer: {} for layer in hidden_layers}
    rep_readers_means = {layer: 0 for layer in hidden_layers}

    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        rep_readers_means[layer] = np.mean(H_test)
        H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
        
        sign = rep_reader.direction_signs[layer]

        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(H) == H[0] for H in H_test])
        
        results[layer] = cors
    
    return results, rep_readers_means



def get_rep_reader(rep_reading_pipeline, train_data, test_data, hidden_layers, rep_token=-1, n_difference=1, direction_method='pca'):
    rep_reader = rep_reading_pipeline.get_directions(
        train_data['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=train_data['labels'], 
        direction_method=direction_method,
    )

    result , _ = test_direction(hidden_layers, rep_reading_pipeline, rep_reader, test_data)
    print(result)
    
    return rep_reader, result



def prob_cal_record(prob_cal_pipeline, dataset, emotion, rep_token, hidden_layers, rep_reader, save_path='record.pkl'):
    assert save_path.endswith('.pkl')
    
    records = []
    for b_out in prob_cal_pipeline(
        dataset, 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        rep_reader=rep_reader,
        batch_size=4):
        bsz = len(b_out['ans_ids'])
        for bid in range(bsz):
            ans_record = AnswerProbabilities(
                ans_probabilities=b_out['ans_probabilities'][bid], 
                ans_ids=b_out['ans_ids'][bid], 
                input_ids=b_out['input_ids'][bid], 
                emotion=emotion, 
                input_text=prob_cal_pipeline.tokenizer.decode(b_out['input_ids'][bid]),
                emotion_activation= {layer: b_out[layer][bid] for layer in hidden_layers}
                )
            records.append(ans_record)
            
    # Save all records
    print(f"Saving records to {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(records, f)
    print(f"Records saved to {save_path}")
    
def load_model_tokenizer(model_name_or_path='gpt2', user_tag =  "[INST]", assistant_tag =  "[/INST]", expand_vocab=False):
    try:
        model = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), max_model_len=600, trust_remote_code=True, enforce_eager=True)
    except Exception as e:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True, trust_remote_code=True).eval()

    use_fast_tokenizer = False #"LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 
    # tokenizer.bos_token_id = 1 if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    # if expand_vocab:
    #     tokenizer.add_special_tokens({'additional_special_tokens': [user_tag, assistant_tag]})
    #     model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer
    
def all_emotion_rep_reader(data, emotions,  rep_reading_pipeline, hidden_layers, rep_token, n_difference, direction_method, save_path='exp_records/emotion_rep_reader.pkl', read_args=None):
    
    # args = {
    #     'rep_token': rep_token,
    #     'hidden_layers': hidden_layers,
    #     'n_difference': n_difference,
    #     'direction_method': direction_method,
    # }
    
    # if save_path is not None and os.path.exists(save_path) and not rebuild:
    #     with open(save_path, 'rb') as f:
    #         emotion_rep_readers = pickle.load(f)
    #         if 'args' in emotion_rep_readers:
    #             if emotion_rep_readers['args'] == args:
    #                 return emotion_rep_readers

    emotion_rep_readers = {'layer_acc': {}}
    for emotion in tqdm(emotions):
        train_data = data[emotion]['train']
        test_data = data[emotion]['test']
        rep_reader, layer_acc = get_rep_reader(rep_reading_pipeline=rep_reading_pipeline,
                                        train_data=train_data,
                                        test_data=test_data,
                                        hidden_layers=hidden_layers,
                                        rep_token=rep_token,
                                        n_difference=n_difference,
                                        direction_method=direction_method)
                                       
        emotion_rep_readers[emotion] = rep_reader
        emotion_rep_readers['layer_acc'][emotion] = layer_acc
    
    
    emotion_rep_readers['args'] = read_args
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(emotion_rep_readers, f)
        
    return emotion_rep_readers 

def dict_to_unique_code(dictionary):
    # Step 1: Serialize the dictionary to a JSON string
    serialized_dict = json.dumps(dictionary, sort_keys=True)
    
    # Step 2: Hash the serialized string using SHA-256
    hash_object = hashlib.sha256(serialized_dict.encode())
    hash_hex = hash_object.hexdigest()
    
    # Step 3: Convert the hash to a base64 string (optional)
    hash_bytes = hash_hex.encode('utf-8')
    unique_code = base64.urlsafe_b64encode(hash_bytes).rstrip(b'=').decode('utf-8')
    
    return unique_code

def oai_response(prompt, client, model="gpt-4o", response_format=None, ):
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
                # {'role': 'system', 'content': 'You are an avereage American.'},
                 {"role": "user", "content": prompt}],
        response_format=response_format,
        seed=42,
    )
    return response.choices[0].message.content



def main():
   
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise",]
    # emotions = ["stress"]
    data_dir = "/home/jjl7137/representation-engineering/data/emotions"
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    user_tag =  "[INST]"
    assistant_tag =  "[/INST]"
    
    # model, tokenizer = load_model_tokenizer(model_name_or_path,user_tag=user_tag, 
    #                                         assistant_tag=assistant_tag,
    #                                         expand_vocab=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="left", legacy=False, token=True, trust_remote_code=True)
    data = primary_emotions_concept_dataset(data_dir, model_name=model_name_or_path, tokenizer=tokenizer)

    # rep_token = -1
    # hidden_layers = list(range(-1, -model.config.num_hidden_layers+8, -1))
    # n_difference = 1
    # direction_method = 'pca'

    # rep_reading_pipeline = pipeline( "rep-reading", model=model, tokenizer=tokenizer)
    # prob_cal_pipeline = pipeline( "rep-reading&prob-calc", model=model, tokenizer=tokenizer, user_tag=user_tag, assistant_tag=assistant_tag)

    # emotion_rep_readers = all_emotion_rep_reader(data, emotions, rep_reading_pipeline, hidden_layers, rep_token, n_difference, direction_method)
    
    # for pid, emotional_prompt in tqdm(enumerate(Negative_SET)):
    #     dataset = EvalDatasets('MATH', prompt_modify_func=lambda question, answer: f' {emotional_prompt} {user_tag} {question} {assistant_tag}: Answer: {answer}')

    #     for emotion in emotions:
    #         rep_reader = emotion_rep_readers[emotion]
    #         prob_cal_record(prob_cal_pipeline=prob_cal_pipeline, 
    #                         dataset=dataset, 
    #                         emotion=emotion, 
    #                         rep_token=None, 
    #                         hidden_layers=hidden_layers, 
    #                         rep_reader=rep_reader, 
    #                         save_path=f'exp_records/MATH/prompt_{pid}_{emotion}_record.pkl') 
    


    
    
if __name__ == "__main__":
    main()

