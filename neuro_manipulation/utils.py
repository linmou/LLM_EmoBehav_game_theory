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

def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the emotion of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    # template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise",]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
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
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
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
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
    except KeyError:
        model = MistralForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
        
    use_fast_tokenizer = False #"LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    if expand_vocab:
        tokenizer.add_special_tokens({'additional_special_tokens': [user_tag, assistant_tag]})
        model.resize_token_embeddings(len(tokenizer))
    
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
    data_dir = "/home/representation-engineering/data/emotions"
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    user_tag =  "[INST]"
    assistant_tag =  "[/INST]"
    
    model, tokenizer = load_model_tokenizer(model_name_or_path,user_tag=user_tag, 
                                            assistant_tag=assistant_tag,
                                            expand_vocab=True)

    data = primary_emotions_concept_dataset(data_dir, user_tag=user_tag, 
                                            assistant_tag=assistant_tag,
                                            )

    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers+8, -1))
    n_difference = 1
    direction_method = 'pca'

    rep_reading_pipeline = pipeline( "rep-reading", model=model, tokenizer=tokenizer)
    prob_cal_pipeline = pipeline( "rep-reading&prob-calc", model=model, tokenizer=tokenizer, user_tag=user_tag, assistant_tag=assistant_tag)

    emotion_rep_readers = all_emotion_rep_reader(data, emotions, rep_reading_pipeline, hidden_layers, rep_token, n_difference, direction_method)
    
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

