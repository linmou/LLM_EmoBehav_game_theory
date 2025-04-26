import pickle
from transformers import pipeline
from neuro_manipulation.repe.pipelines import get_pipeline
from neuro_manipulation.utils import (
    load_model_tokenizer, 
    all_emotion_rep_reader, 
    primary_emotions_concept_dataset,
    dict_to_unique_code
)
from neuro_manipulation.prompt_formats import PromptFormat
from constants import Emotions
from vllm import LLM

def setup_model_and_tokenizer(config):
    model, tokenizer = load_model_tokenizer(
        config['model_name_or_path'],
        expand_vocab=False
    )
    
    prompt_format = PromptFormat(tokenizer)

    return model, tokenizer, prompt_format

def load_emotion_readers(config, model, tokenizer, hidden_layers):
    args = {
        'emotions': Emotions.get_emotions(),
        'data_dir': config['data_dir'],
        'model_name_or_path': config['model_name_or_path'],
        'rep_token': config['rep_token'],
        'hidden_layers': hidden_layers,
        'n_difference': config['n_difference'],
        'direction_method': config['direction_method'],
    }
    
    arg_codes = dict_to_unique_code(args)
    
    try:
        if not config['rebuild']:
            emotion_rep_readers = pickle.load(
                open(f'neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl', 'rb')
            )
            if emotion_rep_readers.get('args') == args:
                return emotion_rep_readers
    except:
        pass

    data = primary_emotions_concept_dataset(
        config['data_dir'],
        model_name=config['model_name_or_path'],
        tokenizer=tokenizer,
    )
    
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    return all_emotion_rep_reader(
        data, config['emotions'], rep_reading_pipeline,
        hidden_layers, config['rep_token'], config['n_difference'],
        config['direction_method'], read_args=args,
        save_path=f'neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl'
    ) 