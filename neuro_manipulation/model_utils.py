import pickle
from transformers import pipeline
from neuro_manipulation.utils import (
    load_model_tokenizer, 
    all_emotion_rep_reader, 
    primary_emotions_concept_dataset,
    dict_to_unique_code
)
from neuro_manipulation.prompt_formats import PromptFormat

def setup_model_and_tokenizer(config):
    prompt_format = PromptFormat.get(config['model_name_or_path'])
    user_tag = prompt_format.user_tag
    assistant_tag = prompt_format.assistant_tag

    model, tokenizer = load_model_tokenizer(
        config['model_name_or_path'],
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        expand_vocab=False
    )

    return model, tokenizer, prompt_format, user_tag, assistant_tag

def load_emotion_readers(config, model, tokenizer, hidden_layers):
    args = {
        'emotions': config['emotions'],
        'data_dir': config['data_dir'],
        'model_name_or_path': config['model_name_or_path'],
        'user_tag': config['user_tag'],
        'assistant_tag': config['assistant_tag'],
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
        user_tag=config['user_tag'],
        assistant_tag=config['assistant_tag']
    )
    
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    return all_emotion_rep_reader(
        data, config['emotions'], rep_reading_pipeline,
        hidden_layers, config['rep_token'], config['n_difference'],
        config['direction_method'], read_args=args,
        save_path=f'neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl'
    ) 