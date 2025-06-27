import yaml
from games.game_configs import get_game_config

def get_repe_eng_config(model_name):
    
    return {
        'emotions': ["happiness", "sadness", "anger", "fear", "disgust", "surprise"],
        'data_dir': "/home/jjl7137/representation-engineering/data/emotions",
        'model_name_or_path': model_name,
        'coeffs': [0.5, 1, 1.5],
        'max_new_tokens': 450,
        'block_name': "decoder_block",
        'control_method': "reading_vec",
        'acc_threshold': 0.,
        'rebuild': False,
        'n_difference': 1,
        'direction_method': 'pca',
        'rep_token': -1,
        'control_layer_id': get_model_config(model_name)
    }

def get_model_config(model_name):
    """
    Deprecated. 
    Now we use the middel 1/3 layers by default to control.    
    """
    
    if 'mistral-7b' in model_name.lower() or 'llama-3' in model_name.lower():
        control_layer_id = list(range(-5, -18, -1))
    else:  # llama default
        control_layer_id = list(range(-11, -30, -1))
    
    return control_layer_id

def get_exp_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)