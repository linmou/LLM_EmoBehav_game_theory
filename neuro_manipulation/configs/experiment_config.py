import os
from typing import Any

import yaml
from dotenv import load_dotenv
from games.game_configs import get_game_config


DEFAULT_USER_HOME = "/home/jjl7137"


def expand_env_placeholders(value: Any) -> Any:
    """Recursively expand env vars in YAML-derived config values."""
    if isinstance(value, dict):
        return {k: expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(v) for v in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        # Fallback if USER_HOME is not exported in current shell.
        return expanded.replace("${USER_HOME}", DEFAULT_USER_HOME)
    return value

def update_repe_config_from_yaml(base_repe_config, yaml_config):
    """
    Update repe_config with values from YAML config.
    
    Args:
        base_repe_config: Base repe configuration dictionary
        yaml_config: Full YAML configuration dictionary
        
    Returns:
        Updated repe_config dictionary
        
    Supported YAML paths:
        repe_config.* -> Direct updates to repe_config
        experiment.emotions -> Updates emotions list
        experiment.data.data_dir -> Updates data_dir
        experiment.llm.model_name -> Updates model_name_or_path (fallback)
    """
    updated_config = base_repe_config.copy()
    
    if 'repe_config' in yaml_config:
        # back compatability for experiments in neuro_manipulation
        repe_section = yaml_config['repe_config']
    elif 'repe_eng_config' in yaml_config:
        repe_section = yaml_config['repe_eng_config']
    else:
        # Direct repe_config section updates
        repe_section = yaml_config
        
    for key, value in repe_section.items():
        updated_config[key] = value
        print(f"✓ Updated repe_config.{key} = {value}")

    if 'emotions' in yaml_config and 'emotions' not in repe_section:
        updated_config['emotions'] = yaml_config['emotions']
        print(f"✓ Updated repe_config.emotions = {yaml_config['emotions']}")
        
    return updated_config

def get_repe_eng_config(model_name, yaml_config_path=None, yaml_config=None):
    """
    Get repe configuration, optionally updated from YAML config.
    
    Args:
        model_name: Model name for base configuration
        yaml_config_path: Path to YAML config file (optional)
        yaml_config: Pre-loaded YAML config dictionary (optional)
        
    Returns:
        Dictionary with repe configuration
    """
    # Base configuration
    base_config = {
        'emotions': ["happiness", "sadness", "anger", "fear", "disgust", "surprise"],
        'data_dir': "data/stimulus/crowd-enVent_textlike",
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
    
    # Update from YAML if provided
    if yaml_config_path is not None:
        yaml_data = get_exp_config(yaml_config_path)
        return update_repe_config_from_yaml(base_config, yaml_data)
    elif yaml_config is not None:
        return update_repe_config_from_yaml(base_config, yaml_config)
    else:
        return base_config

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
    load_dotenv(override=False)
    with open(config_path, 'r') as file:
       raw_config = yaml.safe_load(file)
    return expand_env_placeholders(raw_config)
