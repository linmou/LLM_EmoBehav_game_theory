import argparse
import torch
import yaml
from transformers import AutoModel, AutoTokenizer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_path):
    """Attempts to load a model and tokenizer from the given path."""
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(model_path):
            logging.error(f"Model path does not exist or is not a directory: {model_path}")
            return False, f"Path not found or not a directory: {model_path}"

        logging.info(f"Attempting to load tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info(f"Successfully loaded tokenizer for: {model_path}")

        logging.info(f"Attempting to load model from: {model_path}")
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", token=True, trust_remote_code=True).eval()
        logging.info(f"Successfully loaded model for: {model_path}")
        
        return True, None
    except Exception as e:
        logging.error(f"Failed to load model/tokenizer from {model_path}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Test loading Hugging Face models from a YAML config.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config_file}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return

    if not config or 'experiment' not in config or 'models' not in config['experiment']:
        logging.error("Invalid YAML structure. Expected 'experiment.models' key.")
        return

    model_paths = config['experiment']['models']
    
    if not isinstance(model_paths, list):
        logging.error("'experiment.models' should be a list of model paths.")
        return

    logging.info(f"Found {len(model_paths)} model(s) to test.")

    all_models_loaded = True
    for model_path in model_paths:
        if not isinstance(model_path, str):
            logging.warning(f"Skipping invalid model path entry (not a string): {model_path}")
            all_models_loaded = False
            continue
        
        logging.info(f"--- Testing model: {model_path} ---")
        success, error_message = load_model_and_tokenizer(model_path)
        if success:
            logging.info(f"Successfully loaded: {model_path}")
        else:
            logging.error(f"Failed to load: {model_path}. Error: {error_message}")
            all_models_loaded = False
        logging.info("------------------------------------")

    if all_models_loaded:
        logging.info("All specified models loaded successfully!")
    else:
        logging.warning("Some models failed to load. Please check the logs.")

if __name__ == "__main__":
    main() 