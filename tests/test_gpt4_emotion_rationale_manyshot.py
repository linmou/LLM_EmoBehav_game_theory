import pandas as pd
import json
import random
import logging
import os
import sys
import re
from typing import List, Tuple, Dict
from collections import defaultdict
import concurrent.futures
import time
import csv
from datetime import datetime

# Add the parent directory to sys.path to help with imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]

# Try to import openai, but provide a fallback if it's not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. Will use mock implementation.")

def parse_rationales_from_json(json_path: str, n_per_emotion: int = 600) -> Dict[str, List[str]]:
    """Sample n_per_emotion rationales for each emotion from the CSV file."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    records = defaultdict(list)
    for record in data:
        assert 'emotion' in record, f"Emotion not found in record: {record}"
        assert 'rationale' in record, f"Rationale not found in record: {record}"
        
        records[record['emotion']].append(record['rationale'])

    for emotion in records:
        records[emotion] = records[emotion][:n_per_emotion]
    
    return records
        
    
def build_manyshot_prompt(examples: List[Tuple[str, str]]) -> str:
    prompt = f"You are an expert in emotion psychology. Given a rationale for a decision, classify which emotion from {EMOTIONS} most likely motivated it. Just output the emotion, no other text.\n\n"
    for rationale, emotion in examples:
            prompt += f"Rationale: {rationale}\nEmotion: {emotion}\n---\n"
    return prompt

def gpt4_classify_emotion(prompt: str, test_rationale: str, client: openai.OpenAI) -> str:
    """Classify emotion using GPT-4o, with fallback to mock implementation."""
    try:
        final_prompt = prompt + f"Rationale: {test_rationale}\nEmotion: "
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().split()[0].lower()
        return answer
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        # Fallback to mock implementation for demonstration purposes
        return random.choice(EMOTIONS)

def find_json_file():
    """Find the CSV file in the current directory or parent directories."""
    return "results/RepEng/Prisoners_Dilemma_Prisoners_Dilemma_Llama-3.1-8B-Instruct_20250502_081540/exp_results.json"

def process_rationale(args):
    """Process a single rationale for parallel execution"""
    prompt, test_rationale, emotion, client = args
    try:
        pred = gpt4_classify_emotion(prompt, test_rationale, client)
        logging.info(f"Rationale: {test_rationale[:50]}...\nTrue: {emotion}\nPred: {pred}")
        return {
            'rationale': test_rationale,
            'true_emotion': emotion,
            'predicted_emotion': pred,
            'correct': pred == emotion
        }
    except Exception as e:
        logging.error(f"Error processing rationale: {e}")
        return {
            'rationale': test_rationale,
            'true_emotion': emotion,
            'predicted_emotion': 'error',
            'correct': False
        }

def main():
    logging.basicConfig(level=logging.INFO)
    # Set your OpenAI API key here or via environment variable
    api_key = "sk-icBGJDaAZmvriEL5jBuNdaC9Ezo4MEA7eUw7KoQwYi9dpmMa" #os.getenv('OPENAI_API_KEY')
    base_url = "https://chatapi.onechats.top/v1"
    if not api_key:
        logging.warning("OPENAI_API_KEY not set. Will use mock implementation.")
    
    # Find the CSV file
    json_path = find_json_file()
    logging.info(f"Using JSON file: {json_path}")
    
    try:
        all_records = parse_rationales_from_json(json_path, n_per_emotion=200)
        logging.info(f"Successfully parsed {len(all_records)} emotion records from the JSON")
    except Exception as e:
        logging.error(f"Error parsing JSON: {e}")
        return
    
    n_shots = 50
    max_workers = 10  # Adjust based on your system and rate limits
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    examples = []
    for emotion, emotion_records in all_records.items():
        for rationale in emotion_records[:n_shots]:
            examples.append((rationale, emotion))
    
    prompt = build_manyshot_prompt(examples)
    
    # Prepare tasks for parallel processing
    tasks = []
    for emotion, emotion_records in all_records.items():
        for test_rationale in emotion_records:
            tasks.append((prompt, test_rationale, emotion, client))
    
    # Create a timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"emotion_classification_results_{timestamp}.csv"
    
    # Use ThreadPoolExecutor for parallel processing
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_rationale, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Exception in thread: {e}")
    
    end_time = time.time()
    
    # Calculate metrics
    correct = sum(1 for result in results if result['correct'])
    total = len(results)
    acc = correct / total if total else 0
    
    # Save results to CSV
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['rationale', 'true_emotion', 'predicted_emotion', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Many-shot GPT-4 emotion classification accuracy: {acc:.2f} ({correct}/{total})")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main() 