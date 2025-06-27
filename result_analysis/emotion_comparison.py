import json
import os
import requests
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import concurrent.futures
import time
from tqdm import tqdm

from api_configs import OAI_CONFIG
openai_api_key = OAI_CONFIG["api_key"]
openai_base_url = OAI_CONFIG["base_url"]
target_file = "results/RepEng/Prisoners_Dilemma_New_Template_vllm_Llama-3.1-8B-Instruct_20250424_013532/exp_results.json"

# Maximum number of concurrent requests
MAX_WORKERS = 10
# Rate limiting: requests per second
RATE_LIMIT = 10
# Seconds to sleep between batches to avoid hitting rate limits
RATE_LIMIT_SLEEP = 1.0
ITEM_START = 0
ITEM_END = 100000000

# Define the target emotions for comparison
TARGET_EMOTION = "anger"
NEUTRAL_EMOTION = "neutral"

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def analyze_emotion_expression(rationale, emotion):
    """
    Use GPT-4o-mini to analyze if the rationale expresses the given emotion.
    
    Args:
        rationale (str): The rationale text
        emotion (str): The emotion to check for
        
    Returns:
        bool: True if the rationale expresses the emotion, False otherwise
        float: Confidence score (0-1)
        str: Explanation of the analysis
    """
    # GPT-4o-mini API endpoint
    api_url = f"{openai_base_url}/chat/completions"
    
    # Get API key
    api_key = openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
    Analyze if the following rationale of a decision is directly driven by the emotion of {emotion} of the character who writes the rationale:
    
    "{rationale}"
    
    Respond with a JSON object with the following structure:
    {{
        "expresses_emotion": true/false,
        "confidence": 0.0-1.0,
        "explanation": "brief explanation"
    }}
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an emotion analysis assistant. You will be given with a rationale of a decision and an emotion. You need to evaluate if the character who writes the rationale is directly driven by the emotion."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        result_content = json.loads(result["choices"][0]["message"]["content"])
        
        # Ensure we're returning Python native types
        return (
            bool(result_content["expresses_emotion"]), 
            float(result_content["confidence"]),
            str(result_content["explanation"])
        )
    except Exception as e:
        print(f"Error analyzing emotion expression: {e}")
        return False, 0.0, f"Error: {str(e)}"

def process_item(item, item_id, total_items):
    """
    Process a single item for parallel execution.
    
    Args:
        item (dict): The item to process
        item_id (int): The ID of the item
        total_items (int): Total number of items
        
    Returns:
        dict: The processed result
    """
    emotion = item.get('emotion')
    rationale = item.get('rationale')
    category = item.get('category')
    
    if not all([emotion, rationale, category]):
        print(f"Missing data in item {item_id}, skipping...")
        return None
    
    # Analyze if rationale expresses the emotion
    expresses, confidence, explanation = analyze_emotion_expression(rationale, emotion)
    
    # Create result with Python native types
    result = {
        "item_id": int(item_id),
        "scenario": str(item['scenario']),
        "emotion": str(emotion),
        "category": int(category),
        "expresses_emotion": bool(expresses),
        "confidence": float(confidence),
        "explanation": str(explanation),
        "rationale": str(rationale)
    }
    
    return result

def main():
    # Check if file exists
    if not os.path.isfile(target_file):
        print(f"File not found: {target_file}")
        return
    
    # Read JSON data
    try:
        with open(target_file, 'r') as f:
            data = json.load(f)[ITEM_START:ITEM_END]
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Filter items by emotion (target emotion vs neutral)
    target_emotion_items = [item for item in data if item["emotion"].lower() == TARGET_EMOTION.lower()]
    neutral_emotion_items = [item for item in data if item["emotion"].lower() == NEUTRAL_EMOTION.lower()]
    
    print(f"Found {len(target_emotion_items)} {TARGET_EMOTION} items and {len(neutral_emotion_items)} neutral items to analyze")
    
    # Prepare results storage
    results = []
    emotion_expression_by_type = {
        "target": [],
        "neutral": []
    }
    
    # Process both groups
    all_items = [(item, "target") for item in target_emotion_items] + [(item, "neutral") for item in neutral_emotion_items]
    
    # Process items in parallel batches
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list to store the futures
        futures = []
        
        # Submit all tasks
        for i, (item, item_type) in enumerate(all_items):
            future = executor.submit(process_item, item, i, len(all_items))
            futures.append((future, item_type))
        
        # Process completed futures with a progress bar
        for (future, item_type) in tqdm(
            [(f, t) for f, t in futures], 
            total=len(futures), 
            desc=f"Analyzing {TARGET_EMOTION} vs neutral emotions"
        ):
            result = future.result()
            if result:
                results.append(result)
                
                # Add to type-based collections
                emotion_expression_by_type[item_type].append(bool(result["expresses_emotion"]))
                
                # Sleep briefly to prevent hitting rate limits
                if len(results) % RATE_LIMIT == 0:
                    time.sleep(RATE_LIMIT_SLEEP)
    
    # Sort results by item_id for consistency
    results.sort(key=lambda x: x["item_id"])
    
    # Print detailed results
    for result in results:
        print(f"\nItem {result['item_id']+1}/{len(all_items)}:")
        print(f"Emotion: {result['emotion']}")
        print(f"Rationale: {result['rationale']}")
        print(f"Expresses {result['emotion']}: {result['expresses_emotion']} (confidence: {result['confidence']:.2f})")
        print(f"Explanation: {result['explanation']}")
    
    # Calculate expression rates
    target_expression_rate = sum(emotion_expression_by_type["target"]) / len(emotion_expression_by_type["target"]) if emotion_expression_by_type["target"] else 0
    neutral_expression_rate = sum(emotion_expression_by_type["neutral"]) / len(emotion_expression_by_type["neutral"]) if emotion_expression_by_type["neutral"] else 0
    
    # Chi-square test if we have enough data
    if emotion_expression_by_type["target"] and emotion_expression_by_type["neutral"]:
        observed = [
            [sum(emotion_expression_by_type["target"]), len(emotion_expression_by_type["target"]) - sum(emotion_expression_by_type["target"])],
            [sum(emotion_expression_by_type["neutral"]), len(emotion_expression_by_type["neutral"]) - sum(emotion_expression_by_type["neutral"])]
        ]
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        correlation_significant = p < 0.05
    else:
        chi2, p, correlation_significant = None, None, False
    
    # Save results with custom encoder to handle NumPy types
    output_file = "result_analysis/emotion_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "target_emotion": TARGET_EMOTION,
                "target_expression_rate": float(target_expression_rate),
                "neutral_expression_rate": float(neutral_expression_rate),
                "chi2_value": None if chi2 is None else float(chi2),
                "p_value": None if p is None else float(p),
                "correlation_significant": bool(correlation_significant)
            },
            "individual_results": results
        }, f, indent=2, cls=NumpyEncoder)
    
    # Create visualization
    emotion_types = [f"{TARGET_EMOTION.capitalize()}", "Neutral"]
    expression_rates = [target_expression_rate, neutral_expression_rate]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotion_types, expression_rates, color=['red', 'gray'])
    plt.ylabel('Emotion Expression Rate')
    plt.title(f'Emotion Expression Rate: {TARGET_EMOTION.capitalize()} vs Neutral')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.savefig(f"result_analysis/{TARGET_EMOTION}_vs_neutral_expression.png")
    
    # Print summary
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"{TARGET_EMOTION.capitalize()} emotion expression rate: {target_expression_rate:.2f}")
    print(f"Neutral emotion expression rate: {neutral_expression_rate:.2f}")
    if chi2 is not None and p is not None:
        print(f"Chi-square test: χ² = {chi2:.2f}, p = {p:.4f}")
        print(f"Difference is {'statistically significant' if correlation_significant else 'not statistically significant'}")
    print(f"Results saved to {output_file}")
    print(f"Visualization saved to result_analysis/{TARGET_EMOTION}_vs_neutral_expression.png")

if __name__ == "__main__":
    main() 