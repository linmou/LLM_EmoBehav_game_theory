"""
Simple emotion check dataset for validating emotion manipulation.
Tests basic emotional state awareness with direct questions and classifies responses by emotion category.
"""

import re
from typing import Any, Dict, List

from .base import BaseBenchmarkDataset
from ..data_models import BenchmarkItem


class EmotionCheckDataset(BaseBenchmarkDataset):
    """
    Simple dataset to test emotion manipulation by asking direct questions
    about the model's current emotional state and classifying responses.
    
    Returns detected emotion category instead of binary scoring.
    """
    
    # Emotion expression dictionary for classification
    EMOTION_EXPRESSIONS = {
        "anger": ["angry", "mad", "furious", "irritated", "annoyed", "rage", "upset", "frustrated", "pissed", "livid"],
        "happiness": ["happy", "joyful", "pleased", "content", "cheerful", "delighted", "excited", "glad", "blissful", "ecstatic"],
        "sadness": ["sad", "depressed", "melancholy", "down", "upset", "disappointed", "blue", "gloomy", "miserable", "dejected"],
        "fear": ["afraid", "scared", "anxious", "worried", "nervous", "terrified", "panicked", "frightened", "fearful", "alarmed"],
        "disgust": ["disgusted", "revolted", "repulsed", "sick", "nauseous", "appalled", "disturbed", "repelled", "grossed", "nauseated"],
        "surprise": ["surprised", "amazed", "astonished", "shocked", "startled", "stunned", "bewildered", "astounded", "flabbergasted", "dumbfounded"],
        "neutral": ["neutral", "calm", "peaceful", "fine", "okay", "normal", "relaxed", "stable", "balanced", "composed"]
    }
     
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index with prompt formatting"""
        item = self.items[idx]

        # Extract options from metadata for multiple choice questions
        options = None
        if item.metadata and "options" in item.metadata:
            options = item.metadata["options"]

        # Create prompt using wrapper or default format
        if self.prompt_wrapper:
            prompt = self.prompt_wrapper(
                context=item.context if item.context else "",
                question=item.input_text,
                answer=item.ground_truth,
                options=options,
            )

        else:
            # Default prompt format
            if item.context:
                prompt = (
                    f"Context: {item.context}\nQuestion: {item.input_text}\nAnswer:"
                )
            else:
                prompt = f"{item.input_text}\nAnswer:"

        # Transform ground truth if answer wrapper provided
        ground_truth = (
            self.answer_wrapper(item.ground_truth) 
            if self.answer_wrapper 
            else item.ground_truth
        )

        return {"item": item, "prompt": prompt, "ground_truth": ground_truth}

     
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """Load emotion check questions from JSONL file"""
        
        # Load raw data from file (following standard pattern)
        raw_data = self._load_raw_data()
        
        items = []
        for item_data in raw_data:
            items.append(BenchmarkItem(
                id=item_data["id"],
                input_text=item_data["input"],
                context=None,  # No additional context needed
                ground_truth=item_data["ground_truth"],  # List of valid emotion expressions
                metadata={
                    "category": item_data["category"],
                    "expects_emotion": True,
                    "response_type": "single_word"
                }
            ))
            
        return items
    
    def _classify_emotion_response(self, response: str) -> str:
        """
        Classify response into one of the 6 emotion categories + neutral.
        
        Args:
            response: Model's response text
            
        Returns:
            Detected emotion category name or "unknown"
        """
        if not response:
            return "unknown"
            
        # Clean and extract first meaningful word
        response_clean = response.strip().lower()
        # Remove punctuation and get first word
        first_word = re.sub(r'[^\w\s]', '', response_clean).split()[0] if response_clean.split() else ""
        
        if not first_word:
            return "unknown"
            
        # Check which emotion category this word belongs to
        for emotion_category, expressions in self.EMOTION_EXPRESSIONS.items():
            if first_word in expressions:
                return emotion_category
                
        return "unknown"
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str, prompt: str = "") -> float:
        """
        Use LLM evaluation for emotion classification if configured, otherwise fallback to rule-based.
        
        Returns score from 0-1 based on emotion classification success.
        """
        # If LLM evaluation is configured, use GPT-4o-mini for evaluation
        if hasattr(self, 'llm_eval_config') and self.llm_eval_config is not None:
            from ..evaluation_utils import llm_evaluate_response
            
            # Construct evaluation query with the configured prompt template
            eval_prompt = self.llm_eval_config.get('evaluation_prompt', '')
            query = eval_prompt.format(
                question=prompt,
                response=response
            )
            
            try:
                # Call LLM evaluation
                result = llm_evaluate_response(
                    system_prompt="You are an expert emotion classifier. Always respond with valid JSON format.",
                    query=query,
                    llm_eval_config={
                        "model": self.llm_eval_config.get("model", "gpt-4o-mini"),
                        "temperature": self.llm_eval_config.get("temperature", 0.1)
                    }
                )
                
                # Extract emotion classification from result
                detected_emotion = result.get("emotion", "neutral").lower()
                confidence = result.get("confidence", 0.5)
                
                # Return confidence as score (0.0-1.0) - higher confidence = higher score
                # Could also implement emotion-specific scoring logic here
                return float(confidence)
                
            except Exception as e:
                print(f"LLM evaluation failed: {e}")
                # Fall back to rule-based on error
                detected_emotion = self._classify_emotion_response(response)
                return 1.0 if detected_emotion != "unknown" else 0.0
            
        # Fallback to rule-based classification
        detected_emotion = self._classify_emotion_response(response)
        return 1.0 if detected_emotion != "unknown" else 0.0
    
    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> dict[str, Any]:
        """
        Return detailed emotion classification metrics.
        """
        detected_emotion = self._classify_emotion_response(response)
        
        return {
            "overall_score": 1.0 if detected_emotion != "unknown" else 0.0,
            "detected_emotion": detected_emotion,
            "response_word": response.strip().lower().split()[0] if response.strip() else "",
            "classification_success": detected_emotion != "unknown"
        }
    
    def get_task_metrics(self, task_name: str) -> List[str]:
        """Available metrics for emotion check"""
        return ["emotion_classification", "detection_rate", "response_relevance"]