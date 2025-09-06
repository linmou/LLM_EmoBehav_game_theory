"""
Simple emotion check dataset for validating emotion manipulation.
Tests basic emotional state awareness with direct questions and classifies responses by emotion category.
"""

import re
from typing import Any, List

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
        Classify response into emotion categories.
        
        Note: This returns a score for compatibility but the main value is in 
        evaluate_with_detailed_metrics() which returns the classification.
        """
        detected_emotion = self._classify_emotion_response(response)
        
        # For compatibility, return 1.0 if any emotion detected, 0.0 if unknown
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