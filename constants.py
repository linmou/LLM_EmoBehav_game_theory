from enum import Enum

class Emotions(Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    
    def __init__(self, *args):
        super().__init__()
        # Check for prefix overlaps during enum initialization
        for other in self.__class__:
            if other == self:
                continue
            min_len = min(len(self.value), len(other.value))
            for i in range(3, min_len + 1):
                if self.value[:i] == other.value[:i]:
                    raise ValueError(
                        f"Prefix overlap detected between {self.value} and {other.value}: "
                        f"both share prefix '{self.value[:i]}'"
                    )

    @classmethod
    def from_string(cls, value: str) -> 'Emotions':
        """
        Get an Emotions enum member from its string value.
        
        Args:
            value: String representation of the emotion
            
        Returns:
            Corresponding Emotions enum member
            
        Raises:
            ValueError: If no matching emotion is found
        """
        value = value.lower().strip()
        for emotion in cls:
            emotion_value = emotion.value.lower()
            # Case 1: Exact match
            if emotion_value == value:
                return emotion
            # Case 2: Input is a prefix of emotion value (e.g., "hap" -> "happiness")
            if len(value) > 2 and emotion_value.startswith(value):
                return emotion
            # Case 3: Emotion value is a prefix of input (e.g., "happy" matches "happiness")
            if len(value) > 2 and value.startswith(emotion_value):
                return emotion

        raise ValueError(f"No emotion found matching '{value}'")
    
if __name__ == "__main__":
    print(Emotions.from_string("ang"))
    print(Emotions.from_string("happiness"))
    print(Emotions.from_string("sadness"))
    print(Emotions.from_string("SAD"))

    