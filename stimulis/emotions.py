from constants import Emotions

emotion2stimulus = {
    Emotions.ANGER: "Someone spreading false rumors about you behind your back",
    Emotions.DISGUST: "Finding a dead cockroach in your bedroom",
    Emotions.FEAR: "Waking up to strange noises in your house at midnight",
    Emotions.HAPPINESS: "Your pet running excitedly to greet you when you return home",
    Emotions.SADNESS: "Looking through old photos and realizing how much you miss someone",
    Emotions.SURPRISE: "Running into your childhood best friend unexpectedly at a park",
}

def get_emotion_stimulus(emotion):
    """Get stimulus for emotion, handling both enum and string inputs."""
    if isinstance(emotion, str):
        if emotion.lower() in ["neutral", "none"]:
            return ""
        # Try to convert string to enum
        try:
            emotion = Emotions.from_string(emotion)
        except ValueError:
            return ""
    
    return emotion2stimulus.get(emotion, "")