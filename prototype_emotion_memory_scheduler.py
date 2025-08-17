"""
Prototype: Emotion Scheduling for Long-Context Memory Tasks

This prototype validates the concept of temporal emotion application 
throughout long contexts, addressing the key challenge that memory tasks
require persistent emotional states unlike single-decision game theory.
"""
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
from dataclasses import dataclass

class EmotionScheduleType(Enum):
    """Different patterns for applying emotions throughout long contexts"""
    CONTINUOUS = "continuous"           # Constant emotion throughout
    DECAY = "decay"                    # Emotion diminishes over time  
    PERIODIC = "periodic"              # Emotion applied at intervals
    CONTEXT_TRIGGERED = "context_triggered"  # Emotion based on content
    CRITICAL_MOMENTS = "critical_moments"    # Emotion at key decision points

@dataclass
class EmotionScheduleConfig:
    """Configuration for emotion application patterns"""
    schedule_type: EmotionScheduleType
    emotion: str  # From constants.Emotions
    intensity: float = 1.0
    # Type-specific parameters
    decay_rate: Optional[float] = None      # For DECAY
    period_length: Optional[int] = None     # For PERIODIC  
    trigger_keywords: Optional[List[str]] = None  # For CONTEXT_TRIGGERED
    critical_positions: Optional[List[float]] = None  # For CRITICAL_MOMENTS (as % of context)

class MemoryEmotionScheduler:
    """
    Manages temporal application of emotions during long-context processing.
    
    Key insight: Unlike game theory (single decision), memory tasks need
    emotions applied strategically throughout the entire context window.
    """
    
    def __init__(self, config: EmotionScheduleConfig):
        self.config = config
        self.context_length = 0
        self.current_position = 0
        
    def initialize_context(self, context_length: int, context_text: str = ""):
        """Initialize for a new context"""
        self.context_length = context_length
        self.current_position = 0
        self.context_text = context_text
        
        # Pre-compute schedule for efficiency
        self.emotion_schedule = self._compute_schedule()
        
    def _compute_schedule(self) -> List[Tuple[int, float]]:
        """Compute emotion intensity at each position"""
        schedule = []
        
        if self.config.schedule_type == EmotionScheduleType.CONTINUOUS:
            # Constant emotion throughout
            for pos in range(self.context_length):
                schedule.append((pos, self.config.intensity))
                
        elif self.config.schedule_type == EmotionScheduleType.DECAY:
            # Exponentially decaying emotion
            decay_rate = self.config.decay_rate or 0.001
            for pos in range(self.context_length):
                intensity = self.config.intensity * np.exp(-decay_rate * pos)
                schedule.append((pos, intensity))
                
        elif self.config.schedule_type == EmotionScheduleType.PERIODIC:
            # Periodic emotion application
            period = self.config.period_length or 100
            for pos in range(self.context_length):
                if pos % period < period // 2:  # Active for first half of period
                    schedule.append((pos, self.config.intensity))
                else:
                    schedule.append((pos, 0.0))
                    
        elif self.config.schedule_type == EmotionScheduleType.CRITICAL_MOMENTS:
            # Emotion only at specific context positions
            critical_positions = self.config.critical_positions or [0.1, 0.5, 0.9]
            schedule = [(pos, 0.0) for pos in range(self.context_length)]
            
            for crit_pos in critical_positions:
                abs_pos = int(crit_pos * self.context_length)
                if abs_pos < self.context_length:
                    # Apply emotion in window around critical position
                    window = 50  # tokens
                    for pos in range(max(0, abs_pos - window), 
                                   min(self.context_length, abs_pos + window)):
                        schedule[pos] = (pos, self.config.intensity)
                        
        return schedule
    
    def get_emotion_intensity(self, position: int) -> float:
        """Get emotion intensity at specific position"""
        if position >= len(self.emotion_schedule):
            return 0.0
        return self.emotion_schedule[position][1]
    
    def advance_position(self, new_position: int):
        """Update current position in context"""
        self.current_position = new_position
        
    def get_current_intensity(self) -> float:
        """Get emotion intensity at current position"""
        return self.get_emotion_intensity(self.current_position)

# Validation tests
def test_emotion_schedulers():
    """Test different emotion scheduling patterns"""
    
    context_length = 1000
    
    # Test continuous emotion
    continuous_config = EmotionScheduleConfig(
        schedule_type=EmotionScheduleType.CONTINUOUS,
        emotion="anger",
        intensity=0.8
    )
    
    scheduler = MemoryEmotionScheduler(continuous_config)
    scheduler.initialize_context(context_length)
    
    # Verify continuous application
    assert scheduler.get_emotion_intensity(0) == 0.8
    assert scheduler.get_emotion_intensity(500) == 0.8
    assert scheduler.get_emotion_intensity(999) == 0.8
    
    # Test decay emotion
    decay_config = EmotionScheduleConfig(
        schedule_type=EmotionScheduleType.DECAY,
        emotion="sadness", 
        intensity=1.0,
        decay_rate=0.001
    )
    
    decay_scheduler = MemoryEmotionScheduler(decay_config)
    decay_scheduler.initialize_context(context_length)
    
    # Verify decay pattern
    assert decay_scheduler.get_emotion_intensity(0) == 1.0
    assert decay_scheduler.get_emotion_intensity(500) < 0.7  # Should have decayed
    assert decay_scheduler.get_emotion_intensity(999) < 0.4  # Should have decayed more
    
    # Test critical moments
    critical_config = EmotionScheduleConfig(
        schedule_type=EmotionScheduleType.CRITICAL_MOMENTS,
        emotion="fear",
        intensity=1.0,
        critical_positions=[0.1, 0.5, 0.9]  # At 10%, 50%, 90% of context
    )
    
    critical_scheduler = MemoryEmotionScheduler(critical_config)
    critical_scheduler.initialize_context(context_length)
    
    # Verify critical moments have high intensity
    assert critical_scheduler.get_emotion_intensity(100) > 0.5  # Around 10%
    assert critical_scheduler.get_emotion_intensity(500) > 0.5  # Around 50%
    assert critical_scheduler.get_emotion_intensity(900) > 0.5  # Around 90%
    
    # Verify non-critical moments have low intensity
    assert critical_scheduler.get_emotion_intensity(300) == 0.0  # Between critical points
    
    print("✓ All emotion scheduler tests passed!")

if __name__ == "__main__":
    test_emotion_schedulers()
    
    # Demonstrate different patterns
    context_length = 200
    
    configs = [
        ("Continuous", EmotionScheduleType.CONTINUOUS, {}),
        ("Decay", EmotionScheduleType.DECAY, {"decay_rate": 0.01}),
        ("Periodic", EmotionScheduleType.PERIODIC, {"period_length": 50}),
        ("Critical", EmotionScheduleType.CRITICAL_MOMENTS, {"critical_positions": [0.2, 0.8]})
    ]
    
    print("\nEmotion Intensity Patterns:")
    print("Position", end="\t")
    for name, _, _ in configs:
        print(f"{name:>10}", end="\t")
    print()
    
    # Sample positions to show patterns
    sample_positions = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    
    schedulers = []
    for name, schedule_type, params in configs:
        config = EmotionScheduleConfig(
            schedule_type=schedule_type,
            emotion="anger",
            intensity=1.0,
            **params
        )
        scheduler = MemoryEmotionScheduler(config)
        scheduler.initialize_context(context_length)
        schedulers.append(scheduler)
    
    for pos in sample_positions:
        print(f"{pos:>8}", end="\t")
        for scheduler in schedulers:
            intensity = scheduler.get_emotion_intensity(pos)
            print(f"{intensity:>10.2f}", end="\t")
        print()