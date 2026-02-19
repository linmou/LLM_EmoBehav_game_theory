Feature: Emotion scale steering on subjective sentences
  The system should detect whether steering vectors shift response emotion
  on subjective sentence prompts.

  Scenario: Build emotion_scale dataset from subjective sentence prompts
    Given a JSONL file of subjective sentence prompts
    When benchmark "emotion_check" runs with task_type "emotion_scale"
    Then each prompt is loaded as a free-text item
    And each item category is "emotion_scale"

  Scenario: Score responses with Gemini seven-class emotion judge
    Given an emotion_scale run configured with client "gemini" and model "gemini-2.5-flash"
    When a model response is evaluated
    Then the classifier must output one of:
      | anger     |
      | happiness |
      | sadness   |
      | fear      |
      | disgust   |
      | surprise  |
      | neutral   |
    And score should compare predicted emotion with the activated steering emotion
