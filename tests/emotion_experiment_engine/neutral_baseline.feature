Feature: Neutral-only memory experiment
  Scenario: Run memory experiment with empty emotions list
    Given a configuration file sets emotions to an empty list
    When the experiment series runner instantiates an EmotionExperiment
    Then the experiment should skip loading emotion activation readers
    And the experiment should still execute the neutral baseline sweep successfully
