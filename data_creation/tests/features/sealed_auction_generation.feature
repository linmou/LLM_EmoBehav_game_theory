Feature: sealed auction scenario generation
  The generator creates sealed auction scenarios using fewshot examples.

  Scenario: Generate a valid sealed auction record
    Given a sealed auction fewshot file with example scenarios
    When the generator requests a new scenario
    Then the result includes scenario, description, participants, behavior_choices, game_category, and game_name
    And behavior_choices has devote_low, devote_medium, and devote_high
