Feature: Beauty contest diplomacy scenario generation
  The generator creates diplomacy-themed beauty contest scenarios using fewshot examples.

  Scenario: Generate a valid diplomacy beauty contest record
    Given a diplomacy fewshot file with example scenarios
    When the generator requests a new scenario
    Then the result includes scenario, description, participants, behavior_choices, game_category, and game_name
    And behavior_choices has commit_0 through commit_3
