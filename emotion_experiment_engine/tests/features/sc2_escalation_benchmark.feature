Feature: SC2 escalation benchmark wiring
  In order to evaluate emotion effects on StarCraft II escalation decisions
  As a researcher
  I want a dedicated benchmark that loads the SC2 escalation dataset and uses the game prompt wrapper.

  Scenario: SC2 escalation benchmark exposes options via game-style prompts
    Given a benchmark config with name "sc2_escalation" and task_type "Escalation_Game"
    When I create benchmark components using the registry
    Then the dataset should load scenarios from the SC2 escalation dataset
    And the prompt wrapper should be a game benchmark prompt wrapper

