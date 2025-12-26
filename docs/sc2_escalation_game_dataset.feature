Feature: StarCraft II escalation game dataset
  In order to study emotional reactions to escalation decisions in StarCraft II
  As a researcher
  I want a structured dataset of StarCraft II escalation scenarios with clear escalate vs withdraw options.

  Scenario: Dataset covers core StarCraft II races and escalation choices
    Given the SC2 escalation dataset at "data/sc2/escalation_game.json"
    When I load all scenarios
    Then there should be at least 10 scenarios
    And the "you_play_as" field should include Protoss, Terran, and Zerg at least once
    And each scenario should define escalate and withdraw behaviour decisions for the current player

