# File: docs/testing/behavior_shift_alignment.feature
# Purpose: Integrated behavior contract for result_analysis/behavior_shift_alignment.py.

Feature: Behaviour shift alignment from significance tables
  As a researcher
  I want a metric that turns a significance table into behaviour-shift alignment
  So that I can compare model emotion shifts against literature-grounded human directions

  Scenario: Significant direction matches, mismatches, and neutral expectations are aggregated
    Given a significance table with task, behavior, emotion, delta, and significance
    And the literature expects Prisoners_Dilemma cooperate to decrease under anger
    And the literature expects Prisoners_Dilemma cooperate to stay neutral under fear
    When I compute behaviour shift alignment
    Then a significant negative anger delta is treated as aligned
    And a significant positive fear delta is treated as misaligned
    And a non-significant fear delta is treated as aligned with a neutral expectation
    And the module returns both raw alignment in [-1, 1] and normalized alignment in [0, 1]

  Scenario: Tasks with only non-significant rows are marked NotSig
    Given a significance table where every covered row for a task is non-significant
    When I compute behaviour shift alignment
    Then that task summary is marked as NotSig

  Scenario: Role-specific games use canonical focal behaviors
    Given the default alignment specs
    When I inspect Trust_Game_Trustee and Ultimatum_Game_Proposer
    Then their focal behaviors are return_high and offer_high
    And those tasks are accepted by spec validation
