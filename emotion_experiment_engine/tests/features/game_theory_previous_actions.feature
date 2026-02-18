# Feature file describing expectations for game theory dry run behavior.
Feature: Game theory dry run output quality
  Background:
    Given a configured memory experiment series for the Trust Game trustee benchmark

  Scenario: Previous actions are exposed in constructed prompts
    When the dry run builds the first Trust Game trustee scenario
    Then the formatted scenario text should include previously observed trustor actions

  Scenario: Dry run fails fast when a benchmark configuration raises an error
    When the dry run encounters a benchmark configuration error
    Then the dry run command should exit with a failure instead of reporting success

  Scenario: Dry run does not fail on unresolved benchmark data path used only for logging
    Given a dry-run setup result with benchmark data path unresolved
    When dry run prints benchmark diagnostics
    Then dry run should continue without treating diagnostic path formatting as a config failure

  Scenario: Ultimatum responder scenarios can load without per-config previous action overrides
    Given ultimatum responder raw scenarios without previous action fields
    When game-theory dataset applies default game configuration
    Then at least one responder scenario should be parsed with previous actions populated
