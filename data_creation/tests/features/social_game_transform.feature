Feature: Social game transform pipeline
  Purpose: Validate the transform CLI from curated source rows to loadable Beauty Contest and Escalation Game cases.

  Scenario: Transform valid beauty_contest rows into a success-only dataset
    Given a curated beauty_contest input file with valid rows
    And a mapped few-shot asset and shared rubric
    When the transform CLI runs
    Then it writes a success dataset with only loadable rows
    And it writes separate failure and metadata artifacts

  Scenario: Transform valid escalation_game rows into a success-only dataset
    Given a curated escalation_game input file with valid rows
    And a mapped few-shot asset and shared rubric
    When the transform CLI runs
    Then it writes an escalation success dataset with only loadable rows
    And it writes separate escalation failure and metadata artifacts

  Scenario: Reject unsupported social games loudly
    Given a curated input file
    When the transform CLI runs with an unsupported social game
    Then the command exits with an error

  Scenario: Resume without duplicating completed identities
    Given a prior run with completed success and failure artifacts
    When the transform CLI runs again without rerun mode
    Then completed identities are skipped
    And the final outputs do not duplicate successful rows
