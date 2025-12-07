################################################################################
# File: docs/testing/diplomacy_pd_escalation.feature
# Purpose: Gherkin outline for loading the new escalation Diplomacy dataset
# Last updated: 2025-11-17 (commit 2339280756728260d0e053b6de9ea8d4045597d1)
################################################################################

Feature: Load Diplomacy escalation dataset with gradient options
  Scenario: Build benchmark items from the 2025-11-17 escalation drop
    Given the merged jsonl at data/diplomacy/diplomacy_pd_escalation_20251117.jsonl
    When the DiplomacyGradientDataset reads the first record
    Then it should use the description text as the prompt question
    And include gradient_options as the option list
    And surface the whose_option label in metadata and context
