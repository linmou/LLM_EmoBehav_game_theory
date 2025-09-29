# Feature file for emotion_experiment_engine.evaluate_saved_series CLI wrapper
Feature: Deferred evaluation for experiment series reports
  Background:
    Given a memory experiment series report referencing multiple run directories
    And some runs already contain evaluation summaries and completed README files
    And some runs still have deferred evaluation READMEs without summaries

  Scenario: Running evaluate_saved_series processes only unevaluated runs
    When I invoke evaluate_saved_series with --report and --dry-run on the series
    Then it lists the unevaluated run directories and skips the completed ones
    When I rerun evaluate_saved_series without --dry-run
    Then it executes evaluate_saved for each unevaluated run and rewrites their README files
    And it creates evaluation_summary.json artifacts alongside the existing summaries
