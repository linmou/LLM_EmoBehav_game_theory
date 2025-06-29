import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class EndowmentEffectBehaviorChoices(BehaviorChoices):
    keep_item: str
    trade_item: str

    def get_choices(self):
        return [self.keep_item, self.trade_item]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "keep_item": "Keep the mug",
            "trade_item": "Trade the mug for the chocolate bar",
        }


class EndowmentEffectScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: EndowmentEffectBehaviorChoices
    game_name: str = "Endowment_Effect"
    
    # Override payoff_matrix to be optional since endowment effect doesn't need it
    payoff_matrix: Optional[Dict[str, Any]] = Field(default=None)

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.keep_item:
            return "keep_item"
        elif decision == self.behavior_choices.trade_item:
            return "trade_item"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> EndowmentEffectBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "The Coffee Mug Exchange: Testing Value Perception",
            "participants": [
                {"name": "You", "profile": "Participant in Psychology Experiment"},
            ],
            "behavior_choices": EndowmentEffectBehaviorChoices.example(),
            "description": "You are participating in a psychology experiment. The researcher has given you a coffee mug as a token of appreciation for your participation. Now, the researcher offers you a choice: you can keep the mug, or you can trade it for a chocolate bar of equivalent retail value ($5). Both items are new and of similar quality. The researcher explains that this is part of a study on decision-making and that there are no right or wrong answers. What do you choose to do?",
        }


class EndowmentEffectDecision(GameDecision):
    scenario: ClassVar[Optional[EndowmentEffectScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, EndowmentEffectScenario):
            raise ValueError("Scenario must be an EndowmentEffectScenario")
        cls.scenario = scenario
        cls.model_fields["decision"].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError(
                "Scenario must be set using Decision.set_scenario() before validating"
            )
        return self.scenario.get_behavior_choices().is_valid_choice(decision)


if __name__ == "__main__":
    # Example usage
    example_data = EndowmentEffectScenario.example()
    scenario = EndowmentEffectScenario(**example_data)
    
    print("=== Endowment Effect Scenario ===")
    print(f"Scenario: {scenario.scenario}")
    print(f"Description: {scenario.description}")
    print(f"Participants: {scenario.participants}")
    print(f"Behavior Choices: {scenario.behavior_choices}")
    
    # Test decision validation
    EndowmentEffectDecision.set_scenario(scenario)
    
    # Test valid decision
    decision = EndowmentEffectDecision(
        decision="Keep the mug",
        rational="I value the mug more now that I own it"
    )
    print(f"\nValid decision: {decision.decision}")
    print(f"Rationale: {decision.rational}")
    
    # Test behavior mapping
    behavior = scenario.find_behavior_from_decision(decision.decision)
    print(f"Mapped behavior: {behavior}")