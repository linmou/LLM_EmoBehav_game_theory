from typing import ClassVar, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class SealedAuctionBehaviorChoices(BehaviorChoices):
    devote_none: str
    devote_low: str
    devote_high: str

    def get_choices(self) -> list[str]:
        return [self.devote_none, self.devote_low, self.devote_high]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    @staticmethod
    def example() -> dict:
        return {
            "devote_none": "Keep scouting and position defensively with minimal extra commitment.",
            "devote_low": "Invest moderately: one timing setup and limited units with some risk.",
            "devote_high": "Commit heavily: multiple production cycles and major resource allocation.",
        }


class SealedAuctionScenario(GameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: SealedAuctionBehaviorChoices
    game_name: str = "Sealed_Auction"

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> SealedAuctionBehaviorChoices:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.devote_none:
            return "devote_none"
        if decision == self.behavior_choices.devote_low:
            return "devote_low"
        if decision == self.behavior_choices.devote_high:
            return "devote_high"
        raise ValueError(f"Invalid decision: {decision}")

    @staticmethod
    def example() -> dict:
        return {
            "scenario": "Crimson_Relay_Air_Drop_Bid",
            "description": (
                "You, the Protoss commander, are playing on **Crimson Relay** with long air lanes. "
                "Both sides secretly commit resources; higher commitment gains initiative."
            ),
            "participants": [{"name": "Protoss"}, {"name": "Terran"}],
            "behavior_choices": SealedAuctionBehaviorChoices.example(),
        }


class SealedAuctionDecision(GameDecision):
    scenario: ClassVar[Optional[SealedAuctionScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, SealedAuctionScenario):
            raise ValueError("Scenario must be a SealedAuctionScenario")
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

