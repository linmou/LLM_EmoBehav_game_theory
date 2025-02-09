from pydantic import Field
import random
from typing import ClassVar, Dict, Any, Optional
from games.game import GameDecision, GameScenario, SequentialGameScenario, BehaviorChoices

class EGBehaviorChoices(BehaviorChoices):
    escalation: str
    withdraw: str

    def get_choices(self):
        return [self.escalation, self.withdraw]
    
    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()
    
    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"
    
    @staticmethod
    def example():
        return {
            "escalation": "<escalation action>",
            "withdraw": "<withdraw action>"
        }
    
class EscalationGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: EGBehaviorChoices
    previous_actions_length: int
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Escalation_Game"
    
    @property
    def previous_actions(self) -> list[str]:
        previous_actions = []
        for i in range(self.previous_actions_length):
            previous_actions.append((self.get_participant_names()[i % 2], self.behavior_choices.escalation)) # only one action
        return previous_actions
        
    def get_scenario_info(self) -> dict:
        return {
            "scenario": self.scenario,
            "description": self.description}
        
    def get_participants(self) -> list[dict]:
        return self.participants
    
    def get_payoff_matrix(self) -> Dict[str, Any]:
        return self.payoff_matrix
    
    def get_participant_names(self) -> list[str]:
        return [participant['name'] for participant in self.participants]
    
    def get_behavior_choices(self) -> EGBehaviorChoices:
        return self.behavior_choices
    
    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == 'escalation':
            return self.behavior_choices.escalation
        elif decision == 'withdraw':
            return self.behavior_choices.withdraw
        else:
            raise ValueError(f"Invalid decision: {decision}")
    
    @staticmethod
    def example():
        return {
            "scenario": "Rights Competition",
            "description": "You and Bob are business rivals negotiating with a city council for exclusive rights to build a new shopping complex. The city requires each party to offer a contribution to local development projects as part of their bid. The higher contributor will win the contract, while the loser still has to pay their total offer with no benefit. There are several turns to bid.",
            "participants": [
                {"name": "You", "role": "Participant"},
                {"name": "Bob", "role": "Participant"}
            ],
            "behavior_choices": {
                "escalation": "escalate the bid",
                "withdraw": "withdraw from the rival"
            },
            
        }
        
    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions}
        """
        
class EscalationGameDecision(GameDecision):
    scenario: ClassVar[Optional[EscalationGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, EscalationGameScenario):
            raise ValueError("Scenario must be a EscalationGameScenario")
        cls.scenario = scenario
        cls.model_fields['decision'].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError("Scenario must be set using Decision.set_scenario() before validating")
        return self.scenario.get_behavior_choices().is_valid_choice(decision)

        