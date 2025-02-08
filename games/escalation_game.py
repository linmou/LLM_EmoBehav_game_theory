from typing import Dict, Any
from games.game import SequentialGameScenario, BehaviorChoices

class EGBehaviorChoices(BehaviorChoices):
    option_a: str
    option_b: str

    def get_choices(self):
        return [self.option_a, self.option_b]
    
    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()
    
    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"
    
    @staticmethod
    def example():
        return {
            "option_a": "Option A",
            "option_b": "Option B"
        }
    
class EscalationGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    previous_actions: list[str]
    payoff_matrix: Dict[str, Any]
    game_name: str = "Escalation_Game"
    
    def get_previous_actions(self) -> list[str]:
        return self.previous_actions
    
    def get_scenario_info(self) -> dict:
        return {
            "scenario": self.scenario,
            "description": self.description}
        
    def get_participants(self) -> list[dict]:
        return self.participants
    
    def get_payoff_matrix(self) -> Dict[str, Any]:
        return self.payoff_matrix
    
    