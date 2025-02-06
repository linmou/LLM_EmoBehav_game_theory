from dataclasses import dataclass
import json
from pathlib import Path
import instructor
from openai import OpenAI
from typing import List, Tuple, Type, Union

from games.game import Game, GameScenario, GameDecision
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision
from games.stag_hunt import StagHuntScenario, StagHuntDecision
from payoff_matrix import prisoner_dilemma_very_large, stag_hunt
from stimulis.emotions import emotion2stimulus

@dataclass
class TestResult:
    scenario: GameScenario
    decision: GameDecision
    decision_category: str

    def to_dict(self):
        """Convert TestResult to a dictionary for JSON serialization."""
        return {
            "scenario": self.scenario.scenario,
            "scenario_description": self.scenario.description,
            "decision": self.decision.decision,
            "rationale": self.decision.rational,
            "category": self.decision_category
        }

class GameTheoryTest:
    def __init__(self, model: str = "gpt-4o", base_url: str = None, api_key: str = None):
        """Initialize the game theory test engine with OpenAI client."""
        client = OpenAI(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
        )
        self.client = instructor.patch(client)
        self.model = model

    def load_scenarios(self, game: Game) -> List[Tuple[GameScenario, Type[GameDecision]]]:
        """Load scenarios from JSON files in the specified folder."""
        scenario_decision_pairs = []
        folder = Path(game.folder_path)
        
        if not folder.exists():
            raise ValueError(f"Folder {game.folder_path} does not exist")
            
        for file in folder.glob("*.json"):
            print(f"Loading scenario from {file.name}")
            with open(file, "r") as f:
                data = json.load(f)
                data['payoff_matrix'] = game.payoff_matrix
                scenario = game.create_scenario(data)
                scenario_decision_pairs.append((scenario, game.decision_class))
                
        return scenario_decision_pairs

    def test_scenario(self, 
                     scenario: GameScenario,
                     decision_class: Type[GameDecision],
                     system_message: str = None,
                     temperature: float = 0.7) -> GameDecision:
        """Test a single scenario with the AI model."""
        default_system_message = (
            "You are Alice, an average American."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_message or default_system_message
            },
            {
                "role": "user",
                "content": f"Please analyze the following scenario and make a decision:\n{scenario}. Choice should be one of {scenario.get_behavior_choices().get_choices()}"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_model=decision_class
            )
            # Validate the decision
            category = scenario.find_behavior_from_decision(response.decision)
            if category is None:
                raise ValueError(f"Invalid decision: {response.decision}")
            
            return TestResult(scenario=scenario, decision=response, decision_category=category)
            
        except Exception as e:
            print(f"Error processing scenario: {str(e)}")
            raise



def run_tests(game: Game, config: dict, output_dir: str, emotion: str, system_message: str = "You are Alice, an average American."):
    """Example usage of the GameTheoryTest engine."""
    
    engine = GameTheoryTest(**config)
    results = []

    # Load and test scenarios for each game
    print(f"\nTesting {game.name} scenarios:")
    scenarios = engine.load_scenarios(game)
    
    for scenario, decision_class in scenarios:
        try:
            result = engine.test_scenario(
                scenario,
                decision_class,
                system_message=system_message
            )
            results.append(result)
            print(f"\nScenario: {scenario.scenario}")
            print(f"Emotion: {emotion}")
            print(f"Category: {result.decision_category}")
            print(f"Decision: {result.decision.decision}")
            print(f"Rationale: {result.decision.rational}")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to process scenario {scenario.scenario}: {str(e)}")
    
    # Save results to JSON file
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{game.name}_{emotion}_results.json"
    with open(output_file, 'w') as f:
        json.dump([{**r.to_dict(), "emotion": emotion} for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    from api_configs import GPT4O_CONFIG
    game_name = "Stag_Hunt"
    sh_game = Game(
        name=game_name,
        scenario_class=StagHuntScenario,
        decision_class=StagHuntDecision,
        payoff_matrix=stag_hunt
    )
    emotion = "sadness"
    stimulus = emotion2stimulus[emotion]
    system_message = f"You are Alice, an average American. You feel {emotion} because {stimulus}. Hold this emotion and make a decision."
    run_tests(sh_game, config=GPT4O_CONFIG, output_dir=f"results/{game_name}", emotion=emotion, system_message=system_message)