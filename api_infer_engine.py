from dataclasses import dataclass
import json
from pathlib import Path
import instructor
from openai import OpenAI
from typing import List, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import Emotions, GameNames
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
    def __init__(self, llm_config: dict, generation_config: dict):
        """Initialize the game theory test engine with OpenAI client."""
        client = OpenAI(
           **llm_config
        )
        self.client = instructor.patch(client)
        self.generation_config = generation_config

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
                     system_message: str = None) -> GameDecision:
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
                messages=messages,
                response_model=decision_class,
                **self.generation_config
            )
        
            # Validate the decision
            category = scenario.find_behavior_from_decision(response.decision)
            if category is None:
                raise ValueError(f"Invalid decision: {response.decision}")
            
            return TestResult(scenario=scenario, decision=response, decision_category=category)
            
        except Exception as e:
            print(f"Error processing scenario: {str(e)}")
            raise

def process_scenario(engine, scenario, decision_class, emotion, system_message):
    """Helper function to process a single scenario."""
    try:
        result = engine.test_scenario(
            scenario,
            decision_class,
            system_message=system_message
        )
        print(f"\nScenario: {scenario.scenario}")
        print(f"Emotion: {emotion}")
        print(f"Category: {result.decision_category}")
        print(f"Decision: {result.decision.decision}")
        print(f"Rationale: {result.decision.rational}")
        print("-" * 50)
        return result
    except Exception as e:
        print(f"Failed to process scenario {scenario.scenario}: {str(e)}")
        return None

def run_tests(game: Game, llm_config: dict, generation_config: dict, output_dir: str, emotion: str, intensity: str, system_message: str = "You are Alice, an average American.", max_workers: int = 8, repeat: int = 1):
    """Example usage of the GameTheoryTest engine with parallel processing."""
    
    engine = GameTheoryTest(llm_config, generation_config)
    all_results = []

    # Load scenarios
    print(f"\nTesting {game.name} scenarios:")
    scenarios = engine.load_scenarios(game)
    
    # Run scenarios repeat times
    for repeat_num in range(repeat):
        print(f"\nRunning iteration {repeat_num + 1}/{repeat}")
        # Process scenarios in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scenario = {
                executor.submit(
                    process_scenario, 
                    engine, 
                    scenario, 
                    decision_class, 
                    emotion, 
                    system_message
                ): (scenario, decision_class, repeat_num) 
                for scenario, decision_class in scenarios
            }
            
            for future in as_completed(future_to_scenario):
                result = future.result()
                if result is not None:
                    all_results.append((result, repeat_num))
    
    # Save results to JSON file
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{game.name}_{emotion}_{intensity}_results.json"
    with open(output_file, 'w') as f:
        json.dump([{**r.to_dict(), "emotion": emotion, "intensity": intensity, "repeat_num": rep} for r, rep in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")
    return output_file


if __name__ == "__main__":
    from api_configs import OAI_CONFIG
    game_name = "Stag_Hunt"
    sh_game = Game(
        name=game_name,
        scenario_class=StagHuntScenario,
        decision_class=StagHuntDecision,
        payoff_matrix=stag_hunt
    )
    emotion = 'sad'
    stimulus = emotion2stimulus[Emotions.from_string(emotion)]
    system_message = f"You are Alice, an average American. You feel {emotion} because {stimulus}. Hold this emotion and make a decision."
    run_tests(sh_game, llm_config=OAI_CONFIG, generation_config={"model": "gpt-4o-mini", "temperature": 0.7, "seed": 43}, output_dir=f"results/{game_name}", emotion=emotion, system_message=system_message)