import autogen
from pathlib import Path
from autogen.agentchat.contrib.captainagent import CaptainAgent
from autogen import UserProxyAgent

from games.game import Game
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision
from games.stag_hunt import StagHuntScenario, StagHuntDecision
from payoff_matrix import prisoner_dilemma_very_large, stag_hunt
from games.game_configs import get_game_config, GAME_CONFIGS

AGENT_LIB = f'groupchat/agent_groups/Prisoners_Dilemma/build_history_fbfa3f35e995ccae918f6ef01482a7fb.json'

class ScenarioGenerator:
    def __init__(self, game: Game, participants: list[str], config_path: str, model_ls=["gpt-4o"], load_agent_lib=True):
        self.game = game
        self.participants = participants
        self.config_path = config_path
        self.config_list = autogen.config_list_from_json(
            self.config_path, filter_dict={"model": model_ls}
        )
        self.work_dir = "groupchat"
        # Initialize agents
        self.captain_agent = CaptainAgent(
            name="captain_agent",
            llm_config={
                'config_list': self.config_list,
                'temperature': 0.9,
                'cache_seed': None,
            },
            agent_lib=AGENT_LIB if load_agent_lib else None,
            code_execution_config={"use_docker": False, "work_dir": self.work_dir},
            agent_config_save_path=f"{self.work_dir}/agent_groups/{self.game.name}"
            )
        self.captain_user_proxy = UserProxyAgent(name="captain_user_proxy", human_input_mode="NEVER", code_execution_config={"use_docker": False, "work_dir": self.work_dir})
        
    def generate_simultaneous_scenarios(self, ttl_number: int, batch_size: int):
        iteration = ttl_number // batch_size

        for i in range(iteration):
            existing_scenarios = list(Path(f"{self.work_dir}/scenarios/{self.game.name}").glob('*.json'))
            result = self.captain_user_proxy.initiate_chat(
                self.captain_agent,
                message=f"""Make an unique scenario that masks the {self.game.name} structure, ensure it under a new context that participants won't immediately recognize. 
                Return both the scenario , the profile of the participants {self.participants}, behavior choices
                the profile should not contain anything will impact the decision, like personality, preferences, etc. only the objective description.
                
                * Create a team with at least one agent inside as the participant with think-aloud process, so that you can ensure the participant does not figure out the scenario is a {self.game.name} at the first glance.
                
                Here is the example of the scenario, strictly follow the keys of the dict:
                {self.game.example_scenario}
                
                Here is the existing scenarios, avoid duplication or similarity:
                {existing_scenarios}
                
                Generate {batch_size} scenarios, then save the result into a json file 'scenarios/{self.game.name}/{{scenario}}.json' respectively.
                """,
                max_turns=1,
            )
        return result
    
    def generate_sequential_scenarios(self, ttl_number: int, batch_size: int):
        iteration = ttl_number // batch_size
        for i in range(iteration):
            existing_scenarios = list(Path(f"{self.work_dir}/scenarios/{self.game.name}").glob('*.json'))
            result = self.captain_user_proxy.initiate_chat(
                self.captain_agent,
                message=f"""I want to test participants' reaction in a {self.game.name}. It is a sequential game. Make an unique scenario that masks the {self.game.name} structure, ensure it under a new context that participants won't immediately recognize. 
                Given the names of participants {self.participants}, 
                return both the scenario , the profile of the participants, behavior choices.
                the profile and scenario description should not contain anything will impact the decision, like personality, preferences, etc. only the objective description.
                
                * Create a team with at least one agent inside as the participant with think-aloud process, so that you can ensure the participant does not figure out the scenario is a {self.game.name} at the first glance.
                
                Here is the example of the scenario, strictly follow the keys of the dict:
                {self.game.example_scenario}
                
                Here is the existing scenarios, avoid duplication or similarity:
                {existing_scenarios}
                
                Generate {batch_size} scenarios, then save the result into a json file 'scenarios/{self.game.name}/{{scenario}}.json' respectively.
                """,
                max_turns=1,
            )
        # return result

def generate_simultaneous_dataset( ttl_number: int, batch_size: int=4):
    participants = ["You", "Bob"]
    
    for game_name in GAME_CONFIGS.keys():
        game_cfg = get_game_config(game_name)
        game = Game(name=game_name, 
                    scenario_class=game_cfg['scenario_class'],
                    decision_class=game_cfg['decision_class'],
                    payoff_matrix=game_cfg['payoff_matrix']
                    )
        folder_path = Path(f'groupchat/scenarios/{game_name}')
        folder_path.mkdir(parents=True, exist_ok=True)
        # Count existing scenarios for this game
        existing_scenarios = len(list(folder_path.glob('*.json')))
        to_gen_num = ttl_number - existing_scenarios
        
        if to_gen_num <= 0:
            print(f"Skipping {game_name} - already has {existing_scenarios} scenarios")
            continue
        print(f"Generating {to_gen_num} scenarios for {game_name}")
        
        generator = ScenarioGenerator(game, participants, "config/OAI_CONFIG_LIST", load_agent_lib=True)
        generator.generate_simultaneous_scenarios(ttl_number=to_gen_num, batch_size=batch_size)
        
def generate_sequential_dataset( ttl_number: int, batch_size: int=4):
    participants = ["You", "Bob"]
    
    game_name = "Trust_Game"
    game_cfg = get_game_config(game_name)
    game = Game(name=game_name, 
                scenario_class=game_cfg['scenario_class'],
                decision_class=game_cfg['decision_class'],
                payoff_matrix=game_cfg['payoff_matrix']
                )
    folder_path = Path(f'groupchat/scenarios/{game_name}')
    folder_path.mkdir(parents=True, exist_ok=True)
    # Count existing scenarios for this game
    existing_scenarios = len(list(folder_path.glob('*.json')))
    to_gen_num = ttl_number - existing_scenarios
    
    if to_gen_num <= 0:
        print(f"Skipping {game_name} - already has {existing_scenarios} scenarios")
        return
    print(f"Generating {to_gen_num} scenarios for {game_name}")
    
    generator = ScenarioGenerator(game, participants, "config/OAI_CONFIG_LIST", load_agent_lib=False)
    generator.generate_sequential_scenarios(ttl_number=to_gen_num, batch_size=batch_size)

if __name__ == "__main__":

    generate_sequential_dataset(ttl_number=24, batch_size=3)

