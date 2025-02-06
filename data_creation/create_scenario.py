import autogen
from autogen.agentchat.contrib.captainagent import CaptainAgent
from autogen import UserProxyAgent

from games.game import Game
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision
from games.stag_hunt import StagHuntScenario, StagHuntDecision
from payoff_matrix import prisoner_dilemma_very_large, stag_hunt

AGENT_LIB = f'groupchat/agent_groups/Prisoners_Dilemma/build_history_fbfa3f35e995ccae918f6ef01482a7fb.json'

class ScenarioGenerator:
    def __init__(self, game: Game, participants: list[str], config_path: str, model_ls=["gpt-4o"]):
        self.game = game
        self.participants = participants
        self.config_path = config_path
        self.config_list = autogen.config_list_from_json(
            self.config_path, filter_dict={"model": model_ls}
        )
        
        # Initialize agents
        self.captain_agent = CaptainAgent(
            name="captain_agent",
            llm_config={
                'config_list': self.config_list,
                'temperature': 0.5,
                'cache_seed': None,
            },
            agent_lib=AGENT_LIB,
            code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        )
        self.captain_user_proxy = UserProxyAgent(name="captain_user_proxy", human_input_mode="NEVER")

    def generate_scenarios(self, ttl_number: int, batch_size: int):
        iteration = ttl_number // batch_size
        for i in range(iteration):
            result = self.captain_user_proxy.initiate_chat(
                self.captain_agent,
                message=f"""unique scenario that masks the {self.game.name} structure, ensure it under a new context that participants won't immediately recognize. 
                Return both the scenario , the profile of the participants' names are {self.participants}, behavior choices
                the profile should not contain anything will impact the decision, like personality, preferences, etc. only the objective description.
                
                * Create a team with at least one agent inside as the participant with think-aloud process, so that you can ensure the participant does not figure out the scenario is a {self.game.name} at the first glance.
                
                Here is the example of the scenario, strictly follow the keys of the dict:
                {self.game.example_scenario}
                
                Generate {batch_size} scenarios, then save the result into a json file 'scenarios/{self.game.name}/{{scenario}}.json' respectively.
                """,
                max_turns=1,
            )
        return result


if __name__ == "__main__":
    # Example for Stag Hunt game
    participants = ["You", "Bob"]
    from games.game_configs import get_game_config, GAME_CONFIGS
    from pathlib import Path
    total_num = 36
    for game_name in GAME_CONFIGS.keys():
        game_cfg = get_game_config(game_name)
        game = Game(name=game_name, **game_cfg)
        folder_path = Path(f'groupchat/scenarios/{game_name}')
        folder_path.mkdir(parents=True, exist_ok=True)
        # Count existing scenarios for this game
        existing_scenarios = len(list(folder_path.glob('*.json')))
        to_gen_num = total_num - existing_scenarios  # Target 36 total scenarios
        
        if to_gen_num <= 0:
            print(f"Skipping {game_name} - already has {existing_scenarios} scenarios")
            continue
        print(f"Generating {to_gen_num} scenarios for {game_name}")
        
        generator = ScenarioGenerator(game, participants, "config/OAI_CONFIG_LIST")
        generator.generate_scenarios(ttl_number=to_gen_num, batch_size=4)

    # # Example for Prisoner's Dilemma game
    # pd_game = Game(
    #     name="Prisoners_Dilemma",
    #     scenario_class=PrisonerDilemmaScenario,
    #     decision_class=PrisionerDelimmaDecision,
    #     payoff_matrix=prisoner_dilemma_very_large
    # )
    
    # participants = ["Alice", "Bob"]
    # generator = ScenarioGenerator(pd_game, participants, "config/OAI_CONFIG_LIST")
    # generator.generate_scenarios(12, 4)  

