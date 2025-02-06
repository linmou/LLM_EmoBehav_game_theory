from games.stag_hunt import StagHuntScenario, StagHuntDecision
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision
from games.battle_of_sexes import BattleOfSexesScenario, BattleOfSexesDecision
from games.wait_go_game import WaitGoScenario, WaitGoDecision
from games.duopolistic_competition import DuopolisticCompetitionScenario, DuopolisticCompetitionDecision
from payoff_matrix import stag_hunt, prisoner_dilemma, battle_of_sexes, wait_go, duopolistic_competition

GAME_CONFIGS = {
    "Stag_Hunt": {
        "scenario_class": StagHuntScenario,
        "decision_class": StagHuntDecision,
        "payoff_matrix": stag_hunt,
    },
    "Prisoners_Dilemma": {
        "scenario_class": PrisonerDilemmaScenario,
        "decision_class": PrisionerDelimmaDecision,
        "payoff_matrix": prisoner_dilemma,
    },
    "Battle_Of_Sexes": {
        "scenario_class": BattleOfSexesScenario,
        "decision_class": BattleOfSexesDecision,
        "payoff_matrix": battle_of_sexes,
    },
    "Wait_Go": {
        "scenario_class": WaitGoScenario,
        "decision_class": WaitGoDecision,
        "payoff_matrix": wait_go,
    },
    "Duopolistic_Competition": {
        "scenario_class": DuopolisticCompetitionScenario,
        "decision_class": DuopolisticCompetitionDecision,
        "payoff_matrix": duopolistic_competition,
    }
}

def get_game_config(game_name: str) -> dict:
    """Get the configuration for a specific game.
    
    Args:
        game_name: Name of the game to get configuration for
        
    Returns:
        Dictionary containing game configuration
        
    Raises:
        ValueError: If game_name is not found in configurations
    """
    if game_name not in GAME_CONFIGS:
        available_games = ", ".join(GAME_CONFIGS.keys())
        raise ValueError(f"Unsupported game: {game_name}. Available games: {available_games}")
    
    return GAME_CONFIGS[game_name] 