from games.stag_hunt import StagHuntScenario, StagHuntDecision
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision
from payoff_matrix import stag_hunt, prisoner_dilemma

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