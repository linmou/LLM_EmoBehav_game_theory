from typing import Union

from constants import GameNames
from games.battle_of_sexes import BattleOfSexesDecision, BattleOfSexesScenario
from games.duopolistic_competition import (
    DuopolisticCompetitionDecision,
    DuopolisticCompetitionScenario,
)
from games.escalation_game import EscalationGameDecision, EscalationGameScenario
from games.payoff_matrices import ALL_GAME_PAYOFF
from games.prisoner_delimma import PrisionerDelimmaDecision, PrisonerDilemmaScenario
from games.stag_hunt import StagHuntDecision, StagHuntScenario
from games.trust_game import (
    TrustGameDecision,
    TrustGameTrusteeScenario,
    TrustGameTrustorScenario,
)
from games.ultimatum_game import (
    UltimatumGameDecision,
    UltimatumGameProposerScenario,
    UltimatumGameResponderScenario,
)
from games.wait_go_game import WaitGoDecision, WaitGoScenario

data_path_format = "groupchat/scenarios/{}_all_data_samples.json"  # data path is json containing all data samples
data_folder_format = "groupchat/scenarios/{}"  # data folder is the folder containing the seperated data samples

GAME_CONFIGS = {
    GameNames.STAG_HUNT: {
        "game_name": GameNames.STAG_HUNT.value,
        "scenario_class": StagHuntScenario,
        "decision_class": StagHuntDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.STAG_HUNT],
        "data_path": data_path_format.format(GameNames.STAG_HUNT.value),
    },
    GameNames.PRISONERS_DILEMMA: {
        "game_name": GameNames.PRISONERS_DILEMMA.value,
        "scenario_class": PrisonerDilemmaScenario,
        "decision_class": PrisionerDelimmaDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.PRISONERS_DILEMMA],
        "data_path": data_path_format.format(GameNames.PRISONERS_DILEMMA.value),
    },
    GameNames.BATTLE_OF_SEXES: {
        "game_name": GameNames.BATTLE_OF_SEXES.value,
        "scenario_class": BattleOfSexesScenario,
        "decision_class": BattleOfSexesDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.BATTLE_OF_SEXES],
        "data_path": data_path_format.format(GameNames.BATTLE_OF_SEXES.value),
    },
    GameNames.WAIT_GO: {
        "game_name": GameNames.WAIT_GO.value,
        "scenario_class": WaitGoScenario,
        "decision_class": WaitGoDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.WAIT_GO],
        "data_path": data_path_format.format(GameNames.WAIT_GO.value),
    },
    GameNames.DUOPOLISTIC_COMPETITION: {
        "game_name": GameNames.DUOPOLISTIC_COMPETITION.value,
        "scenario_class": DuopolisticCompetitionScenario,
        "decision_class": DuopolisticCompetitionDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.DUOPOLISTIC_COMPETITION],
        "data_path": data_path_format.format(GameNames.DUOPOLISTIC_COMPETITION.value),
    },
    GameNames.ESCALATION_GAME: {
        "game_name": GameNames.ESCALATION_GAME.value,
        "scenario_class": EscalationGameScenario,
        "decision_class": EscalationGameDecision,
        "payoff_matrix": ALL_GAME_PAYOFF[GameNames.ESCALATION_GAME],
        "data_path": data_path_format.format(GameNames.ESCALATION_GAME.value),
    },
    GameNames.TRUST_GAME_TRUSTOR: {
        "game_name": GameNames.TRUST_GAME_TRUSTOR.value,
        "scenario_class": TrustGameTrustorScenario,
        "decision_class": TrustGameDecision,
        "payoff_matrix": dict(),
        "data_path": data_path_format.format(GameNames.TRUST_GAME_TRUSTOR.value),
        "data_folder": data_folder_format.format(GameNames.TRUST_GAME_TRUSTOR.value),
    },
    GameNames.TRUST_GAME_TRUSTEE: {
        "game_name": GameNames.TRUST_GAME_TRUSTEE.value,
        "scenario_class": TrustGameTrusteeScenario,
        "decision_class": TrustGameDecision,
        "payoff_matrix": dict(),
        "data_path": data_path_format.format(GameNames.TRUST_GAME_TRUSTOR.value),
        "data_folder": data_folder_format.format(GameNames.TRUST_GAME_TRUSTOR.value),
    },
    GameNames.ULTIMATUM_GAME_PROPOSER: {
        "game_name": GameNames.ULTIMATUM_GAME_PROPOSER.value,
        "scenario_class": UltimatumGameProposerScenario,
        "decision_class": UltimatumGameDecision,
        "payoff_matrix": dict(),
        "data_path": data_path_format.format(GameNames.ULTIMATUM_GAME_PROPOSER.value),
        "data_folder": data_folder_format.format(
            GameNames.ULTIMATUM_GAME_PROPOSER.value
        ),
    },
    GameNames.ULTIMATUM_GAME_RESPONDER: {
        "game_name": GameNames.ULTIMATUM_GAME_RESPONDER.value,
        "scenario_class": UltimatumGameResponderScenario,
        "decision_class": UltimatumGameDecision,
        "payoff_matrix": dict(),
        "data_path": data_path_format.format(GameNames.ULTIMATUM_GAME_PROPOSER.value),
        "data_folder": data_folder_format.format(
            GameNames.ULTIMATUM_GAME_PROPOSER.value
        ),
    },
}


def get_game_config(game_name: Union[str, GameNames]) -> dict:
    """Get the configuration for a specific game.

    Args:
        game_name: Name of the game to get configuration for

    Returns:
        Dictionary containing game configuration

    Raises:
        ValueError: If game_name is not found in configurations
    """
    if isinstance(game_name, str):
        game_name = GameNames.from_string(game_name)

    if game_name not in GAME_CONFIGS:
        available_games = ", ".join(GAME_CONFIGS.keys())
        raise ValueError(
            f"Unsupported game: {game_name}. Available games: {available_games}"
        )

    return GAME_CONFIGS[game_name]
