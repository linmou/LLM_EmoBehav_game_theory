# tests/games/test_sealed_auction_game_config.py
# Purpose: validate sealed auction game config wiring and scenario defaults + parsing.
from constants import GameNames
from games.game_configs import get_game_config
from games.sealed_auction import SealedAuctionDecision, SealedAuctionScenario


def test_sealed_auction_game_config():
    config = get_game_config(GameNames.SEALED_AUCTION)

    assert config["game_name"] == GameNames.SEALED_AUCTION.value
    assert config["scenario_class"] is SealedAuctionScenario
    assert config["decision_class"] is SealedAuctionDecision
    assert isinstance(config["payoff_matrix"], dict)
    assert (
        config["data_path"]
        == "data_creation/scenario_creation/langgraph_creation/Diplomacy_Sealed_Auction_all_data_samples.json"
    )


def test_sealed_auction_scenario_parses_choices():
    scenario = SealedAuctionScenario(
        scenario="Test Auction",
        description="Bids are submitted secretly; highest bid wins.",
        participants=[
            {"name": "You (Commander of France)"},
            {"name": "Commander of England"},
            {"name": "Commander of Germany"},
            {"name": "Commander of a minor power"},
        ],
        payoff_matrix={},
        behavior_choices={
            "devote_low": "Low bid",
            "devote_medium": "Medium bid",
            "devote_high": "High bid",
        },
        game_category="SEALED_BID_AUCTION_MULTIPARTY",
    )

    choices = scenario.get_behavior_choices().get_choices()
    assert choices == ["Low bid", "Medium bid", "High bid"]
    assert scenario.find_behavior_from_decision("Medium bid") == "devote_medium"
