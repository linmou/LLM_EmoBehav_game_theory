"""
Store payoff matrices in a unified data structure for both simultaneous and sequential games.
"""

from typing import Dict, Optional, Tuple

from pydantic import BaseModel

from constants import GameNames

from .game_tree import PayoffMatrix

# SIMULTANEOUS GAMES

# Prisoner's Dilemma
# | 3,3 | 0,5 |
# | 5,0 | 1,1 |
prisoners_dilemma = {
    "p1": {
        "cooperate": {"p2_cooperate": 3, "p2_defect": 0},
        "defect": {"p2_cooperate": 5, "p2_defect": 1},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 3, "p1_defect": 0},
        "defect": {"p1_cooperate": 5, "p1_defect": 1},
    },
}


class PayoffLeaf(BaseModel):
    actions: Tuple[str, str]
    payoffs: Optional[Tuple[int, int]]  # payoffs for each player
    ranks: Tuple[int, int]  # ranks of this payoff for each player


PD = [
    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3), ranks=(3, 3)),
    PayoffLeaf(actions=("cooperate", "defect"), payoffs=(0, 5), ranks=(1, 4)),
    PayoffLeaf(actions=("defect", "cooperate"), payoffs=(5, 0), ranks=(4, 1)),
    PayoffLeaf(actions=("defect", "defect"), payoffs=(1, 1), ranks=(2, 2)),
]

# Stag Hunt
# | 3,3 | 0,1 |
# | 1,0 | 1,1 |
stag_hunt = {
    "p1": {
        "stag": {"p2_stag": 3, "p2_hare": 0},
        "hare": {"p2_stag": 1, "p2_hare": 1},
    },
    "p2": {
        "stag": {"p1_stag": 3, "p1_hare": 0},
        "hare": {"p1_stag": 1, "p1_hare": 1},
    },
}

# Battle of the Sexes
# | 2,1 | 0,0 |
# | 0,0 | 1,2 |
battle_of_sexes = {
    "Alice": {
        "opera": {"Bob_opera": 2, "Bob_football": 0},
        "football": {"Bob_opera": 0, "Bob_football": 1},
    },
    "Bob": {
        "opera": {"Alice_opera": 1, "Alice_football": 0},
        "football": {"Alice_opera": 0, "Alice_football": 2},
    },
}

# Duopolistic Competition
duopolistic_competition = {
    "Alice": {
        "choice_1": {
            "Bob_choice_1": 0,
            "Bob_choice_2": 0,
            "Bob_choice_3": 0,
            "Bob_choice_4": 0,
            "Bob_choice_5": 0,
            "Bob_choice_6": 0,
        },
        "choice_2": {
            "Bob_choice_1": 9,
            "Bob_choice_2": 7,
            "Bob_choice_3": 5,
            "Bob_choice_4": 3,
            "Bob_choice_5": 1,
            "Bob_choice_6": -1,
        },
        "choice_3": {
            "Bob_choice_1": 14,
            "Bob_choice_2": 10,
            "Bob_choice_3": 6,
            "Bob_choice_4": 2,
            "Bob_choice_5": -2,
            "Bob_choice_6": -2,
        },
        "choice_4": {
            "Bob_choice_1": 15,
            "Bob_choice_2": 9,
            "Bob_choice_3": 3,
            "Bob_choice_4": -3,
            "Bob_choice_5": -3,
            "Bob_choice_6": -3,
        },
        "choice_5": {
            "Bob_choice_1": 12,
            "Bob_choice_2": 4,
            "Bob_choice_3": -4,
            "Bob_choice_4": -4,
            "Bob_choice_5": -4,
            "Bob_choice_6": -4,
        },
        "choice_6": {
            "Bob_choice_1": 5,
            "Bob_choice_2": -5,
            "Bob_choice_3": -5,
            "Bob_choice_4": -5,
            "Bob_choice_5": -5,
            "Bob_choice_6": -5,
        },
    },
    "Bob": {
        "choice_1": {
            "Alice_choice_1": 0,
            "Alice_choice_2": 0,
            "Alice_choice_3": 0,
            "Alice_choice_4": 0,
            "Alice_choice_5": 0,
            "Alice_choice_6": 0,
        },
        "choice_2": {
            "Alice_choice_1": 9,
            "Alice_choice_2": 7,
            "Alice_choice_3": 5,
            "Alice_choice_4": 3,
            "Alice_choice_5": 1,
            "Alice_choice_6": -1,
        },
        "choice_3": {
            "Alice_choice_1": 14,
            "Alice_choice_2": 10,
            "Alice_choice_3": 6,
            "Alice_choice_4": 2,
            "Alice_choice_5": -2,
            "Alice_choice_6": -2,
        },
        "choice_4": {
            "Alice_choice_1": 15,
            "Alice_choice_2": 9,
            "Alice_choice_3": 3,
            "Alice_choice_4": -3,
            "Alice_choice_5": -3,
            "Alice_choice_6": -3,
        },
        "choice_5": {
            "Alice_choice_1": 12,
            "Alice_choice_2": 4,
            "Alice_choice_3": -4,
            "Alice_choice_4": -4,
            "Alice_choice_5": -4,
            "Alice_choice_6": -4,
        },
        "choice_6": {
            "Alice_choice_1": 5,
            "Alice_choice_2": -5,
            "Alice_choice_3": -5,
            "Alice_choice_4": -5,
            "Alice_choice_5": -5,
            "Alice_choice_6": -5,
        },
    },
}

# Wait Go Game
wait_go = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 0, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 2, "Bob_choice_2": -4},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 0, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 2, "Alice_choice_2": -4},
    },
}

# SEQUENTIAL GAMES

# Escalation Game
# Nash equilibrium: Player 1 chooses "withdraw"
escalation_game = {
    "withdraw": [0, 0],
    "escalate": {
        "withdraw": [1, -2],
        "escalate": {
            "withdraw": [-2, 1],
            "escalate": [-1, -1],
        },
    },
}

# Monopoly Game
# Nash equilibrium: Player 1 chooses "choice_2", Player 2 chooses "choice_1"
monopoly_game = {
    "choice_1": [0, 2],
    "choice_2": {
        "choice_1": [2, 1],
        "choice_2": [-1, -1],
    },
}

# Ultimatum Game (simplified)
# Player 1 proposes a split (fair/unfair), Player 2 accepts or rejects
ultimatum_game = {
    "fair_split": {
        "accept": [5, 5],
        "reject": [0, 0],
    },
    "unfair_split": {
        "accept": [8, 2],
        "reject": [0, 0],
    },
}

# Hot Cold Game
hot_cold_game = {
    "Alice_choice_1": {"Bob_choice_1": [3, 2], "Bob_choice_2": [2, 3]},
    "Alice_choice_2": {"Bob_choice_1": [1, 4], "Bob_choice_2": [4, 1]},
}

# Draco Game
draco = {
    "Alice_choice_1": {
        "Bob_choice_1": [5, 5],
        "Bob_choice_2": {"Alice_choice_1": [2, 2], "Alice_choice_2": [3, 4]},
    },
    "Alice_choice_2": {
        "Bob_choice_1": [4, 5],
        "Bob_choice_2": {"Alice_choice_1": [5, 3], "Alice_choice_2": [2, 2]},
    },
}

# Trigame
trigame = {
    "Alice_choice_1": {
        "Bob_choice_1": {"Alice_choice_1": [20, 3], "Alice_choice_2": [0, 4]},
        "Bob_choice_2": {"Alice_choice_1": [2, 5], "Alice_choice_2": [3, 4]},
    },
    "Alice_choice_2": {
        "Bob_choice_1": {"Alice_choice_1": [1, 5], "Alice_choice_2": [4, 10]},
        "Bob_choice_2": {"Alice_choice_1": [2, 1], "Alice_choice_2": [3, 2]},
    },
}

# Create PayoffMatrix objects using GameNames enum as keys
SIMULTANEOUS_GAMES = {
    GameNames.PRISONERS_DILEMMA: PayoffMatrix.from_simultaneous_dict(
        prisoners_dilemma, name="Prisoner's Dilemma"
    ),
    GameNames.STAG_HUNT: PayoffMatrix.from_simultaneous_dict(
        stag_hunt, name="Stag Hunt"
    ),
    GameNames.BATTLE_OF_SEXES: PayoffMatrix.from_simultaneous_dict(
        battle_of_sexes, name="Battle of the Sexes"
    ),
    GameNames.DUOPOLISTIC_COMPETITION: PayoffMatrix.from_simultaneous_dict(
        duopolistic_competition, name="Duopolistic Competition"
    ),
    GameNames.WAIT_GO: PayoffMatrix.from_simultaneous_dict(
        wait_go, name="Wait/Go Game"
    ),
    # Assuming game_of_chicken and rock_paper_scissors have corresponding GameNames entries
    # If not, they need to be added to constants.py or handled differently
    # GameNames.GAME_OF_CHICKEN: PayoffMatrix.from_simultaneous_dict(
    #     game_of_chicken, name="Game of Chicken"
    # ),
    # GameNames.ROCK_PAPER_SCISSORS: PayoffMatrix.from_simultaneous_dict(
    #     rock_paper_scissors, name="Rock-Paper-Scissors"
    # ),
}

SEQUENTIAL_GAMES = {
    GameNames.ESCALATION_GAME: PayoffMatrix.from_sequential_dict(
        escalation_game, players=["p1", "p2"], name="Escalation Game"
    ),
    GameNames.MONOPOLY_GAME: PayoffMatrix.from_sequential_dict(
        monopoly_game, players=["p1", "p2"], name="Monopoly Game"
    ),
    # Using ULTIMATUM_GAME_PROPOSER as the key for the base ultimatum game payoff
    # The specific role logic might be handled elsewhere or requires separate PayoffMatrix instances
    GameNames.ULTIMATUM_GAME_PROPOSER: PayoffMatrix.from_sequential_dict(
        ultimatum_game, players=["proposer", "responder"], name="Ultimatum Game"
    ),
    GameNames.HOT_COLD_GAME: PayoffMatrix.from_sequential_dict(
        hot_cold_game,
        players=["Alice", "Bob"],
        name="Hot/Cold Game",  # Assuming Alice, Bob are players
    ),
    GameNames.DRACO_GAME: PayoffMatrix.from_sequential_dict(
        draco,
        players=["Alice", "Bob"],
        name="Draco Game",  # Assuming Alice, Bob are players
    ),
    GameNames.TRI_GAME: PayoffMatrix.from_sequential_dict(
        trigame,
        players=["Alice", "Bob"],
        name="Trigame",  # Assuming Alice, Bob are players
    ),
    # Note: TRUST_GAME payoff matrix definition is missing from the provided old file.
}

# Combined dictionary of all games
ALL_GAME_PAYOFF = {**SIMULTANEOUS_GAMES, **SEQUENTIAL_GAMES}


if __name__ == "__main__":
    print(ALL_GAME_PAYOFF[GameNames.ULTIMATUM_GAME_PROPOSER].describe_game())
