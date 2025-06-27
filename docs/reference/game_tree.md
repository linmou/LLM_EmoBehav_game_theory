# Deprecated

# Game Tree Data Structure

This document explains the game tree data structure used to represent payoff matrices for both simultaneous and sequential games.

## Overview

The game tree structure provides a unified way to represent game-theoretic scenarios in a flexible, tree-based format. This approach allows us to:

1. Represent both simultaneous and sequential games using a common data structure
2. Generate natural language descriptions of game payoffs
3. Support complex game scenarios with multiple decision points and players

## Data Structure

The game tree consists of three types of nodes:

### 1. TerminalNode

Represents the payoffs at the end of a game path.

```python
@dataclass
class TerminalNode:
    payoffs: Tuple[float, ...]  # e.g., (3, 3) for (p1, p2)
```

### 2. DecisionNode

Represents a sequential move where one player makes a decision.

```python
@dataclass
class DecisionNode:
    player: str  # e.g., "p1" or "Alice"
    actions: Dict[str, 'GameNode']  # maps action -> next node
```

### 3. SimultaneousNode

Represents a simultaneous move where multiple players choose actions at once.

```python
@dataclass
class SimultaneousNode:
    players: Tuple[str, ...]  # e.g., ("p1", "p2") or ("Alice", "Bob")
    actions: Dict[Tuple[str, ...], 'GameNode']  # maps (action1, action2, ...) -> next node
```

## PayoffMatrix Class

The `PayoffMatrix` class encapsulates a game tree and provides methods to:

1. Create a PayoffMatrix from different dictionary formats
2. Generate natural language descriptions of the game

```python
class PayoffMatrix:
    def __init__(self, game_tree: GameNode, name: str = "", description: str = ""):
        self.game_tree = game_tree
        self.name = name
        self.description = description
    
    @staticmethod
    def from_simultaneous_dict(payoff_dict): ...
    
    @staticmethod
    def from_sequential_dict(seq_dict, players=None): ...
    
    def describe_game(self): ...
    
    def get_natural_language_description(self, participants: Optional[List[str]] = None): ...
```

## Example Usage

### Simultaneous Game (Prisoner's Dilemma)

```python
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

pd_matrix = PayoffMatrix.from_simultaneous_dict(prisoners_dilemma, name="Prisoner's Dilemma")
print(pd_matrix.get_natural_language_description())
```

### Sequential Game (Ultimatum Game)

```python
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

ug_matrix = PayoffMatrix.from_sequential_dict(ultimatum_game, players=["proposer", "responder"])
print(ug_matrix.get_natural_language_description())
```

## Integration with GameScenario

The `GameScenario` class has been updated to support both the traditional dictionary-based payoff matrices and the new `PayoffMatrix` class. This provides backward compatibility while enabling more complex game structures.

When a `PayoffMatrix` instance is provided, the natural language description is generated directly via `payoff_matrix.get_natural_language_description(participants=...)`, using the participant names from the scenario. 