# PayoffMatrix Documentation

The `PayoffMatrix` class is a data structure for representing game theory payoff matrices in a unified way for both simultaneous and sequential games.

## Key Classes

### PayoffLeaf

The `PayoffLeaf` class represents a single outcome in a game with associated payoffs:

```python
PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3))
```

- `actions`: Tuple of actions taken by each player (player1_action, player2_action)
- `payoffs`: Tuple of payoffs received by each player (player1_payoff, player2_payoff)
- `ranks`: Optional tuple of ranks for this outcome from each player's perspective (calculated automatically)

### PayoffMatrix

The `PayoffMatrix` class represents the complete payoff structure of a game:

```python
matrix = PayoffMatrix(
    player_num=2,
    payoff_leaves=[
        PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3)),
        PayoffLeaf(actions=("cooperate", "defect"), payoffs=(0, 5)),
        PayoffLeaf(actions=("defect", "cooperate"), payoffs=(5, 0)),
        PayoffLeaf(actions=("defect", "defect"), payoffs=(1, 1)),
    ]
)
```

- `player_num`: Number of players in the game
- `payoff_leaves`: List of PayoffLeaf objects representing all possible outcomes
- `ordered_payoff_leaves`: Dictionary mapping player indices to their ordered preferences (calculated automatically)

## Features

### Preference Ordering

The `build_ranks` validator automatically calculates each player's preference ordering among all possible outcomes. For each player, outcomes are sorted from highest to lowest payoff.

### Human-Readable Description

The `__str__` method provides a textual description of the game, including:
- Basic descriptions of all possible outcomes and their payoffs
- Each player's preference ordering from most to least preferred outcomes

Example output:
```
Game Payoffs:
When player 1 chooses 'cooperate' and player 2 chooses 'cooperate', player 1 gets 3 and player 2 gets 3.
When player 1 chooses 'cooperate' and player 2 chooses 'defect', player 1 gets 0 and player 2 gets 5.
When player 1 chooses 'defect' and player 2 chooses 'cooperate', player 1 gets 5 and player 2 gets 0.
When player 1 chooses 'defect' and player 2 chooses 'defect', player 1 gets 1 and player 2 gets 1.

Preference Ordering (from most to least preferred outcomes):

Player 1's preferences:
1. defect vs cooperate (payoff: 5)
2. cooperate vs cooperate (payoff: 3)
3. defect vs defect (payoff: 1)
4. cooperate vs defect (payoff: 0)

Player 2's preferences:
1. cooperate vs defect (payoff: 5)
2. cooperate vs cooperate (payoff: 3)
3. defect vs defect (payoff: 1)
4. defect vs cooperate (payoff: 0)
```

## Usage

To create a new game matrix:

```python
game_matrix = PayoffMatrix(
    player_num=2,
    payoff_leaves=[
        # List all possible outcomes with their payoffs
        PayoffLeaf(actions=("action1", "action1"), payoffs=(payoff1_1, payoff1_2)),
        PayoffLeaf(actions=("action1", "action2"), payoffs=(payoff2_1, payoff2_2)),
        # ...
    ]
)

# Get human-readable description
print(game_matrix)

# Access ordered preferences for player 0 (first player)
print(game_matrix.ordered_payoff_leaves[0])
``` 