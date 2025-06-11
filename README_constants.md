# Constants Module Documentation

## Overview

The `constants.py` module provides enumeration classes for emotions and game theory classifications used throughout the LLM Emotion-Behavior Game Theory project. It includes robust string matching and categorization capabilities for both emotions and game types.

## Classes

### Emotions Enum

Defines six basic emotions with intelligent string matching capabilities.

**Available Emotions:**
- `ANGER` - "anger"
- `HAPPINESS` - "happiness" 
- `SADNESS` - "sadness"
- `DISGUST` - "disgust"
- `FEAR` - "fear"
- `SURPRISE` - "surprise"

**Key Features:**
- **Prefix overlap detection**: Prevents ambiguous emotion definitions during initialization
- **Flexible string matching**: Supports exact matches, prefix matching, and case-insensitive input
- **Input validation**: Ensures minimum length requirements for prefix matching

**Methods:**
- `get_emotions()`: Returns list of all emotion string values
- `from_string(value)`: Converts string to Emotions enum with intelligent matching

**Example Usage:**
```python
# Exact match
emotion = Emotions.from_string("happiness")  # Returns Emotions.HAPPINESS

# Prefix match  
emotion = Emotions.from_string("ang")        # Returns Emotions.ANGER

# Case insensitive
emotion = Emotions.from_string("SAD")        # Returns Emotions.SADNESS
```

### GameType Enum

Categorizes games by their temporal structure.

**Values:**
- `SIMULTANEOUS` - "simultaneous" - Players make decisions at the same time
- `SEQUENTIAL` - "sequential" - Players make decisions in turns

### SymmetryType Enum

Categorizes games by their role symmetry between players.

**Values:**
- `SYMMETRIC` - "symmetric" - Both players have identical roles and action spaces
- `ASYMMETRIC` - "asymmetric" - Players have different roles (e.g., Proposer vs Responder)

### GameNames Enum

Comprehensive enumeration of game theory scenarios with dual classification system.

**Classification Dimensions:**
1. **Temporal Structure** (GameType): Simultaneous vs Sequential
2. **Role Symmetry** (SymmetryType): Symmetric vs Asymmetric

**Available Games:**

#### Simultaneous + Symmetric Games:
- `PRISONERS_DILEMMA` - Classic cooperation dilemma
- `BATTLE_OF_SEXES` - Coordination with conflicting preferences
- `WAIT_GO` - Traffic-like coordination game
- `DUOPOLISTIC_COMPETITION` - Market competition scenario
- `STAG_HUNT` - Trust and cooperation game

#### Sequential + Symmetric Games:
- `ESCALATION_GAME` - Conflict escalation scenario
- `MONOPOLY_GAME` - Market entry/exit decisions
- `HOT_COLD_GAME` - Information search game
- `DRACO_GAME` - Multi-stage decision making
- `TRI_GAME` - Three-option sequential choice

#### Sequential + Asymmetric Games:
- `TRUST_GAME_TRUSTOR` - Trust game from trustor perspective
- `TRUST_GAME_TRUSTEE` - Trust game from trustee perspective  
- `ULTIMATUM_GAME_PROPOSER` - Ultimatum game proposer role
- `ULTIMATUM_GAME_RESPONDER` - Ultimatum game responder role

## Key Methods

### Game Filtering Methods

```python
# Get games by single criterion
simultaneous_games = GameNames.get_simultaneous_games()
sequential_games = GameNames.get_sequential_games()
symmetric_games = GameNames.get_symmetric_games()
asymmetric_games = GameNames.get_asymmetric_games()

# Get games by combined criteria
sim_sym_games = GameNames.get_games_by_type(
    game_type=GameType.SIMULTANEOUS, 
    symmetry_type=SymmetryType.SYMMETRIC
)
```

### Type Lookup Methods

```python
# Get game type from string
game_type = GameNames.get_game_type("prisoners")          # Returns GameType.SIMULTANEOUS
symmetry_type = GameNames.get_symmetry_type("trust_game") # Returns SymmetryType.ASYMMETRIC
```

### String Conversion

```python
# Flexible string matching
game = GameNames.from_string("prisoners")     # Prefix match -> PRISONERS_DILEMMA
game = GameNames.from_string("ESCALATION_GAME")  # Exact match -> ESCALATION_GAME
```

### Boolean Helper Methods

```python
game = GameNames.PRISONERS_DILEMMA

# Check temporal structure
game.is_simultaneous()  # True
game.is_sequential()    # False

# Check symmetry
game.is_symmetric()     # True  
game.is_asymmetric()    # False
```

## Design Patterns

### Dual Classification System

Each game is classified along two independent dimensions:
1. **Temporal**: When decisions are made (simultaneous/sequential)
2. **Symmetry**: Whether players have identical roles (symmetric/asymmetric)

This allows for precise filtering and analysis:
```python
# Find all simultaneous symmetric games
games = GameNames.get_games_by_type(GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC)

# Find only asymmetric games regardless of timing
games = GameNames.get_games_by_type(symmetry_type=SymmetryType.ASYMMETRIC)
```

### Flexible String Matching

Both `Emotions` and `GameNames` support intelligent string matching:
- **Exact matches**: Full string comparison
- **Prefix matching**: Partial string matching (minimum 3 characters)
- **Case insensitive**: Automatic case normalization
- **Space/underscore handling**: Automatic conversion for game names

### Type Safety

All enums provide strong typing with validation:
- Initialization-time checks prevent invalid configurations
- Method returns are typed enum members, not strings
- ValueError exceptions for invalid inputs with descriptive messages

## Error Handling

The module provides comprehensive error handling:

```python
# Invalid emotion
try:
    Emotions.from_string("invalid")
except ValueError as e:
    print(f"Error: {e}")  # "No emotion found matching 'invalid'"

# Invalid game
try:
    GameNames.from_string("nonexistent_game")
except ValueError as e:
    print(f"Error: {e}")  # "No game found matching 'nonexistent_game'"
```

## Testing

The module includes comprehensive unit tests in `test_constants.py` covering:
- All enum functionality
- String matching edge cases
- Filtering methods
- Type lookup operations
- Error conditions
- Boolean helper methods

Run tests with:
```bash
python -m unittest test_constants.py -v
```

## Usage Examples

### Basic Game Classification

```python
from constants import GameNames, GameType, SymmetryType

# Get all symmetric games
symmetric_games = GameNames.get_symmetric_games()
print([game.value for game in symmetric_games])

# Get simultaneous games only
simultaneous_games = GameNames.get_games_by_type(game_type=GameType.SIMULTANEOUS)

# Check specific game properties
game = GameNames.TRUST_GAME_TRUSTOR
print(f"Sequential: {game.is_sequential()}")  # True
print(f"Asymmetric: {game.is_asymmetric()}")  # True
```

### Emotion Processing

```python
from constants import Emotions

# Process user input
user_input = "hap"  # User types partial emotion
try:
    emotion = Emotions.from_string(user_input)
    print(f"Detected emotion: {emotion.value}")  # "happiness"
except ValueError:
    print("Emotion not recognized")
```

This module provides a robust foundation for emotion and game theory classification with flexible input handling and comprehensive categorization capabilities. 