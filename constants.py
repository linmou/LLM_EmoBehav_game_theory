from enum import Enum

class Emotions(Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    
    def __init__(self, *args):
        super().__init__()
        # Check for prefix overlaps during enum initialization
        for other in self.__class__:
            if other == self:
                continue
            min_len = min(len(self.value), len(other.value))
            for i in range(3, min_len + 1):
                if self.value[:i] == other.value[:i]:
                    raise ValueError(
                        f"Prefix overlap detected between {self.value} and {other.value}: "
                        f"both share prefix '{self.value[:i]}'"
                    )

    @classmethod
    def from_string(cls, value: str) -> 'Emotions':
        """
        Get an Emotions enum member from its string value.
        
        Args:
            value: String representation of the emotion
            
        Returns:
            Corresponding Emotions enum member
            
        Raises:
            ValueError: If no matching emotion is found
        """
        value = value.lower().strip()
        for emotion in cls:
            emotion_value = emotion.value.lower()
            # Case 1: Exact match
            if emotion_value == value:
                return emotion
            # Case 2: Input is a prefix of emotion value (e.g., "hap" -> "happiness")
            if len(value) > 2 and emotion_value.startswith(value):
                return emotion
            # Case 3: Emotion value is a prefix of input (e.g., "happy" matches "happiness")
            if len(value) > 2 and value.startswith(emotion_value):
                return emotion

        raise ValueError(f"No emotion found matching '{value}'")

class GameType(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"

class GameNames(Enum):
    # Simultaneous Games
    PRISONERS_DILEMMA = ("Prisoners_Dilemma", GameType.SIMULTANEOUS)
    BATTLE_OF_SEXES = ("Battle_Of_Sexes", GameType.SIMULTANEOUS)
    WAIT_GO = ("Wait_Go", GameType.SIMULTANEOUS)
    DUOPOLISTIC_COMPETITION = ("Duopolistic_Competition", GameType.SIMULTANEOUS)
    STAG_HUNT = ("Stag_Hunt", GameType.SIMULTANEOUS)
    
    # Sequential Games
    ESCALATION_GAME = ("Escalation_Game", GameType.SEQUENTIAL)
    MONOPOLY_GAME = ("Monopoly_Game", GameType.SEQUENTIAL)
    HOT_COLD_GAME = ("Hot_Cold_Game", GameType.SEQUENTIAL)
    DRACO_GAME = ("Draco_Game", GameType.SEQUENTIAL)
    TRI_GAME = ("Tri_Game", GameType.SEQUENTIAL)

    def __init__(self, value: str, game_type: GameType):
        self._value_ = value
        self.game_type = game_type

    @property
    def value(self) -> str:
        return self._value_

    @classmethod
    def from_string(cls, value: str) -> 'GameNames':
        """
        Get a GameNames enum member from its string value.
        
        Args:
            value: String representation of the game name
            
        Returns:
            Corresponding GameNames enum member
            
        Raises:
            ValueError: If no matching game is found
        """
        value = value.strip().replace(" ", "_")
        
        # Try exact match first
        for game in cls:
            if game.value.lower() == value.lower():
                return game
                
        # Try prefix match if exact match fails
        for game in cls:
            if len(value) > 2 and game.value.lower().startswith(value.lower()):
                return game
        
        raise ValueError(f"No game found matching '{value}'")

    @classmethod
    def get_game_type(cls, game_name: str) -> GameType:
        """
        Get the game type (simultaneous/sequential) from a game name string.
        
        Args:
            game_name: String representation of the game name
            
        Returns:
            GameType enum member
            
        Raises:
            ValueError: If no matching game is found
        """
        game = cls.from_string(game_name)
        return game.game_type

    @classmethod
    def get_simultaneous_games(cls) -> list['GameNames']:
        """Returns list of all simultaneous games"""
        return [game for game in cls if game.game_type == GameType.SIMULTANEOUS]

    @classmethod
    def get_sequential_games(cls) -> list['GameNames']:
        """Returns list of all sequential games"""
        return [game for game in cls if game.game_type == GameType.SEQUENTIAL]
    
    def is_sequential(self) -> bool:
        """Returns True if the game is sequential, False otherwise"""
        return self.game_type == GameType.SEQUENTIAL
    
    def is_simultaneous(self) -> bool:
        """Returns True if the game is simultaneous, False otherwise"""
        return self.game_type == GameType.SIMULTANEOUS

if __name__ == "__main__":
    print(Emotions.from_string("ang"))
    print(Emotions.from_string("happiness"))
    print(Emotions.from_string("sadness"))
    print(Emotions.from_string("SAD"))

    # Test game type functionality
    print("\nTesting game types:")
    print("Simultaneous games:", [game.value for game in GameNames.get_simultaneous_games()])
    print("Sequential games:", [game.value for game in GameNames.get_sequential_games()])
    print("PRISONERS_DILEMMA type:", GameNames.PRISONERS_DILEMMA.game_type)
    print("ESCALATION_GAME type:", GameNames.ESCALATION_GAME.game_type)

    # Test from_string and get_game_type functionality
    print("\nTesting string conversion:")
    print("From 'prisoners':", GameNames.from_string("prisoners").value)
    print("From 'ESCALATION_GAME':", GameNames.from_string("ESCALATION_GAME").value)
    
    print("\nTesting game type lookup:")
    print("Type of 'prisoners':", GameNames.get_game_type("prisoners"))
    print("Type of 'escalation':", GameNames.get_game_type("escalation"))
    
    # Test error handling
    try:
        GameNames.from_string("invalid_game")
    except ValueError as e:
        print("\nExpected error:", e)

    