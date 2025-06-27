import unittest

from constants import Emotions, GameNames, GameType, SymmetryType


class TestEmotions(unittest.TestCase):
    """Test cases for the Emotions enum."""

    def test_get_emotions(self):
        """Test that get_emotions returns all emotion values."""
        emotions = Emotions.get_emotions()
        expected = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
        self.assertEqual(sorted(emotions), sorted(expected))

    def test_from_string_exact_match(self):
        """Test from_string with exact matches."""
        self.assertEqual(Emotions.from_string("anger"), Emotions.ANGER)
        self.assertEqual(Emotions.from_string("happiness"), Emotions.HAPPINESS)
        self.assertEqual(Emotions.from_string("SADNESS"), Emotions.SADNESS)

    def test_from_string_prefix_match(self):
        """Test from_string with prefix matching."""
        self.assertEqual(Emotions.from_string("ang"), Emotions.ANGER)
        self.assertEqual(Emotions.from_string("hap"), Emotions.HAPPINESS)
        self.assertEqual(Emotions.from_string("sad"), Emotions.SADNESS)

    def test_from_string_invalid(self):
        """Test from_string with invalid emotion."""
        with self.assertRaises(ValueError):
            Emotions.from_string("invalid")
        with self.assertRaises(ValueError):
            Emotions.from_string("x")  # Too short


class TestGameType(unittest.TestCase):
    """Test cases for the GameType enum."""

    def test_game_type_values(self):
        """Test GameType enum values."""
        self.assertEqual(GameType.SIMULTANEOUS.value, "simultaneous")
        self.assertEqual(GameType.SEQUENTIAL.value, "sequential")


class TestSymmetryType(unittest.TestCase):
    """Test cases for the SymmetryType enum."""

    def test_symmetry_type_values(self):
        """Test SymmetryType enum values."""
        self.assertEqual(SymmetryType.SYMMETRIC.value, "symmetric")
        self.assertEqual(SymmetryType.ASYMMETRIC.value, "asymmetric")


class TestGameNames(unittest.TestCase):
    """Test cases for the GameNames enum."""

    def test_game_initialization(self):
        """Test that games are initialized with correct types."""
        # Test a simultaneous symmetric game
        self.assertEqual(GameNames.PRISONERS_DILEMMA.game_type, GameType.SIMULTANEOUS)
        self.assertEqual(
            GameNames.PRISONERS_DILEMMA.symmetry_type, SymmetryType.SYMMETRIC
        )

        # Test a sequential asymmetric game
        self.assertEqual(GameNames.TRUST_GAME_TRUSTOR.game_type, GameType.SEQUENTIAL)
        self.assertEqual(
            GameNames.TRUST_GAME_TRUSTOR.symmetry_type, SymmetryType.ASYMMETRIC
        )

    def test_from_string_exact_match(self):
        """Test from_string with exact matches."""
        game = GameNames.from_string("Prisoners_Dilemma")
        self.assertEqual(game, GameNames.PRISONERS_DILEMMA)

    def test_from_string_prefix_match(self):
        """Test from_string with prefix matching."""
        game = GameNames.from_string("prisoners")
        self.assertEqual(game, GameNames.PRISONERS_DILEMMA)

        game = GameNames.from_string("escalation")
        self.assertEqual(game, GameNames.ESCALATION_GAME)

    def test_from_string_invalid(self):
        """Test from_string with invalid game name."""
        with self.assertRaises(ValueError):
            GameNames.from_string("invalid_game")

    def test_get_game_type(self):
        """Test get_game_type method."""
        game_type = GameNames.get_game_type("prisoners")
        self.assertEqual(game_type, GameType.SIMULTANEOUS)

        game_type = GameNames.get_game_type("escalation")
        self.assertEqual(game_type, GameType.SEQUENTIAL)

    def test_get_symmetry_type(self):
        """Test get_symmetry_type method."""
        symmetry_type = GameNames.get_symmetry_type("prisoners")
        self.assertEqual(symmetry_type, SymmetryType.SYMMETRIC)

        symmetry_type = GameNames.get_symmetry_type("trust_game_trustor")
        self.assertEqual(symmetry_type, SymmetryType.ASYMMETRIC)

    def test_get_simultaneous_games(self):
        """Test get_simultaneous_games method."""
        simultaneous_games = GameNames.get_simultaneous_games()
        expected_names = [
            "Prisoners_Dilemma",
            "Battle_Of_Sexes",
            "Wait_Go",
            "Duopolistic_Competition",
            "Stag_Hunt",
        ]
        actual_names = [game.value for game in simultaneous_games]
        self.assertEqual(sorted(actual_names), sorted(expected_names))

    def test_get_sequential_games(self):
        """Test get_sequential_games method."""
        sequential_games = GameNames.get_sequential_games()
        # Should include all games that are not simultaneous
        for game in sequential_games:
            self.assertEqual(game.game_type, GameType.SEQUENTIAL)

    def test_get_symmetric_games(self):
        """Test get_symmetric_games method."""
        symmetric_games = GameNames.get_symmetric_games()
        expected_symmetric = [
            "Prisoners_Dilemma",
            "Battle_Of_Sexes",
            "Wait_Go",
            "Duopolistic_Competition",
            "Stag_Hunt",
            "Escalation_Game",
            "Monopoly_Game",
            "Hot_Cold_Game",
            "Draco_Game",
            "Tri_Game",
        ]
        actual_names = [game.value for game in symmetric_games]
        self.assertEqual(sorted(actual_names), sorted(expected_symmetric))

    def test_get_asymmetric_games(self):
        """Test get_asymmetric_games method."""
        asymmetric_games = GameNames.get_asymmetric_games()
        expected_asymmetric = [
            "Trust_Game_Trustor",
            "Trust_Game_Trustee",
            "Ultimatum_Game_Proposer",
            "Ultimatum_Game_Responder",
        ]
        actual_names = [game.value for game in asymmetric_games]
        self.assertEqual(sorted(actual_names), sorted(expected_asymmetric))

    def test_get_games_by_type_both_filters(self):
        """Test get_games_by_type with both game_type and symmetry_type filters."""
        # Simultaneous + Symmetric
        games = GameNames.get_games_by_type(
            GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC
        )
        expected = [
            "Prisoners_Dilemma",
            "Battle_Of_Sexes",
            "Wait_Go",
            "Duopolistic_Competition",
            "Stag_Hunt",
        ]
        actual_names = [game.value for game in games]
        self.assertEqual(sorted(actual_names), sorted(expected))

        # Sequential + Asymmetric
        games = GameNames.get_games_by_type(
            GameType.SEQUENTIAL, SymmetryType.ASYMMETRIC
        )
        expected = [
            "Trust_Game_Trustor",
            "Trust_Game_Trustee",
            "Ultimatum_Game_Proposer",
            "Ultimatum_Game_Responder",
        ]
        actual_names = [game.value for game in games]
        self.assertEqual(sorted(actual_names), sorted(expected))

    def test_get_games_by_type_single_filter(self):
        """Test get_games_by_type with only one filter."""
        # Only game_type filter
        games = GameNames.get_games_by_type(game_type=GameType.SIMULTANEOUS)
        simultaneous_games = GameNames.get_simultaneous_games()
        self.assertEqual(len(games), len(simultaneous_games))

        # Only symmetry_type filter
        games = GameNames.get_games_by_type(symmetry_type=SymmetryType.SYMMETRIC)
        symmetric_games = GameNames.get_symmetric_games()
        self.assertEqual(len(games), len(symmetric_games))

    def test_get_games_by_type_no_filter(self):
        """Test get_games_by_type with no filters returns all games."""
        all_games = GameNames.get_games_by_type()
        self.assertEqual(len(all_games), len(list(GameNames)))

    def test_boolean_methods(self):
        """Test boolean helper methods."""
        # Test is_sequential/is_simultaneous
        self.assertTrue(GameNames.PRISONERS_DILEMMA.is_simultaneous())
        self.assertFalse(GameNames.PRISONERS_DILEMMA.is_sequential())

        self.assertTrue(GameNames.ESCALATION_GAME.is_sequential())
        self.assertFalse(GameNames.ESCALATION_GAME.is_simultaneous())

        # Test is_symmetric/is_asymmetric
        self.assertTrue(GameNames.PRISONERS_DILEMMA.is_symmetric())
        self.assertFalse(GameNames.PRISONERS_DILEMMA.is_asymmetric())

        self.assertTrue(GameNames.TRUST_GAME_TRUSTOR.is_asymmetric())
        self.assertFalse(GameNames.TRUST_GAME_TRUSTOR.is_symmetric())

    def test_all_games_have_both_attributes(self):
        """Test that all games have both game_type and symmetry_type attributes."""
        for game in GameNames:
            self.assertIsInstance(game.game_type, GameType)
            self.assertIsInstance(game.symmetry_type, SymmetryType)

    def test_asymmetric_games_are_sequential(self):
        """Test that all asymmetric games are sequential (based on current data)."""
        asymmetric_games = GameNames.get_asymmetric_games()
        for game in asymmetric_games:
            self.assertEqual(game.game_type, GameType.SEQUENTIAL)


if __name__ == "__main__":
    unittest.main()
