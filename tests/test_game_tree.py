import unittest
from typing import List

from pydantic import BaseModel, Field

from constants import GameNames
from games.game import BehaviorChoices, GameScenario
from games.game_tree import DecisionNode, PayoffMatrix, SimultaneousNode, TerminalNode
from games.payoff_matrices import ALL_GAME_PAYOFF


class TestGameTree(unittest.TestCase):
    """Test cases for the game tree data structure"""

    def test_terminal_node(self):
        """Test creating a terminal node"""
        node = TerminalNode(payoffs=(3, 5))
        self.assertEqual(node.payoffs, (3, 5))

    def test_decision_node(self):
        """Test creating a decision node"""
        terminal1 = TerminalNode(payoffs=(1, 0))
        terminal2 = TerminalNode(payoffs=(0, 1))

        node = DecisionNode(player="Alice")
        node.actions["left"] = terminal1
        node.actions["right"] = terminal2

        self.assertEqual(node.player, "Alice")
        self.assertEqual(len(node.actions), 2)
        self.assertEqual(node.actions["left"].payoffs, (1, 0))
        self.assertEqual(node.actions["right"].payoffs, (0, 1))

    def test_simultaneous_node(self):
        """Test creating a simultaneous node"""
        terminal1 = TerminalNode(payoffs=(3, 3))
        terminal2 = TerminalNode(payoffs=(0, 5))
        terminal3 = TerminalNode(payoffs=(5, 0))
        terminal4 = TerminalNode(payoffs=(1, 1))

        node = SimultaneousNode(players=("Alice", "Bob"))
        node.actions[("cooperate", "cooperate")] = terminal1
        node.actions[("cooperate", "defect")] = terminal2
        node.actions[("defect", "cooperate")] = terminal3
        node.actions[("defect", "defect")] = terminal4

        self.assertEqual(node.players, ("Alice", "Bob"))
        self.assertEqual(len(node.actions), 4)
        self.assertEqual(node.actions[("cooperate", "cooperate")].payoffs, (3, 3))
        self.assertEqual(node.actions[("defect", "defect")].payoffs, (1, 1))

    def test_payoff_matrix_from_simultaneous_dict(self):
        """Test creating a PayoffMatrix from a simultaneous game dictionary"""
        pd_dict = {
            "p1": {
                "cooperate": {"p2_cooperate": 3, "p2_defect": 0},
                "defect": {"p2_cooperate": 5, "p2_defect": 1},
            },
            "p2": {
                "cooperate": {"p1_cooperate": 3, "p1_defect": 0},
                "defect": {"p1_cooperate": 5, "p1_defect": 1},
            },
        }

        matrix = PayoffMatrix.from_simultaneous_dict(pd_dict, name="Prisoner's Dilemma")

        self.assertEqual(matrix.name, "Prisoner's Dilemma")
        self.assertIsInstance(matrix.game_tree, SimultaneousNode)
        self.assertEqual(matrix.game_tree.players, ("p1", "p2"))
        self.assertEqual(len(matrix.game_tree.actions), 4)

        # Check payoffs
        self.assertEqual(
            matrix.game_tree.actions[("cooperate", "cooperate")].payoffs, (3, 3)
        )
        self.assertEqual(
            matrix.game_tree.actions[("cooperate", "defect")].payoffs, (0, 5)
        )
        self.assertEqual(
            matrix.game_tree.actions[("defect", "cooperate")].payoffs, (5, 0)
        )
        self.assertEqual(matrix.game_tree.actions[("defect", "defect")].payoffs, (1, 1))

    def test_payoff_matrix_from_sequential_dict(self):
        """Test creating a PayoffMatrix from a sequential game dictionary"""
        ultimatum_dict = {
            "fair_split": {
                "accept": [5, 5],
                "reject": [0, 0],
            },
            "unfair_split": {
                "accept": [8, 2],
                "reject": [0, 0],
            },
        }

        matrix = PayoffMatrix.from_sequential_dict(
            ultimatum_dict, players=["proposer", "responder"]
        )

        self.assertIsInstance(matrix.game_tree, DecisionNode)
        self.assertEqual(matrix.game_tree.player, "proposer")
        self.assertEqual(len(matrix.game_tree.actions), 2)

        # Check second level
        fair_node = matrix.game_tree.actions["fair_split"]
        unfair_node = matrix.game_tree.actions["unfair_split"]

        self.assertIsInstance(fair_node, DecisionNode)
        self.assertEqual(fair_node.player, "responder")
        self.assertEqual(len(fair_node.actions), 2)

        # Check terminal nodes
        self.assertEqual(fair_node.actions["accept"].payoffs, (5, 5))
        self.assertEqual(fair_node.actions["reject"].payoffs, (0, 0))
        self.assertEqual(unfair_node.actions["accept"].payoffs, (8, 2))
        self.assertEqual(unfair_node.actions["reject"].payoffs, (0, 0))

    def test_describe_game_simultaneous(self):
        """Test generating descriptions for a simultaneous game"""
        pd_dict = {
            "p1": {
                "cooperate": {"p2_cooperate": 3, "p2_defect": 0},
                "defect": {"p2_cooperate": 5, "p2_defect": 1},
            },
            "p2": {
                "cooperate": {"p1_cooperate": 3, "p1_defect": 0},
                "defect": {"p1_cooperate": 5, "p1_defect": 1},
            },
        }

        matrix = PayoffMatrix.from_simultaneous_dict(pd_dict)
        descriptions = matrix.describe_game()

        self.assertEqual(len(descriptions), 4)
        self.assertIn("p1=cooperate → p2=cooperate: payoffs = (3, 3)", descriptions)
        self.assertIn("p1=cooperate → p2=defect: payoffs = (0, 5)", descriptions)
        self.assertIn("p1=defect → p2=cooperate: payoffs = (5, 0)", descriptions)
        self.assertIn("p1=defect → p2=defect: payoffs = (1, 1)", descriptions)

    def test_describe_game_sequential(self):
        """Test generating descriptions for a sequential game"""
        ultimatum_dict = {
            "fair_split": {
                "accept": [5, 5],
                "reject": [0, 0],
            },
            "unfair_split": {
                "accept": [8, 2],
                "reject": [0, 0],
            },
        }

        matrix = PayoffMatrix.from_sequential_dict(
            ultimatum_dict, players=["proposer", "responder"]
        )
        descriptions = matrix.describe_game()

        self.assertEqual(len(descriptions), 4)
        self.assertIn(
            "proposer=fair_split → responder=accept: payoffs = (5, 5)", descriptions
        )
        self.assertIn(
            "proposer=fair_split → responder=reject: payoffs = (0, 0)", descriptions
        )
        self.assertIn(
            "proposer=unfair_split → responder=accept: payoffs = (8, 2)", descriptions
        )
        self.assertIn(
            "proposer=unfair_split → responder=reject: payoffs = (0, 0)", descriptions
        )


class TestBehaviorChoices(BehaviorChoices):
    """Test implementation of BehaviorChoices for testing"""

    choices: List[str] = Field(default=["cooperate", "defect"])

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.choices

    def get_choices(self) -> list[str]:
        return self.choices

    @staticmethod
    def example() -> dict:
        return {"choices": ["cooperate", "defect"]}


class TestScenario(GameScenario):
    """Test implementation of GameScenario for testing"""

    name: str = "Test Scenario"
    description: str = "A test scenario"
    participants: List[dict] = Field(default=[{"name": "Alice"}, {"name": "Bob"}])
    behavior_choices: TestBehaviorChoices = Field(default_factory=TestBehaviorChoices)

    def get_scenario_info(self) -> dict:
        return {"scenario": self.name, "description": self.description}

    def get_behavior_choices(self) -> BehaviorChoices:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        return decision

    @staticmethod
    def example() -> dict:
        return {
            "name": "Test Scenario",
            "description": "A test scenario",
            "participants": [{"name": "Player1"}, {"name": "Player2"}],
        }


class TestGameScenarioWithPayoffMatrix(unittest.TestCase):
    """Test the integration of PayoffMatrix with GameScenario"""

    def test_simultaneous_payoff_matrix_description(self):
        """Test that a GameScenario uses participant names for simultaneous games"""
        scenario = TestScenario(
            participants=[{"name": "PlayerOne"}, {"name": "PlayerTwo"}],
            payoff_matrix=ALL_GAME_PAYOFF[GameNames.PRISONERS_DILEMMA],
        )

        description = scenario.payoff_matrix.get_natural_language_description(
            participants=[p["name"] for p in scenario.participants]
        )
        print(
            f"\nSimultaneous Description:\n{description}"
        )  # Optional: print for debugging

        self.assertIn(
            "PlayerOne=cooperate → PlayerTwo=cooperate: payoffs = (3, 3)", description
        )
        self.assertIn(
            "PlayerOne=cooperate → PlayerTwo=defect: payoffs = (0, 5)", description
        )
        self.assertIn(
            "PlayerOne=defect → PlayerTwo=cooperate: payoffs = (5, 0)", description
        )
        self.assertIn(
            "PlayerOne=defect → PlayerTwo=defect: payoffs = (1, 1)", description
        )

    def test_sequential_payoff_matrix_description(self):
        """Test that a GameScenario uses participant names for sequential games"""
        scenario = TestScenario(
            participants=[{"name": "Proposer"}, {"name": "Responder"}],
            payoff_matrix=ALL_GAME_PAYOFF[GameNames.ULTIMATUM_GAME_PROPOSER],
        )

        description = scenario.payoff_matrix.get_natural_language_description(
            participants=[p["name"] for p in scenario.participants]
        )
        print(
            f"\nSequential Description:\n{description}"
        )  # Optional: print for debugging

        self.assertIn(
            "Proposer=fair_split → Responder=accept: payoffs = (5, 5)", description
        )
        self.assertIn(
            "Proposer=fair_split → Responder=reject: payoffs = (0, 0)", description
        )
        self.assertIn(
            "Proposer=unfair_split → Responder=accept: payoffs = (8, 2)", description
        )
        self.assertIn(
            "Proposer=unfair_split → Responder=reject: payoffs = (0, 0)", description
        )


# --- End of moved code --- #


if __name__ == "__main__":
    unittest.main()
