"""
Unit tests for the PayoffMatrix class and its functionality.
"""

import os
import sys
import unittest

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.payoff_matrices import PayoffLeaf, PayoffMatrix


class TestPayoffMatrix(unittest.TestCase):

    def test_prisoners_dilemma(self):
        """Test creating and using a prisoners dilemma payoff matrix"""
        pd_matrix = PayoffMatrix(
            player_num=2,
            payoff_leaves=[
                PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3)),
                PayoffLeaf(actions=("cooperate", "defect"), payoffs=(0, 5)),
                PayoffLeaf(actions=("defect", "cooperate"), payoffs=(5, 0)),
                PayoffLeaf(actions=("defect", "defect"), payoffs=(1, 1)),
            ],
        )

        # Test that ordered_payoff_leaves is correctly calculated
        self.assertEqual(len(pd_matrix.ordered_payoff_leaves), 2)

        # Player 1's best outcome should be (defect, cooperate)
        self.assertEqual(pd_matrix.ordered_payoff_leaves[0][0], ("defect", "cooperate"))

        # Player 2's best outcome should be (cooperate, defect)
        self.assertEqual(pd_matrix.ordered_payoff_leaves[1][0], ("cooperate", "defect"))

        # Test that string representation contains payoffs
        str_repr = str(pd_matrix)
        self.assertIn("cooperate", str_repr)
        self.assertIn("defect", str_repr)
        self.assertIn("player 1 gets 3", str_repr)
        self.assertIn("player 2 gets 5", str_repr)

    def test_stag_hunt(self):
        """Test creating and using a stag hunt payoff matrix"""
        stag_hunt = PayoffMatrix(
            player_num=2,
            payoff_leaves=[
                PayoffLeaf(actions=("stag", "stag"), payoffs=(3, 3)),
                PayoffLeaf(actions=("stag", "hare"), payoffs=(0, 1)),
                PayoffLeaf(actions=("hare", "stag"), payoffs=(1, 0)),
                PayoffLeaf(actions=("hare", "hare"), payoffs=(1, 1)),
            ],
        )

        # Both players should prefer (stag, stag) as their best outcome
        self.assertEqual(stag_hunt.ordered_payoff_leaves[0][0], ("stag", "stag"))
        self.assertEqual(stag_hunt.ordered_payoff_leaves[1][0], ("stag", "stag"))

        # Player 1's preference order should be: (stag, stag), (hare, stag), (hare, hare), (stag, hare)
        expected_p1_order = [
            ("stag", "stag"),
            ("hare", "stag"),
            ("hare", "hare"),
            ("stag", "hare"),
        ]
        self.assertEqual(stag_hunt.ordered_payoff_leaves[0], expected_p1_order)

        # Player 2's preference order should be: (stag, stag), (stag, hare), (hare, hare), (hare, stag)
        expected_p2_order = [
            ("stag", "stag"),
            ("stag", "hare"),
            ("hare", "hare"),
            ("hare", "stag"),
        ]
        self.assertEqual(stag_hunt.ordered_payoff_leaves[1], expected_p2_order)


if __name__ == "__main__":
    unittest.main()
