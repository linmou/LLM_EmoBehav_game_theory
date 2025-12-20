"""
Tests for `neuro_manipulation/utils.py::{test_direction,get_rep_reader}`.

Purpose: Ensure direction-evaluation path defaults to the expected batch size.
"""

import unittest
from unittest.mock import MagicMock


class TestRepReaderBatchSize(unittest.TestCase):
    def test_test_direction_uses_default_batch_size_32(self):
        from neuro_manipulation.utils import test_direction

        rep_reading_pipeline = MagicMock()
        rep_reading_pipeline.return_value = [{-1: 0}]

        rep_reader = MagicMock()
        rep_reader.direction_signs = {-1: 1}

        test_data = {"data": ["x"], "labels": [[1, 0]]}
        test_direction(
            hidden_layers=[-1],
            rep_reading_pipeline=rep_reading_pipeline,
            rep_reader=rep_reader,
            test_data=test_data,
            rep_token=-1,
        )

        _, kwargs = rep_reading_pipeline.call_args
        self.assertEqual(kwargs["batch_size"], 32)

    def test_get_rep_reader_uses_default_train_batch_size_32(self):
        from neuro_manipulation.utils import get_rep_reader

        rep_reading_pipeline = MagicMock()
        rep_reading_pipeline.get_directions.return_value = MagicMock(direction_signs={-1: 1})
        rep_reading_pipeline.return_value = [{-1: 0}]

        train_data = {"data": ["x"], "labels": [[1, 0]]}
        test_data = {"data": ["x"], "labels": [[1, 0]]}

        get_rep_reader(
            rep_reading_pipeline=rep_reading_pipeline,
            train_data=train_data,
            test_data=test_data,
            hidden_layers=[-1],
            rep_token=-1,
            n_difference=1,
            direction_method="pca",
        )

        _, kwargs = rep_reading_pipeline.get_directions.call_args
        self.assertEqual(kwargs["batch_size"], 32)
