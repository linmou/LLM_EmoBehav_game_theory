"""
Tests for `neuro_manipulation/threading_env.py::ensure_mkl_threading_layer`.

Purpose: vLLM spawns a subprocess that can crash when `MKL_THREADING_LAYER=INTEL`
is set alongside `libgomp` (common on many HPC/conda setups). We enforce a safe
default so vLLM model inspection doesn't die before inference starts.
"""

import os
import unittest
from unittest.mock import patch


class TestMKLThreadingEnv(unittest.TestCase):
    def test_sets_default_when_unset(self):
        from neuro_manipulation.threading_env import ensure_mkl_threading_layer

        with patch.dict(os.environ, {}, clear=True):
            ensure_mkl_threading_layer()
            self.assertEqual(os.environ.get("MKL_THREADING_LAYER"), "GNU")

    def test_overrides_intel_to_gnu(self):
        from neuro_manipulation.threading_env import ensure_mkl_threading_layer

        with patch.dict(os.environ, {"MKL_THREADING_LAYER": "INTEL"}, clear=True):
            ensure_mkl_threading_layer()
            self.assertEqual(os.environ.get("MKL_THREADING_LAYER"), "GNU")

    def test_keeps_gnu(self):
        from neuro_manipulation.threading_env import ensure_mkl_threading_layer

        with patch.dict(os.environ, {"MKL_THREADING_LAYER": "GNU"}, clear=True):
            ensure_mkl_threading_layer()
            self.assertEqual(os.environ.get("MKL_THREADING_LAYER"), "GNU")

