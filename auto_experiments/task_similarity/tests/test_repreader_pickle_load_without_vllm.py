"""
Responsible: neuro_manipulation/repe/__init__.py
Purpose: Ensure unpickling RepReader objects does not import vLLM at package import time.
"""

from __future__ import annotations

import pickle

import numpy as np


def test_unpickle_repreader_without_importing_vllm() -> None:
    # Importing this submodule requires importing the package first; if
    # neuro_manipulation.repe.__init__ imports vLLM, it will crash in this env.
    from neuro_manipulation.repe.rep_readers import PCARepReader

    rr = PCARepReader(n_components=1)
    rr.directions = {-1: np.zeros((1, 4), dtype=np.float32)}

    blob = pickle.dumps({"anger": rr}, protocol=pickle.HIGHEST_PROTOCOL)
    loaded = pickle.loads(blob)

    assert isinstance(loaded, dict)
    assert "anger" in loaded
    assert hasattr(loaded["anger"], "directions")
