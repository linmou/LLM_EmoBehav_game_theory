"""
Keep `neuro_manipulation.repe` import-light.

Why: RepReader pickles commonly reference classes under this package. Importing this
package must not eagerly pull optional heavy deps (notably `vllm`), otherwise basic
workflows like unpickling readers or running analysis scripts will crash.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

# RepReading symbols (safe: no vLLM import).
from .rep_readers import *  # noqa: F401,F403
from .rep_reading_pipeline import *  # noqa: F401,F403


def __getattr__(name: str):
    # Lazy accessors for optional / heavy components.
    if name == "repe_pipeline_registry":
        from .pipelines import repe_pipeline_registry

        return repe_pipeline_registry

    raise AttributeError(name)
