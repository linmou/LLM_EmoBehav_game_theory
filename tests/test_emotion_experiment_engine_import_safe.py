"""
tests/test_emotion_experiment_engine_import_safe.py
Purpose: Ensure importing `emotion_experiment_engine` does not SIGABRT by eagerly importing GPU-heavy deps.
"""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestEmotionExperimentEngineImportSafety(unittest.TestCase):
    def test_import_does_not_abort(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import emotion_experiment_engine; print('ok')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"import aborted/failed: rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

