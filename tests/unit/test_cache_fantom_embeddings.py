# Test for scripts/cache_fantom_embeddings.py: single-file and directory caching.
"""Ensure the merged caching script handles file and directory workflows."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "cache_fantom_embeddings", REPO_ROOT / "scripts" / "cache_fantom_embeddings.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cfe = load_script_module()


class DummyEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return np.vstack([
            np.full(4, float(idx + 1), dtype=np.float32)
            for idx, _ in enumerate(texts)
        ])


def test_cache_single_file(monkeypatch, tmp_path):
    jsonl = tmp_path / "fantom_short_belief_gen_accessible.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({
                    "context": "ctx",
                    "question": "q",
                    "answer": "alpha",
                    "wrong_answer": "omega",
                }),
                json.dumps({
                    "context": "ctx",
                    "question": "q2",
                    "answer": "alpha",
                    "wrong_answer": "beta",
                }),
            ]
        )
    )

    monkeypatch.setattr(
        "emotion_memory_experiments.datasets.fantom.FantomDataset._get_embedder",
        lambda self: DummyEmbedder(),
    )

    result = cfe.cache_embeddings_for_file(
        jsonl, "short_belief_gen_accessible", limit=0, persist_prefix=None
    )

    assert result.keys_path.exists()
    assert result.vecs_path.exists()

    keys = json.loads(result.keys_path.read_text())
    assert keys == ["alpha", "omega", "beta"]

    vecs = np.load(result.vecs_path)
    assert vecs.shape == (3, 4)


def test_cache_directory(monkeypatch, tmp_path):
    for stem in (
        "fantom_short_belief_gen_accessible",
        "fantom_short_belief_gen_inaccessible",
    ):
        content = json.dumps({
            "context": "ctx",
            "question": "q",
            "answer": stem,
            "wrong_answer": "wrong",
        })
        (tmp_path / f"{stem}.jsonl").write_text(content)

    monkeypatch.setattr(
        "emotion_memory_experiments.datasets.fantom.FantomDataset._get_embedder",
        lambda self: DummyEmbedder(),
    )

    results = cfe.cache_embeddings_in_dir(tmp_path, limit=0)

    assert len(results) == 2
    for res in results:
        assert res.keys_path.exists()
        assert res.vecs_path.exists()
        keys = json.loads(res.keys_path.read_text())
        assert len(keys) == 2
