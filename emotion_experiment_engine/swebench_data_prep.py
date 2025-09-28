"""Utilities for Phase 0 SWE-bench data preparation (offline retrieval + text dataset)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_PROMPT_STYLE = "style-3"
DEFAULT_FILE_SOURCE = "bm25"
DEFAULT_DOCUMENT_ENCODING = "file_name_and_contents"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_dataset_name(name: str) -> str:
    return name.replace("/", "_")


def _build_retrieval_command(
    *,
    python_executable: str,
    dataset_name: str,
    output_dir: Path,
    document_encoding: str,
    topk: int,
) -> List[str]:
    return [
        python_executable,
        "-m",
        "swebench.inference.make_datasets.bm25_retrieval",
        "--dataset_name_or_path",
        dataset_name,
        "--document_encoding_style",
        document_encoding,
        "--output_dir",
        str(output_dir),
        "--save_topk",
        str(topk),
    ]


def _build_text_dataset_command(
    *,
    python_executable: str,
    dataset_name: str,
    output_dir: Path,
    retrieval_file: Path,
    prompt_style: str,
    file_source: str,
    k: int,
    max_context_len: int,
    tokenizer_name: str,
) -> List[str]:
    return [
        python_executable,
        "-m",
        "swebench.inference.make_datasets.create_text_dataset",
        "--dataset_name_or_path",
        dataset_name,
        "--output_dir",
        str(output_dir),
        "--prompt_style",
        prompt_style,
        "--file_source",
        file_source,
        "--retrieval_file",
        str(retrieval_file),
        "--k",
        str(k),
        "--max_context_len",
        str(max_context_len),
        "--tokenizer_name",
        tokenizer_name,
    ]


def prepare_data(
    *,
    dataset_name: str = DEFAULT_DATASET,
    swebench_root: Path,
    cache_root: Path = Path("./cache"),
    python_executable: str = "python",
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    document_encoding: str = DEFAULT_DOCUMENT_ENCODING,
    file_source: str = DEFAULT_FILE_SOURCE,
    retrieval_topk: int = 20,
    text_k: int = 20,
    max_context_len: int = 32768,
    tokenizer_name: str = "llama",
    skip_retrieval: bool = False,
    dry_run: bool = False,
) -> List[List[str]]:
    swebench_root = Path(swebench_root)
    if not swebench_root.exists():
        raise FileNotFoundError(f"Expected SWE-bench repo at {swebench_root}")

    cache_root = Path(cache_root)
    retrieval_dir = _ensure_dir(cache_root / "retrieval_results")
    dataset_dir = _ensure_dir(cache_root / "datasets")

    retrieval_file = retrieval_dir / f"{_sanitize_dataset_name(dataset_name)}.retrieval.jsonl"

    commands: List[List[str]] = []

    if not skip_retrieval:
        commands.append(
            _build_retrieval_command(
                python_executable=python_executable,
                dataset_name=dataset_name,
                output_dir=retrieval_dir,
                document_encoding=document_encoding,
                topk=retrieval_topk,
            )
        )
    elif not retrieval_file.exists():
        raise FileNotFoundError(
            f"skip_retrieval=True but retrieval file missing: {retrieval_file}"
        )

    commands.append(
        _build_text_dataset_command(
            python_executable=python_executable,
            dataset_name=dataset_name,
            output_dir=dataset_dir,
            retrieval_file=retrieval_file,
            prompt_style=prompt_style,
            file_source=file_source,
            k=text_k,
            max_context_len=max_context_len,
            tokenizer_name=tokenizer_name,
        )
    )

    if dry_run:
        return commands

    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=str(swebench_root))

    return commands


def main(argv: Optional[Iterable[str]] = None) -> int:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description="Prepare offline SWE-bench datasets")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--swebench-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("./cache"))
    parser.add_argument("--python", default="python")
    parser.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE)
    parser.add_argument("--document-encoding", default=DEFAULT_DOCUMENT_ENCODING)
    parser.add_argument("--retrieval-topk", type=int, default=20)
    parser.add_argument("--text-k", type=int, default=20)
    parser.add_argument("--max-context-len", type=int, default=32768)
    parser.add_argument("--tokenizer-name", default="llama")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    commands = prepare_data(
        dataset_name=args.dataset,
        swebench_root=args.swebench_root,
        cache_root=args.cache_root,
        python_executable=args.python,
        prompt_style=args.prompt_style,
        document_encoding=args.document_encoding,
        retrieval_topk=args.retrieval_topk,
        text_k=args.text_k,
        max_context_len=args.max_context_len,
        tokenizer_name=args.tokenizer_name,
        skip_retrieval=args.skip_retrieval,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

