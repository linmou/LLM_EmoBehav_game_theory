"""
Quick concurrency probe for the Gemini API using GEMINI_CONFIG.

This script is exploratory, not meant for normal test runs.
It sends concurrent ListModels requests and prints status statistics
per concurrency level so you can see where errors start to appear.
"""

import concurrent.futures
import statistics
import time
from collections import Counter

import requests

try:  # Allow running as a script from the repo root.
    from api_configs import GEMINI_CONFIG
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from api_configs import GEMINI_CONFIG


def run_batch(concurrency: int, repeats: int = 1):
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": GEMINI_CONFIG["api_key"]}

    def single_call():
        start = time.time()
        try:
            r = requests.get(url, params=params, timeout=20)
            return r.status_code, time.time() - start, None
        except Exception as exc:  # pragma: no cover - exploratory script
            return None, time.time() - start, repr(exc)

    statuses: list[int | None] = []
    latencies: list[float] = []
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_call) for _ in range(concurrency * repeats)
        ]
        for f in concurrent.futures.as_completed(futures):
            status, latency, err = f.result()
            statuses.append(status)
            latencies.append(latency)
            if err is not None:
                errors.append(err)

    counter = Counter(statuses)
    ok = counter.get(200, 0)
    total = len(statuses)
    error_ratio = 1.0 - (ok / total if total else 0.0)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0)

    print(
        f"concurrency={concurrency:3d} "
        f"total={total:4d} 200={ok:4d} error_ratio={error_ratio:.2f} "
        f"latency_ms_p50={p50*1000:.1f} latency_ms_p95={p95*1000:.1f} "
        f"status_counts={dict(counter)} errors={len(errors)}"
    )


def main():
    # Keep this conservative to avoid hammering the API.
    levels = [1, 5, 10, 20, 40, 80]
    for c in levels:
        run_batch(c, repeats=1)


if __name__ == "__main__":
    main()
