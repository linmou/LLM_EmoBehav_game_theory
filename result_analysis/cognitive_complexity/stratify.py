from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def tertile_bins(values: Iterable[float]) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Split values into 3 bins with near-equal counts (stable ascending order).

    Returns (low_bin, mid_bin, high_bin) as tuples of the original sorted values.
    """
    arr: List[float] = sorted(values)
    n = len(arr)
    if n == 0:
        return tuple(), tuple(), tuple()
    base = n // 3
    rem = n % 3
    # Distribute remainder to low then mid for stability
    n_low = base + (1 if rem > 0 else 0)
    n_mid = base + (1 if rem > 1 else 0)
    n_high = n - n_low - n_mid
    low = tuple(arr[:n_low])
    mid = tuple(arr[n_low : n_low + n_mid])
    high = tuple(arr[n_low + n_mid :])
    return low, mid, high

