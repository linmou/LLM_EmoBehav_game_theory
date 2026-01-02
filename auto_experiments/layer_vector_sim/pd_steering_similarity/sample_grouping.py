"""
Sample grouping utilities for PD steering similarity.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class PDSample:
    sample_id: str
    baseline_choice: str
    steered_choice: str

    @property
    def switched_to_defect(self) -> bool:
        return self.baseline_choice != "defect" and self.steered_choice == "defect"


def load_samples(raw_results_path: Path) -> List[PDSample]:
    data = json.loads(Path(raw_results_path).read_text())
    samples: List[PDSample] = []
    for entry in data:
        samples.append(
            PDSample(
                sample_id=str(entry.get("id")),
                baseline_choice=str(entry.get("baseline_choice")),
                steered_choice=str(entry.get("steered_choice")),
            )
        )
    return samples


def filter_switchers(samples: Iterable[PDSample]) -> List[PDSample]:
    return [s for s in samples if s.switched_to_defect]


def filter_non_switchers(samples: Iterable[PDSample]) -> List[PDSample]:
    return [s for s in samples if not s.switched_to_defect]
