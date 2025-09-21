"""Utilities to repair participant role annotations in game datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

TRUST_EXPECTED_ROLES = ("Trustor", "Trustee")
ULTIMATUM_EXPECTED_ROLES = ("Proposer", "Responder")


@dataclass
class RepairReport:
    trust_fixed: int
    ultimatum_fixed: int


def repair_participant_roles(
    *, trust_game_path: Path, ultimatum_responder_path: Path
) -> RepairReport:
    trust_fixed = _repair_file(trust_game_path, TRUST_EXPECTED_ROLES)
    ultimatum_fixed = _repair_file(ultimatum_responder_path, ULTIMATUM_EXPECTED_ROLES)
    return RepairReport(trust_fixed=trust_fixed, ultimatum_fixed=ultimatum_fixed)


def _repair_file(path: Path, expected_roles: Sequence[str]) -> int:
    records = _load_json(path)
    fixed = 0
    for record in records:
        participants = record.get("participants")
        if not isinstance(participants, list) or len(participants) < len(expected_roles):
            continue

        current_roles = {
            participant.get("role")
            for participant in participants
            if isinstance(participant, dict)
        }

        if all(role in current_roles for role in expected_roles) and all(
            isinstance(participants[idx], dict)
            and participants[idx].get("role") == role
            and participants[idx].get("name")
            for idx, role in enumerate(expected_roles)
            if idx < len(participants)
        ):
            continue

        changed = False
        for idx, role in enumerate(expected_roles):
            if idx >= len(participants):
                break
            participant = participants[idx]
            if not isinstance(participant, dict):
                continue
            if participant.get("role") != role:
                participant["role"] = role
                changed = True
            if not participant.get("name"):
                participant["name"] = f"Player_{role}"
                changed = True

        if changed:
            fixed += 1

    if fixed:
        _write_json(path, records)
    return fixed


def _load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair participant roles for Trust Game and Ultimatum Game datasets.",
    )
    parser.add_argument(
        "--trust-path",
        type=Path,
        required=True,
        help="Path to Trust Game dataset JSON (Trust_Game_Trustor_all_data_samples.json)",
    )
    parser.add_argument(
        "--ultimatum-path",
        type=Path,
        required=True,
        help="Path to Ultimatum Game dataset JSON (Ultimatum_Game_Proposer_all_data_samples.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = repair_participant_roles(
        trust_game_path=args.trust_path,
        ultimatum_responder_path=args.ultimatum_path,
    )
    print(
        f"Trust Game records repaired: {report.trust_fixed}\n"
        f"Ultimatum Game records repaired: {report.ultimatum_fixed}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
