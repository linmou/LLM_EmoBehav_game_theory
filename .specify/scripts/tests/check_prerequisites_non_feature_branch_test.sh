#!/usr/bin/env bash
# Test for .specify/scripts/bash/check-prerequisites.sh: verifies single-spec fallback on non-feature branches.

set -euo pipefail

repo_root="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
git_root="$(cd "$repo_root" && git rev-parse --show-toplevel)"
expected_feature_dir="$(cd "$git_root/specs/002-pd-steering-sim" && pwd)"

json_output="$(cd "$repo_root" && .specify/scripts/bash/check-prerequisites.sh --json --paths-only)"

python - <<'PY' "$json_output" "$expected_feature_dir"
import json
import sys

data = json.loads(sys.argv[1])
expected = sys.argv[2]

feature_dir = data.get("FEATURE_DIR")
if feature_dir != expected:
    raise SystemExit(
        f"FEATURE_DIR mismatch:\nexpected: {expected}\nactual:   {feature_dir}"
    )
PY
