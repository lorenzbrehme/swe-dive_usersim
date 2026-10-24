"""SWE-Bench → Harbor task adapter (PoC implementation).

Reads SWE-Bench instance rows (the ``rows`` format produced by
``load_swe-bench_data.ipynb`` / the Hugging Face dataset API) and emits one
Harbor task directory per instance:

    <out_dir>/<instance_id>/
        task.toml          — environment image, resources, metadata
        instruction.md     — the problem statement shown to the agent
        environment/       — empty dir (docker_image set in task.toml)
        tests/
            test.sh        — verifier: applies test patch, runs pytest, writes reward
            test_patch.diff
            eval_spec.json — FAIL_TO_PASS, PASS_TO_PASS, base_commit, test_files
            parse_report.py — standalone pytest-output scorer

Usage::

    python -m swe_dive.adapters.swebench \\
        --input ../swe_bench_test.json \\
        --out   swe_dive/datasets/swebench-verified-poc \\
        --indices 90

Known limitation: the generic verifier uses ``python -m pytest`` and works for
pytest-based repos (astropy, sympy, …).  Repos with custom test runners (e.g.
Django) require repo-specific ``test.sh`` overrides; a future improvement is to
call ``swebench.make_test_spec`` when the ``swebench`` package is available.
"""

import argparse
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("swe_dive.adapters.swebench")


# ---------------------------------------------------------------------------
# Image naming
# ---------------------------------------------------------------------------

def instance_image_key(instance_id: str) -> str:
    """Map SWE-Bench instance id → Docker image name component.

    ``__`` is replaced with ``_1776_`` to produce a valid Docker image name
    (double-underscore is not allowed in image name components).
    """
    return instance_id.lower().replace("__", "_1776_")


def docker_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_image_key(instance_id)}:latest"


# ---------------------------------------------------------------------------
# Test-file extraction from a unified diff
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def test_files_from_patch(test_patch: str) -> list[str]:
    """Return the list of files modified by the test patch."""
    return _DIFF_FILE_RE.findall(test_patch)


# ---------------------------------------------------------------------------
# Verifier templates
# ---------------------------------------------------------------------------

_TEST_SH = textwrap.dedent("""\
    #!/usr/bin/env bash
    # SWE-Bench verifier: apply test patch, run pytest, score with parse_report.py
    set -uo pipefail

    mkdir -p /logs/verifier /logs/agent

    cd /testbed
    git config --global --add safe.directory /testbed 2>/dev/null || true

    # Activate the testbed conda environment if available.
    if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate testbed 2>/dev/null || true
    fi

    # Reset test files to base_commit so the test patch applies cleanly.
    BASE_COMMIT=$(python3 -c "import json; print(json.load(open('/tests/eval_spec.json'))['base_commit'])")
    TEST_FILES=$(python3 -c "import json; print(' '.join(json.load(open('/tests/eval_spec.json'))['test_files']))")
    if [ -n "$TEST_FILES" ]; then
        # shellcheck disable=SC2086
        git checkout "$BASE_COMMIT" -- $TEST_FILES 2>/dev/null || true
    fi

    # Apply the test patch.
    if ! git apply -v /tests/test_patch.diff 2>&1; then
        if ! git apply --3way /tests/test_patch.diff 2>&1; then
            echo "ERROR: failed to apply test patch" >&2
            echo 0 > /logs/verifier/reward.txt
            exit 0
        fi
    fi

    # Collect test node ids (FAIL_TO_PASS + PASS_TO_PASS).
    NODE_IDS=$(python3 -c "
import json
spec = json.load(open('/tests/eval_spec.json'))
ids = json.loads(spec['fail_to_pass']) + json.loads(spec['pass_to_pass'])
print(' '.join(ids))
")

    # Run pytest.
    # shellcheck disable=SC2086
    python3 -m pytest -rA -p no:cacheprovider --tb=no -q $NODE_IDS \\
        2>&1 | tee /logs/agent/test_output.txt

    # Score and write reward.
    python3 /tests/parse_report.py \\
        --eval-spec /tests/eval_spec.json \\
        --pytest-output /logs/agent/test_output.txt \\
        > /logs/verifier/reward.txt
    echo "reward: $(cat /logs/verifier/reward.txt)"
""")

_PARSE_REPORT_PY = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Score a pytest -rA run against SWE-Bench FAIL_TO_PASS / PASS_TO_PASS lists.

    Exits 0 in all cases; writes "1" (resolved) or "0" (unresolved) to stdout.
    \"\"\"

    import argparse
    import json
    import re
    import sys


    _SUMMARY_RE = re.compile(
        r"^(PASSED|FAILED|ERROR|SKIPPED)\\s+(\\S+)", re.MULTILINE
    )


    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--eval-spec", required=True)
        p.add_argument("--pytest-output", required=True)
        return p.parse_args()


    def load_ids(raw: str) -> list[str]:
        \"\"\"Handle both a JSON list string and a plain list.\"\"\"
        if isinstance(raw, list):
            return raw
        return json.loads(raw)


    def main():
        args = parse_args()
        with open(args.eval_spec) as f:
            spec = json.load(f)

        fail_to_pass = load_ids(spec["fail_to_pass"])
        pass_to_pass = load_ids(spec["pass_to_pass"])

        with open(args.pytest_output) as f:
            output = f.read()

        status: dict[str, str] = {}
        for match in _SUMMARY_RE.finditer(output):
            verdict, node_id = match.group(1), match.group(2)
            # Keep the worst outcome if a test appears multiple times.
            existing = status.get(node_id)
            if existing != "FAILED":
                status[node_id] = verdict

        passed = {nid for nid, v in status.items() if v == "PASSED"}

        f2p_ok = all(nid in passed for nid in fail_to_pass)
        p2p_ok = all(nid in passed for nid in pass_to_pass)

        print(1 if (f2p_ok and p2p_ok) else 0)


    if __name__ == "__main__":
        main()
""")

_TASK_TOML_TEMPLATE = """\
schema_version = "1.3"

[task]
name = "swebench/{instance_id}"
description = "{repo} \u2014 {instance_id}"
keywords = ["swebench", "python", "coding"]

[metadata]
repo = "{repo}"
base_commit = "{base_commit}"
version = "{version}"
difficulty = "{difficulty}"

[agent]
timeout_sec = 3600.0

[verifier]
timeout_sec = 1800.0

# Collect the conversation log and trajectory as named artifacts so they
# appear in `harbor view` and survive separate-verifier transfers.
artifacts = [
    "/logs/agent/conversation.json",
    "/logs/agent/trajectory.json",
]

[environment]
docker_image = "{docker_image}"
cpus = 2
memory_mb = 8192
"""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SWEBenchAdapter:
    """Generate Harbor task directories from a SWE-Bench JSON rows file."""

    def __init__(
        self,
        rows_path: str | Path,
        indices: list[int] | None = None,
        instance_ids: list[str] | None = None,
    ) -> None:
        self._rows_path = Path(rows_path)
        self._indices = set(indices) if indices else None
        self._instance_ids = set(instance_ids) if instance_ids else None

    def _load_rows(self) -> list[dict[str, Any]]:
        with open(self._rows_path) as f:
            data = json.load(f)
        # Support both {"rows": [...]} and a plain list.
        return data["rows"] if isinstance(data, dict) else data

    def iter_instances(self) -> Iterator[dict[str, Any]]:
        for row in self._load_rows():
            idx = row.get("row_idx", 0)
            instance = row.get("row", row)
            if self._indices is not None and idx not in self._indices:
                continue
            if self._instance_ids is not None:
                if instance.get("instance_id") not in self._instance_ids:
                    continue
            yield instance

    def emit_task(self, instance: dict[str, Any], out_dir: Path) -> Path:
        instance_id: str = instance["instance_id"]
        task_dir = out_dir / instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "environment").mkdir(exist_ok=True)
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # instruction.md
        (task_dir / "instruction.md").write_text(
            instance["problem_statement"], encoding="utf-8"
        )

        # task.toml
        (task_dir / "task.toml").write_text(
            _TASK_TOML_TEMPLATE.format(
                instance_id=instance_id,
                repo=instance["repo"],
                base_commit=instance["base_commit"],
                version=instance.get("version", ""),
                difficulty=instance.get("difficulty", ""),
                docker_image=docker_image(instance_id),
            ),
            encoding="utf-8",
        )

        # tests/test_patch.diff
        (tests_dir / "test_patch.diff").write_text(
            instance["test_patch"], encoding="utf-8"
        )

        # tests/eval_spec.json
        eval_spec = {
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "fail_to_pass": instance["FAIL_TO_PASS"],
            "pass_to_pass": instance["PASS_TO_PASS"],
            "test_files": test_files_from_patch(instance["test_patch"]),
        }
        (tests_dir / "eval_spec.json").write_text(
            json.dumps(eval_spec, indent=2), encoding="utf-8"
        )

        # tests/test.sh
        test_sh = tests_dir / "test.sh"
        test_sh.write_text(_TEST_SH, encoding="utf-8")
        test_sh.chmod(0o755)

        # tests/parse_report.py
        (tests_dir / "parse_report.py").write_text(
            _PARSE_REPORT_PY, encoding="utf-8"
        )

        return task_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Harbor task directories from a SWE-Bench rows JSON."
    )
    parser.add_argument("--input", required=True, help="Path to swe_bench rows JSON.")
    parser.add_argument("--out", required=True, help="Output directory for tasks.")
    parser.add_argument(
        "--indices", nargs="*", type=int,
        help="Select rows by row_idx (space-separated).",
    )
    parser.add_argument(
        "--instance-ids", nargs="*",
        help="Select rows by instance_id (space-separated).",
    )
    args = parser.parse_args(argv)

    adapter = SWEBenchAdapter(
        rows_path=args.input,
        indices=args.indices,
        instance_ids=args.instance_ids,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for instance in adapter.iter_instances():
        task_dir = adapter.emit_task(instance, out)
        logger.info("  emitted %s  →  %s", task_dir.name, task_dir)
        count += 1

    if count == 0:
        logger.error("No instances matched; check --indices / --instance-ids.")
        return 1
    logger.info("Done. %d task(s) written to %s", count, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
