#!/usr/bin/env python3
"""Print the raw data used by the SWE-DIVE wastage report."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "Expert Nitpicker": {
        "pilot": "2026-08-26__20-15-44_expert_nitpicker_10runs",
        "case_study": "2026-08-25__11-40-30_expert_nitpicker",
    },
    "Mind Changer": {
        "pilot": "2026-08-26__20-16-04_mindchanger_10runs",
        "case_study": "2026-08-25__12-18-26_mindchanger",
    },
    "Vague Requester": {
        "pilot": "2026-08-26__21-10-41_vaguerequester_10turns",
        "case_study": "2026-08-25__11-57-13_vaguerequester",
    },
        "Base_mode":
    {
       "pilot": "2026-08-28__09-09-11_base_mode_10runs",
        "case_study": "2026-08-28__09-48-40_basemode",
    },
}
@dataclass(frozen=True)
class Trial:
    condition: str
    task: str
    resolved: bool
    target_passed: int
    target_total: int
    regressions: int
    regression_total: int
    turns: int
    input_tokens: int
    output_tokens: int
    files_read: int
    files_edited: int
    changed_files: tuple[str, ...]
    lines_deleted: int
    lines_added: int

    @property
    def passed_target_tests(self) -> bool:
        return self.target_total > 0 and self.target_passed == self.target_total

    @property
    def introduced_regression(self) -> bool:
        return self.regressions > 0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_activity(path: Path) -> tuple[int, int]:
    files = {"read": set(), "edit": set()}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        update = event.get("payload", {}).get("update", {})
        kind = update.get("kind")
        if kind in files:
            files[kind].update(
                location["path"]
                for location in update.get("locations", [])
                if location.get("path")
            )
    return len(files["read"]), len(files["edit"])


def diff_stats(diff: str) -> tuple[tuple[str, ...], int, int]:
    files = []
    deleted = 0
    added = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            files.append(line.split(" b/", 1)[1])
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return tuple(files), deleted, added


def load_trial(condition: str, conversation_path: Path) -> Trial:
    trial_dir = conversation_path.parent.parent
    conversation = read_json(conversation_path)
    result = read_json(trial_dir / "result.json")
    task = result["task_id"]["name"]
    report = read_json(trial_dir / "verifier" / "report.json")[task]
    tests = report["tests_status"]
    target = tests["FAIL_TO_PASS"]
    regression = tests["PASS_TO_PASS"]
    usage = conversation.get("agent_usage") or {}
    files_read, files_edited = file_activity(
        trial_dir / "agent" / "acp-events.jsonl"
    )
    changed_files, lines_deleted, lines_added = diff_stats(
        conversation.get("git_diff", "")
    )
    turns = sum(
        bool(
            response.get("agent_response")
            or response.get("agent_changes")
            or response.get("tool_calls")
        )
        for response in conversation["sim_response"]
    )
    return Trial(
        condition=condition,
        task=task,
        resolved=bool(report["resolved"]),
        target_passed=len(target["success"]),
        target_total=len(target["success"]) + len(target["failure"]),
        regressions=len(regression["failure"]),
        regression_total=len(regression["success"]) + len(regression["failure"]),
        turns=turns,
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        files_read=files_read,
        files_edited=files_edited,
        changed_files=changed_files,
        lines_deleted=lines_deleted,
        lines_added=lines_added,
    )


def mean(trials: list[Trial], field: str) -> float:
    return statistics.fmean(getattr(trial, field) for trial in trials)


def stdev(trials: list[Trial], field: str) -> float:
    values = [getattr(trial, field) for trial in trials]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def print_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(value) for value in row))
    print()


def pilot_rewards(job_dir: Path) -> dict[str, bool]:
    result = read_json(job_dir / "result.json")
    evals = result["stats"]["evals"]
    if len(evals) != 1:
        raise ValueError(f"Expected one evaluation in {job_dir / 'result.json'}")
    reward_stats = next(iter(evals.values()))["reward_stats"]["reward"]
    rewards = {
        trial_name: bool(float(reward))
        for reward, trial_names in reward_stats.items()
        for trial_name in trial_names
    }
    if len(rewards) != result["n_total_trials"]:
        raise ValueError(f"Incomplete reward data in {job_dir / 'result.json'}")
    return rewards


def main() -> None:
    trials = [
        load_trial(condition, conversation_path)
        for condition, jobs in JOBS.items()
        for job in jobs.values()
        for conversation_path in sorted(
            (ROOT / "jobs" / job).glob("*/agent/conversation.json")
        )
    ]
    by_condition = {
        condition: [trial for trial in trials if trial.condition == condition]
        for condition in JOBS
    }
    for condition, condition_trials in by_condition.items():
        tasks = [trial.task for trial in condition_trials]
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"{condition} contains duplicate tasks")

    missing_trials: dict[str, list[str]] = {}
    summary_rows = []
    for condition, condition_trials in by_condition.items():
        jobs = JOBS[condition]
        rewards = pilot_rewards(ROOT / "jobs" / jobs["pilot"])
        pilot_trials = [
            trial
            for trial in condition_trials
            if trial.task != "astropy__astropy-12907"
        ]
        recorded_names = {
            conversation_path.parent.parent.name
            for conversation_path in (
                ROOT / "jobs" / jobs["pilot"]
            ).glob("*/agent/conversation.json")
        }
        missing = sorted(set(rewards) - recorded_names)
        missing_trials[condition] = missing
        if any(not rewards[trial_name] for trial_name in missing):
            raise ValueError(
                f"Cannot infer test outcomes for unresolved missing trials: {missing}"
            )
        case_study = next(
            trial
            for trial in condition_trials
            if trial.task == "astropy__astropy-12907"
        )
        total = len(rewards) + 1
        if total != 11:
            raise ValueError(f"{condition} has {total} tasks instead of 11")
        summary_rows.append(
            (
                condition,
                f"{sum(rewards.values()) + case_study.resolved}/{total}",
                (
                    f"{sum(t.passed_target_tests for t in pilot_trials) + len(missing) + case_study.passed_target_tests}"
                    f"/{total}"
                ),
                (
                    f"{sum(t.introduced_regression for t in pilot_trials) + case_study.introduced_regression}"
                    f"/{total}"
                ),
                f"{mean(condition_trials, 'turns'):.2f} ± {stdev(condition_trials, 'turns'):.2f}",
                f"{len(condition_trials)}/{total}",
            )
        )

    print("SUMMARY")
    print_table(
        (
            "condition",
            "resolved",
            "all_target_tests_passed",
            "introduced_regression",
            "mean_std_turns_per_recorded_instance",
            "instances_with_turn_data",
        ),
        summary_rows,
    )

    print("RESOURCE MEANS")
    print_table(
        (
            "condition",
            "instances_with_resource_data",
            "mean_std_input_tokens_per_recorded_instance",
            "mean_std_output_tokens_per_recorded_instance",
            "mean_std_distinct_files_read_per_recorded_instance",
            "mean_std_distinct_files_edited_per_recorded_instance",
        ),
        [
            (
                condition,
                len(rows),
                f"{mean(rows, 'input_tokens'):.1f} ± {stdev(rows, 'input_tokens'):.1f}",
                f"{mean(rows, 'output_tokens'):.1f} ± {stdev(rows, 'output_tokens'):.1f}",
                f"{mean(rows, 'files_read'):.2f} ± {stdev(rows, 'files_read'):.2f}",
                f"{mean(rows, 'files_edited'):.2f} ± {stdev(rows, 'files_edited'):.2f}",
            )
            for condition, rows in by_condition.items()
        ],
    )

    base_condition = "Base_mode"
    base_rows = by_condition[base_condition]
    print("DELTA VS BASE_MODE (mean, no std)")
    print_table(
        (
            "condition",
            "delta_mean_turns",
            "delta_mean_input_tokens",
            "delta_mean_output_tokens",
            "delta_mean_files_read",
            "delta_mean_files_edited",
        ),
        [
            (
                condition,
                f"{mean(rows, 'turns') - mean(base_rows, 'turns'):+.2f}",
                f"{mean(rows, 'input_tokens') - mean(base_rows, 'input_tokens'):+.1f}",
                f"{mean(rows, 'output_tokens') - mean(base_rows, 'output_tokens'):+.1f}",
                f"{mean(rows, 'files_read') - mean(base_rows, 'files_read'):+.2f}",
                f"{mean(rows, 'files_edited') - mean(base_rows, 'files_edited'):+.2f}",
            )
            for condition, rows in by_condition.items()
            if condition != base_condition
        ],
    )

    print("DETAILED VERIFIER RESULTS")
    print_table(
        (
            "condition",
            "task",
            "resolved",
            "FAIL_TO_PASS",
            "PASS_TO_FAIL",
            "turns",
            "input_tokens",
            "output_tokens",
            "files_read",
            "files_edited",
        ),
        [
            (
                trial.condition,
                trial.task,
                str(trial.resolved).lower(),
                f"{trial.target_passed}/{trial.target_total}",
                f"{trial.regressions}/{trial.regression_total}",
                trial.turns,
                trial.input_tokens,
                trial.output_tokens,
                trial.files_read,
                trial.files_edited,
            )
            for trial in trials
        ],
    )

    print("TRIALS WITH AGGREGATE REWARD ONLY")
    print_table(
        ("condition", "trial", "resolved"),
        [
            (condition, trial_name, "true")
            for condition, trial_names in missing_trials.items()
            for trial_name in trial_names
        ],
    )

    print("ASTROPY 12907 FIGURE DATA")
    print_table(
        (
            "condition",
            "resolved",
            "turns",
            "full_diff_changed_files",
            "full_diff_lines",
            "file_names",
        ),
        [
            (
                trial.condition,
                str(trial.resolved).lower(),
                trial.turns,
                len(trial.changed_files),
                f"-{trial.lines_deleted}/+{trial.lines_added}",
                ",".join(trial.changed_files),
            )
            for trial in trials
            if trial.task == "astropy__astropy-12907"
        ],
    )

    print("DEFINITIONS")
    print("Summary outcome denominators include 10 pilot tasks plus Astropy.")
    print("Turn and resource means include only instances with detailed artifacts.")
    print("Token counts: conversation.json agent_usage input/output fields.")
    print("File counts are absolute distinct ACP locations, not paper deltas.")
    print("Regressions: failures in the verifier PASS_TO_PASS set.")
    print("Diff statistics: complete final conversation.json git_diff.")
    print("Paper token/file deltas need baseline and ground-truth data not on submission.")


if __name__ == "__main__":
    main()
