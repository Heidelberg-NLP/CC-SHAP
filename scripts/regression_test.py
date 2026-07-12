"""Before/after regression harness for CC-SHAP.

Runs the same fixed sample through both dependency stacks:
  * "before" : the original Python 3.11 conda env (`ccshap-before`, via micromamba)
  * "after"  : the modernized Python 3.11 uv env (`.venv`)

Both stacks share the *same* vendored `shap/` source, so this isolates the effect of
the dependency upgrade (torch 2.1->2.5, transformers 4.35->4.4x). Runs are seeded via
CCSHAP_SEED and use the target model as its own Lanham helper (CCSHAP_HELPER_MODEL) so
the comparison stays small and token-free.

The "before" stack is slow and rarely changes, so its output is recorded on disk under
regression_results/before/ and reused: run once with ``--env before``, then ``--env
after`` compares the current code against the recorded baseline without re-running it.

Wall-clock time per stack is logged (regression_results/<stack>/<key>.time) to inform
the speed optimizations planned as a last step.

Usage:
    uv run python scripts/regression_test.py --task comve --model gpt2 --env before
    uv run python scripts/regression_test.py --task comve --model gpt2  # vs recorded
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MICROMAMBA = REPO / "bin" / "micromamba"
MAMBA_ROOT = REPO / ".micromamba"
BEFORE_ENV = "ccshap-before"
RESULTS = REPO / "regression_results"

SKIP_SUBTREES = {"shap_plot_info_post_hoc", "shap_plot_info_cot"}

# The verdict gates on the headline CC-SHAP scores and the discrete test outcomes. The
# `other_measures_*` (mse/var/kl_div/...) are auxiliary distance metrics on the raw SHAP
# ratio vectors -- numerically volatile (var is a sum of 4th powers) and meaningless to
# threshold across fp16 CUDA/torch versions -- so they are reported but not gated.
def is_informational(field: str) -> bool:
    return field.startswith("other_measures")


def key(task: str, model: str) -> str:
    return f"{task}_{model}"


def result_json_path(task: str, model: str, num: int) -> Path:
    return REPO / "results_json" / f"{task}_{model}_{num}.json"


def saved_paths(stack: str, task: str, model: str) -> tuple[Path, Path]:
    base = RESULTS / stack / key(task, model)
    return base.with_suffix(".json"), base.with_suffix(".time")


def run_stack(stack: str, task: str, model: str, num: int,
              seed: int) -> tuple[dict, float]:
    args = ["faithfulness.py", task, model, str(num)]
    env = dict(os.environ, CCSHAP_SEED=str(seed), CCSHAP_HELPER_MODEL=model)
    if stack == "before":
        env["MAMBA_ROOT_PREFIX"] = str(MAMBA_ROOT)
        cmd = [str(MICROMAMBA), "run", "-n", BEFORE_ENV, "python", *args]
    else:
        cmd = ["uv", "run", "python", *args]

    out_path = result_json_path(task, model, num)
    out_path.unlink(missing_ok=True)

    print(f"\n=== [{stack}] {' '.join(cmd)} ===", flush=True)
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start

    summary = [ln for ln in proc.stdout.splitlines() if "%" in ln or "mean score" in ln]
    print("\n".join(summary) if summary else proc.stdout[-500:])
    print(f"[{stack}] wall-clock: {elapsed:.1f}s")
    if proc.returncode != 0:
        print(f"[{stack}] STDERR tail:\n{proc.stderr[-2000:]}", file=sys.stderr)
        raise SystemExit(f"[{stack}] run failed (exit {proc.returncode})")
    if not out_path.exists():
        raise SystemExit(f"[{stack}] produced no result json at {out_path}")

    json_dest, time_dest = saved_paths(stack, task, model)
    json_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, json_dest)
    time_dest.write_text(f"{elapsed:.3f}\n")
    return json.loads(out_path.read_text()), elapsed


def flatten(entry: dict, prefix: str = "") -> tuple[dict[str, float], dict[str, str]]:
    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {}
    for field, value in entry.items():
        if field in SKIP_SUBTREES:
            continue
        name = f"{prefix}{field}"
        if isinstance(value, dict):
            sub_num, sub_cat = flatten(value, f"{name}.")
            numeric.update(sub_num)
            categorical.update(sub_cat)
        elif isinstance(value, (int, float)):  # bool is an int subclass, handled here
            numeric[name] = float(value)
        elif isinstance(value, str):
            try:
                numeric[name] = float(value)
            except ValueError:
                categorical[name] = value
    return numeric, categorical


def collect_diffs(before: dict, after: dict, shared: list[str]):
    """Per-field absolute numeric diffs, plus categorical (prediction) agreement."""
    per_field: dict[str, list[float]] = {}
    categorical_total = categorical_mismatch = 0
    for k in shared:
        bn, bc = flatten(before[k])
        an, ac = flatten(after[k])
        for field, value in bn.items():
            if field in an:
                per_field.setdefault(field, []).append(abs(value - an[field]))
        for field, value in bc.items():
            if field in ac:
                categorical_total += 1
                categorical_mismatch += int(value != ac[field])
    return per_field, categorical_total, categorical_mismatch


def compare(before: dict, after: dict, tol: float) -> bool:
    shared = [k for k in before if k in after]
    print("\n================ comparison ================")
    print(f"samples compared: {len(shared)}")

    per_field, cat_total, cat_mismatch = collect_diffs(before, after, shared)
    if not per_field:
        raise SystemExit("no comparable numeric fields (did both runs write results?)")

    worst = 0.0
    worst_info = 0.0
    for field in sorted(per_field):
        diffs = per_field[field]
        mx = max(diffs)
        info = is_informational(field)
        if info:
            worst_info = max(worst_info, mx)
            flag = "info"
        else:
            worst = max(worst, mx)
            flag = "OK" if mx <= tol else "!!"
        print(f"  {flag:5s}{field:36s} n={len(diffs):3d}  "
              f"max|Δ|={mx:.3e}  mean|Δ|={sum(diffs) / len(diffs):.3e}")

    if cat_total:
        print(f"\ncategorical (predictions/labels) agreement: "
              f"{cat_total - cat_mismatch}/{cat_total} (informational)")
    print(f"auxiliary other_measures worst max|Δ| = {worst_info:.3e} (informational)")
    print("--------------------------------------------")
    verdict = "PASS" if worst <= tol else "FAIL"
    print(f"primary worst max|Δ| = {worst:.3e}  (tol={tol:.1e})  ->  {verdict}")
    return worst <= tol


def load_recorded(stack: str, task: str, model: str) -> dict:
    json_path, _ = saved_paths(stack, task, model)
    if not json_path.exists():
        raise SystemExit(f"no recorded {stack} baseline at {json_path}; "
                         f"run with --env {stack} first")
    return json.loads(json_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="comve")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-1,
                    help="max allowed abs diff on numeric fields (fp16 GPU parity)")
    ap.add_argument("--env", choices=["both", "before", "after"], default="after",
                    help="'before' records the baseline; 'after' compares against it")
    args = ap.parse_args()

    if args.env == "before":
        run_stack("before", args.task, args.model, args.num, args.seed)
        return

    if args.env == "both":
        before, _ = run_stack("before", args.task, args.model, args.num, args.seed)
    else:
        before = load_recorded("before", args.task, args.model)

    after, _ = run_stack("after", args.task, args.model, args.num, args.seed)
    ok = compare(before, after, args.tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
