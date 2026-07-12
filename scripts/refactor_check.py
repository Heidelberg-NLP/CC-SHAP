"""Verify a code change (e.g. a refactor) preserves CC-SHAP outputs.

Unlike ``regression_test.py`` (which compares the two *dependency stacks* running the
same code), this compares the *current code* against a golden baseline captured earlier,
in the same uv env with the same seed. A faithful refactor should reproduce the baseline
essentially bit-for-bit.

Workflow:
    # BEFORE changing code, capture the baseline:
    uv run python scripts/refactor_check.py --task comve --model gpt2 --capture
    # AFTER changing code, check it still matches:
    uv run python scripts/refactor_check.py --task comve --model gpt2 --check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_test import compare, key, run_stack

GOLDEN = Path(__file__).resolve().parent.parent / "regression_results" / "golden"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="comve")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="save the current output as the golden baseline")
    mode.add_argument("--check", action="store_true",
                      help="compare the current output against the golden baseline")
    args = ap.parse_args()

    result, _ = run_stack("after", args.task, args.model, args.num, args.seed)
    dest = GOLDEN / f"{key(args.task, args.model)}.json"

    if args.capture:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result))
        print(f"\ncaptured golden baseline -> {dest}")
        return

    if not dest.exists():
        raise SystemExit(f"no golden baseline at {dest}; run with --capture first")
    ok = compare(json.loads(dest.read_text()), result, args.tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
