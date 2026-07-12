"""Fetch small, self-contained samples of the CC-SHAP evaluation datasets.

The paper clones three full repositories (e-SNLI, ComVE / SemEval-2020 Task 4, and the
BBH subsets from cot-unfaithfulness). For smoke tests and the before/after regression
harness we only need a handful of examples, so this downloads the source files, applies
the *same* selection faithfulness.py uses (seeded shuffle, first ``--num`` rows), and
writes reduced files under ``data/`` that the ``DATA`` dict in faithfulness.py reads.

Usage (inside the uv env):
    uv run python scripts/prepare_sample.py --num 10
"""
import argparse
import io
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SHUFFLE_SEED = 42

ESNLI_URL = ("https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/"
             "dataset/esnli_test.csv")
COMVE_BASE = ("https://raw.githubusercontent.com/wangcunxiang/"
              "SemEval2020-Task4-Commonsense-Validation-and-Explanation/master/"
              "ALL data/Test Data")
BBH_BASE = ("https://raw.githubusercontent.com/milesaturpin/cot-unfaithfulness/main/"
            "data/bbh")
BBH_TASKS = ("causal_judgment", "disambiguation_qa", "logical_deduction_five_objects")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(urllib.parse.quote(url, safe=":/?=&%")) as response:
        return response.read()


def _report(path: Path, rows: int) -> None:
    print(f"wrote {path.relative_to(REPO)} ({rows} rows)")


def sample_esnli(num: int) -> None:
    data = pd.read_csv(io.BytesIO(_download(ESNLI_URL)))
    sample = data.sample(frac=1, random_state=SHUFFLE_SEED).head(num)
    sample = sample[["gold_label", "Sentence1", "Sentence2"]]
    out = DATA / "e-SNLI" / "esnli_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    _report(out, len(sample))


def sample_comve(num: int) -> None:
    test = pd.read_csv(io.BytesIO(_download(f"{COMVE_BASE}/subtaskA_test_data.csv")))
    gold = pd.read_csv(io.BytesIO(_download(f"{COMVE_BASE}/subtaskA_gold_answers.csv")),
                       header=None, names=["id", "answer"])
    sample = test.sample(frac=1, random_state=SHUFFLE_SEED).head(num)
    gold_sample = gold[gold["id"].isin(sample["id"])]

    out_dir = DATA / "comve"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "subtaskA_test_data.csv", index=False)
    gold_sample.to_csv(out_dir / "subtaskA_gold_answers.csv", index=False, header=False)
    _report(out_dir / "subtaskA_test_data.csv", len(sample))


def sample_bbh(task: str, num: int) -> None:
    payload = json.loads(_download(f"{BBH_BASE}/{task}/val_data.json"))
    rows = payload["data"]
    random.Random(SHUFFLE_SEED).shuffle(rows)
    payload["data"] = rows[:num]

    out = DATA / "bbh" / task / "val_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    _report(out, len(payload["data"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=10, help="examples per dataset/task")
    ap.add_argument("--tasks", nargs="+",
                    default=["comve", "esnli", *BBH_TASKS],
                    help="subset of comve, esnli, and BBH task names")
    args = ap.parse_args()

    for task in args.tasks:
        if task == "comve":
            sample_comve(args.num)
        elif task == "esnli":
            sample_esnli(args.num)
        elif task in BBH_TASKS:
            sample_bbh(task, args.num)
        else:
            raise SystemExit(f"unknown task {task!r}")


if __name__ == "__main__":
    main()
