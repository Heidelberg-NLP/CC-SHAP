"""Load the first ``num_samples`` examples of a task into a uniform ``Example`` list.

The ComVE and e-SNLI CSVs are shuffled with a fixed pandas seed; BBH is shuffled with
the global ``random`` (so it participates in the ``CCSHAP_SEED`` determinism the harness
relies on). ``wrong_answer`` is drawn with ``random.choice`` for BBH and e-SNLI,
matching the original script's RNG consumption order.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import BBH_TASKS, DATA, LABELS
from .prompts import Prompts

_ESNLI_LABEL = {"entailment": "A", "contradiction": "B", "neutral": "C"}


@dataclass
class Example:
    formatted_input: str
    correct_answer: str
    wrong_answer: str
    sent0: str = ""  # ComVE sentence pair, used by the input-from-explanation test
    sent1: str = ""


def load_examples(task: str, prompts: Prompts, num_samples: int) -> list[Example]:
    if task == "comve":
        return _load_comve(prompts, num_samples)
    if task in BBH_TASKS:
        return _load_bbh(task, num_samples)
    if task == "esnli":
        return _load_esnli(task, prompts, num_samples)
    raise ValueError(f"Unknown task {task!r}")


def _load_comve(prompts: Prompts, num_samples: int) -> list[Example]:
    data = pd.read_csv(DATA["comve"][0]).sample(frac=1, random_state=42)
    gold = pd.read_csv(DATA["comve"][1], header=None, names=["id", "answer"])

    examples: list[Example] = []
    for idx, sent0, sent1 in zip(data["id"], data["sent0"], data["sent1"]):
        if len(examples) >= num_samples:
            break
        gold_answer = gold[gold["id"] == idx]["answer"].values[0]
        correct = "A" if gold_answer == 0 else "B"
        wrong = "A" if gold_answer == 1 else "B"
        examples.append(Example(prompts.example_comve(sent0, sent1),
                                correct, wrong, sent0, sent1))
    return examples


def _load_bbh(task: str, num_samples: int) -> list[Example]:
    data = json.loads(Path(DATA["bbh"].format(task=task)).read_text())["data"]
    random.shuffle(data)

    examples: list[Example] = []
    for row in data:
        if len(examples) >= num_samples:
            break
        correct = LABELS[task][row["multiple_choice_scores"].index(1)]
        wrong = random.choice([x for x in LABELS[task] if x != correct])
        examples.append(Example(row["parsed_inputs"] + ".", correct, wrong))
    return examples


def _load_esnli(task: str, prompts: Prompts, num_samples: int) -> list[Example]:
    data = pd.read_csv(DATA["esnli"]).sample(frac=1, random_state=42)

    examples: list[Example] = []
    for gold_label, sent0, sent1 in zip(data["gold_label"],
                                        data["Sentence1"], data["Sentence2"]):
        if len(examples) >= num_samples:
            break
        correct = _ESNLI_LABEL[gold_label]
        wrong = random.choice([x for x in LABELS[task] if x != correct])
        examples.append(Example(prompts.example_esnli(sent0, sent1), correct, wrong))
    return examples
