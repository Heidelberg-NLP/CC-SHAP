"""Run the full faithfulness suite over one task, record per-sample JSON, and report.

Per sample we measure model accuracy (direct and via CoT), CC-SHAP self-consistency
(post-hoc and CoT), and the four prior-work consistency tests, then aggregate to the
percentages the paper reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .cc_shap import CCShap, CCShapResult
from .config import LABELS, TESTS
from .consistency_tests import ConsistencyTests, LanhamResult
from .data import Example, load_examples
from .model import LanguageModel
from .prompts import Prompts

RESULTS_DIR = Path("results_json")
_ZERO_SCORE = CCShapResult(0, 0, 0, 0, 0, 0, plot_info=0)
_ZERO_LANHAM = LanhamResult(0, 0, 0, 0)


@dataclass
class SampleOutput:
    ask_input: str
    prediction: str
    ask_for_final_answer: str
    prediction_cot: str
    atanasova_counterfact: int
    atanasova_input_from_expl: int
    turpin: int
    lanham: LanhamResult
    post_hoc: CCShapResult
    cot: CCShapResult


@dataclass
class Totals:
    accuracy: int = 0
    accuracy_cot: int = 0
    atanasova_counterfact: int = 0
    atanasova_input_from_expl: int = 0
    turpin: int = 0
    lanham_early: int = 0
    lanham_mistake: int = 0
    lanham_paraphrase: int = 0
    lanham_filler: int = 0
    cc_shap_post_hoc: float = 0.0
    cc_shap_cot: float = 0.0


def _evaluate_example(ccshap: CCShap, tests: ConsistencyTests, task: str,
                      example: Example) -> SampleOutput:
    model, prompts, labels = ccshap.model, ccshap.prompts, tests.labels

    ask_input = prompts.answer(example.formatted_input)
    prediction = model.classify(ask_input, labels)
    cot_prompt = prompts.cot(example.formatted_input)
    generated_cot = model.generate(cot_prompt, repeat_input=True)
    ask_for_final_answer = prompts.final_answer(generated_cot)
    prediction_cot = model.classify(ask_for_final_answer, labels)

    counterfact = (tests.atanasova_counterfactual(example.formatted_input, prediction)
                   if "atanasova_counterfactual" in TESTS else 0)
    if "atanasova_input_from_expl" in TESTS and task == "comve":
        input_from_expl = tests.atanasova_input_from_expl(
            example.sent0, example.sent1, prediction, example.correct_answer)
    else:
        input_from_expl = 0
    post_hoc = (ccshap.measure(example.formatted_input, labels, "post_hoc")
                if "cc_shap-posthoc" in TESTS else _ZERO_SCORE)
    turpin = (tests.turpin(example.formatted_input, prediction_cot,
                           example.correct_answer, example.wrong_answer)
              if "turpin" in TESTS else 0)
    lanham = (tests.lanham(prediction_cot, generated_cot, cot_prompt)
              if "lanham" in TESTS else _ZERO_LANHAM)
    cot = (ccshap.measure(example.formatted_input, labels, "cot")
           if "cc_shap-cot" in TESTS else _ZERO_SCORE)

    return SampleOutput(ask_input, prediction, ask_for_final_answer, prediction_cot,
                        counterfact, input_from_expl, turpin, lanham, post_hoc, cot)


def _accumulate(totals: Totals, example: Example, out: SampleOutput) -> None:
    totals.accuracy += int(out.prediction == example.correct_answer)
    totals.accuracy_cot += int(out.prediction_cot == example.correct_answer)
    totals.atanasova_counterfact += out.atanasova_counterfact
    totals.atanasova_input_from_expl += out.atanasova_input_from_expl
    totals.turpin += out.turpin
    totals.lanham_early += out.lanham.early_answering
    totals.lanham_mistake += out.lanham.adding_mistakes
    totals.lanham_paraphrase += out.lanham.paraphrasing
    totals.lanham_filler += out.lanham.filler_tokens
    totals.cc_shap_post_hoc += out.post_hoc.score
    totals.cc_shap_cot += out.cot.score


def _other_measures(result: CCShapResult) -> dict:
    return {"dist_correl": f"{result.dist_correl:.2f}", "mse": f"{result.mse:.2f}",
            "var": f"{result.var:.2f}", "kl_div": f"{result.kl_div:.2f}",
            "js_div": f"{result.js_div:.2f}"}


def _record(example: Example, out: SampleOutput, totals: Totals) -> dict:
    return {
        "input": example.formatted_input,
        "correct_answer": example.correct_answer,
        "model_input": out.ask_input,
        "model_prediction": out.prediction,
        "model_input_cot": out.ask_for_final_answer,
        "model_prediction_cot": out.prediction_cot,
        "accuracy": totals.accuracy,
        "accuracy_cot": totals.accuracy_cot,
        "atanasova_counterfact": out.atanasova_counterfact,
        "atanasova_input_from_expl": out.atanasova_input_from_expl,
        "cc_shap-posthoc": f"{out.post_hoc.score:.2f}",
        "turpin": out.turpin,
        "lanham_early": out.lanham.early_answering,
        "lanham_mistake": out.lanham.adding_mistakes,
        "lanham_paraphrase": out.lanham.paraphrasing,
        "lanham_filler": out.lanham.filler_tokens,
        "cc_shap-cot": f"{out.cot.score:.2f}",
        "other_measures_post_hoc": _other_measures(out.post_hoc),
        "other_measures_cot": _other_measures(out.cot),
        "shap_plot_info_post_hoc": out.post_hoc.plot_info,
        "shap_plot_info_cot": out.cot.plot_info,
    }


def _report(task: str, model_name: str, totals: Totals, count: int) -> None:
    def pct(value: float) -> float:
        return value * 100 / count

    print(f"Ran {list(TESTS)} on {task} data with model {model_name}. "
          f"Reporting accuracy and faithfulness percentage.\n")
    print(f"Accuracy %                  : {pct(totals.accuracy):.2f}")
    print(f"Atanasova Counterfact %     : {pct(totals.atanasova_counterfact):.2f}")
    print(f"Atanasova Input from Expl % : {pct(totals.atanasova_input_from_expl):.2f}")
    print(f"CC-SHAP post-hoc mean score : {totals.cc_shap_post_hoc / count:.2f}")
    print(f"Accuracy CoT %              : {pct(totals.accuracy_cot):.2f}")
    print(f"Turpin %                    : {pct(totals.turpin):.2f}")
    print(f"Lanham Early Answering %    : {pct(totals.lanham_early):.2f}")
    print(f"Lanham Filler %             : {pct(totals.lanham_filler):.2f}")
    print(f"Lanham Mistake %            : {pct(totals.lanham_mistake):.2f}")
    print(f"Lanham Paraphrase %         : {pct(totals.lanham_paraphrase):.2f}")
    print(f"CC-SHAP CoT mean score      : {totals.cc_shap_cot / count:.2f}")


def run(model: LanguageModel, helper: LanguageModel, nlp, task: str,
        num_samples: int, write: bool = True) -> None:
    labels = LABELS[task]
    prompts = Prompts(model.name, task)
    ccshap = CCShap(model, prompts)
    tests = ConsistencyTests(model, helper, prompts, nlp, labels)

    examples = load_examples(task, prompts, num_samples)

    totals = Totals()
    results = {}
    for k, example in enumerate(tqdm(examples)):
        out = _evaluate_example(ccshap, tests, task, example)
        _accumulate(totals, example, out)
        results[f"{task}_{model.name}_{k}"] = _record(example, out, totals)

    count = len(examples)
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{task}_{model.name}_{count}.json").write_text(
            json.dumps(results))
    _report(task, model.name, totals, count)
