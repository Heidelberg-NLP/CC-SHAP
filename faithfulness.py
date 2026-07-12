"""Command-line entry point for CC-SHAP.

Runs the faithfulness suite for one model over one task (ComVE, e-SNLI, or BBH):
for each example it measures CC-SHAP self-consistency (post-hoc and CoT) plus the four
prior-work tests (Atanasova x2, Turpin, Lanham), then reports accuracy and per-test
faithfulness percentages and writes a per-sample result json. The method lives in the
``ccshap`` package; this file only parses arguments, seeds for determinism, builds the
model (and the Lanham helper model), and hands off to ``ccshap.evaluation.run``.

Must be run from the repository root so that ``import shap`` resolves to the vendored
``shap/`` package (a modified copy) rather than any pip-installed shap:

    uv run python faithfulness.py comve gpt2 10
    uv run python faithfulness.py esnli llama2-7b-chat 50
    uv run python faithfulness.py logical_deduction_five_objects mistral-7b 20

The Lanham test uses a large chat model to paraphrase / corrupt reasoning chains; it
defaults to ``llama2-13b-chat`` and can be overridden with ``CCSHAP_HELPER_MODEL`` (the
regression harness sets it to the model under test to stay small and token-free).
"""
import argparse
import random

from ccshap_repro import helper_model_key, load_env, maybe_seed

load_env()  # set HF token + cache dir before importing transformers/huggingface_hub

import spacy

from ccshap.config import DEFAULT_HELPER_MODEL
from ccshap.evaluation import run
from ccshap.model import LanguageModel


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="comve, esnli, or a BBH task name")
    ap.add_argument("model", help="model key from ccshap.config.MODELS, e.g. gpt2")
    ap.add_argument("num_samples", type=int)
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing the per-sample result json")
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_sm")
    random.seed(42)
    maybe_seed()  # also seed numpy/torch when CCSHAP_SEED is set (regression harness)

    model = LanguageModel.load(args.model)
    helper_name = helper_model_key(DEFAULT_HELPER_MODEL)
    helper = model if helper_name == args.model else LanguageModel.load(helper_name)

    run(model, helper, nlp, args.task, args.num_samples, write=not args.no_write)


if __name__ == "__main__":
    main()
