# CC-SHAP

This is the implementation of the paper "On Measuring Faithfulness or Self-Consistency of Natural Language Explanations" https://arxiv.org/abs/2311.07466 accepted at ACL 2024!

CC-SHAP compares a model's input contributions to its **answer** against its input
contributions to its **explanation** (via Shapley values); the more the two agree, the
more self-consistent the model. `faithfulness.py` runs CC-SHAP alongside five prior
self-consistency tests (Atanasova et al. counterfactual & input-from-explanation,
Turpin et al. biasing, Lanham et al. CoT corruptions) over three datasets.

## Setup

This repo has been modernized to run on a current Python 3.11 stack managed with
[uv](https://docs.astral.sh/uv/), while keeping a reproduction of the original stack for
regression testing (see [Before/after regression](#beforeafter-regression)).

```bash
uv sync                                     # create .venv from pyproject.toml
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Create a `.env` in the repo root (it is gitignored) for HuggingFace access and cache
location:

```dotenv
HF_TOKEN=hf_...                             # required for gated models (Llama-2, ...)
HF_CACHE=/path/with/space/for/model/weights # sets HF_HOME so weights don't fill $HOME
```

`.env` is loaded automatically by `faithfulness.py` (via `ccshap_repro.load_env`) before
any HuggingFace import, so the token and cache directory take effect.

## Models

Models are selected by the second CLI argument to `faithfulness.py` and downloaded from
the HuggingFace Hub on first use. The keys map to Hub ids in the `MODELS` dict.

| `model` arg | Hub id | Notes |
| --- | --- | --- |
| `gpt2` | `gpt2` | tiny, ungated — used for smoke tests and the regression harness |
| `mistral-7b` / `mistral-7b-chat` | `mistralai/Mistral-7B-v0.1` / `-Instruct-v0.1` | ungated, fits one RTX 3090 in fp16 |
| `falcon-7b` / `falcon-7b-chat` | `tiiuae/falcon-7b` / `-instruct` | ungated |
| `llama2-7b` / `-13b` (+ `-chat`) | `meta-llama/Llama-2-*` | **gated** — needs `HF_TOKEN` |
| `bloom-7b1`, `opt-30b`, `falcon-40b`, ... | see `MODELS` | 30B/40B need >48 GB (multi-GPU / offload) |

The Lanham test paraphrases / edits the CoT with a **helper model** (the paper uses
`llama2-13b-chat`). Override it with `CCSHAP_HELPER_MODEL=<key>`; the regression harness
sets it to the run's own model so the smoke test stays small and token-free.

## Datasets

The paper clones three full repositories. For smoke tests and regression we only need a
handful of examples, fetched by `scripts/prepare_sample.py` into `data/`:

```bash
uv run python scripts/prepare_sample.py --num 10
```

This writes reduced files that the `DATA` dict in `faithfulness.py` points at:

- `data/comve/subtaskA_{test_data,gold_answers}.csv` (ComVE / SemEval-2020 Task 4)
- `data/e-SNLI/esnli_test.csv` (e-SNLI)
- `data/bbh/<task>/val_data.json` (`causal_judgment`, `disambiguation_qa`, `logical_deduction_five_objects`)

For the full benchmark, clone the source repos
([e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI),
[ComVE](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation),
[BBH samples](https://github.com/milesaturpin/cot-unfaithfulness)) and edit the `DATA`
dict in `faithfulness.py` to point at them.

## Usage

Run from the repository root (so `import shap` resolves to the vendored `shap/`):

```bash
uv run python faithfulness.py <task> <model> <num_samples>
# e.g.
uv run python faithfulness.py comve gpt2 10
uv run python faithfulness.py esnli mistral-7b-chat 100
```

`<task>` is one of `comve`, `esnli`, `causal_judgment`, `disambiguation_qa`,
`logical_deduction_five_objects`. Results are written to `results_json/`.

## Before/after regression

Two environments run the *same* vendored `shap/` so we can check that the dependency
upgrade preserves behavior:

- **before**: the original stack (Python 3.11, torch 2.1, transformers 4.35) in a
  micromamba env, created from `environment.before.yml`.
- **after**: the modernized uv env (`.venv`).

```bash
# one-time: create the legacy env
MAMBA_ROOT_PREFIX="$PWD/.micromamba" ./bin/micromamba env create -y -f environment.before.yml

# record the (slow) before baseline once, then compare the after stack against it
uv run python scripts/regression_test.py --task comve --model gpt2 --num 2 --env before
uv run python scripts/regression_test.py --task comve --model gpt2 --num 2
```

Runs are seeded via `CCSHAP_SEED`, and wall-clock time per stack is logged under
`regression_results/<stack>/*.time`. `scripts/refactor_check.py` similarly guards code
refactors against a golden baseline captured in the same env.

## Cite
```bibtex
@article{parcalabescu2023measuring,
  title={On measuring faithfulness or self-consistency of natural language explanations},
  author={Parcalabescu, Letitia and Frank, Anette},
  journal={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
  year={2024},
  url      = {https://arxiv.org/abs/2311.07466},
}
```

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .
