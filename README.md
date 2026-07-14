# CC-SHAP

This is the implementation of the paper "On Measuring Faithfulness or Self-Consistency of Natural Language Explanations" https://arxiv.org/abs/2311.07466 accepted at ACL 2024!

CC-SHAP compares a model's input contributions to its **answer** against its input
contributions to its **explanation** (via Shapley values); the more the two agree, the
more self-consistent the model. `faithfulness.py` runs CC-SHAP alongside five prior
self-consistency tests (Atanasova et al. counterfactual & input-from-explanation,
Turpin et al. biasing, Lanham et al. CoT corruptions) over three datasets.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python 3.11):

```bash
uv sync                                     # create .venv from pyproject.toml
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

For HuggingFace access and a custom cache location, copy `.env.example` to `.env`
(gitignored) and fill it in:

```dotenv
HF_TOKEN=hf_...                             # required for gated models (Llama-2, ...)
HF_CACHE=/path/with/space/for/model/weights # sets HF_HOME so weights don't fill $HOME
```

`.env` is loaded automatically by `faithfulness.py` (via `hf_env.load_env`) before any
HuggingFace import, so the token and cache directory take effect.

## Models

Models are selected by the second CLI argument to `faithfulness.py` and downloaded from
the HuggingFace Hub on first use. The keys map to Hub ids in the `MODELS` dict.

| `model` arg | Hub id | Notes |
| --- | --- | --- |
| `gpt2` | `gpt2` | tiny, ungated — used for smoke tests |
| `mistral-7b` / `mistral-7b-chat` | `mistralai/Mistral-7B-v0.1` / `-Instruct-v0.1` | ungated, fits one RTX 3090 in fp16 |
| `falcon-7b` / `falcon-7b-chat` | `tiiuae/falcon-7b` / `-instruct` | ungated |
| `llama2-7b` / `-13b` (+ `-chat`) | `meta-llama/Llama-2-*` | **gated** — needs `HF_TOKEN` |
| `bloom-7b1`, `opt-30b`, `falcon-40b`, ... | see `MODELS` | 30B/40B need >48 GB (multi-GPU / offload) |

The Lanham test paraphrases / edits the CoT with a **helper model** (the paper uses
`llama2-13b-chat`). Override it with `--helper-model <key>`, e.g. set it to the run's own
model to keep a smoke test small and token-free.

## Datasets

The paper clones three full repositories. For smoke tests we only need a handful of
examples, fetched by `scripts/prepare_sample.py` into `data/`:

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
`logical_deduction_five_objects`. Results are written to `results_json/`. Add
`--helper-model <key>` to change the Lanham helper, or `--no-write` to skip the json.

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
