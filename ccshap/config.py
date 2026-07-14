"""Static configuration for CC-SHAP: model registry, task label sets, data paths.

Dataset paths point at the reduced samples written by ``scripts/prepare_sample.py``;
clone the full datasets (see README) and edit ``DATA`` to reproduce the paper.
"""

MAX_NEW_TOKENS = 100

# short name -> HuggingFace id (or local path for the paper's private checkpoints)
MODELS = {
    "bloom-7b1": "bigscience/bloom-7b1",
    "opt-30b": "facebook/opt-30b",
    "llama30b": "/workspace/mitarb/parcalabescu/llama30b_hf",
    "oasst-sft-6-llama-30b": "/workspace/mitarb/parcalabescu/transformers-xor_env/"
                             "oasst-sft-6-llama-30b-xor/oasst-sft-6-llama-30b",
    "gpt2": "gpt2",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-chat": "mistralai/Mistral-7B-Instruct-v0.1",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-7b-chat": "tiiuae/falcon-7b-instruct",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-40b-chat": "tiiuae/falcon-40b-instruct",
}

# task -> ordered multiple-choice labels; index in this list maps to the gold answer
LABELS = {
    "comve": ["A", "B"],
    "causal_judgment": ["A", "B"],
    "disambiguation_qa": ["A", "B", "C"],
    "logical_deduction_five_objects": ["A", "B", "C", "D", "E"],
    "esnli": ["A", "B", "C"],
}

BBH_TASKS = ("causal_judgment", "disambiguation_qa", "logical_deduction_five_objects")

DATA = {
    "comve": ("data/comve/subtaskA_test_data.csv",
              "data/comve/subtaskA_gold_answers.csv"),
    "esnli": "data/e-SNLI/esnli_test.csv",
    "bbh": "data/bbh/{task}/val_data.json",
}

# faithfulness tests to run per sample; drop names here to skip them
TESTS = (
    "atanasova_counterfactual",
    "atanasova_input_from_expl",
    "cc_shap-posthoc",
    "turpin",
    "lanham",
    "cc_shap-cot",
)

# the paper uses a large chat model to paraphrase / corrupt CoT in the Lanham test
DEFAULT_HELPER_MODEL = "llama2-13b-chat"
