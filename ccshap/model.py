"""Thin wrapper around a HuggingFace causal LM for generation and label scoring."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_NEW_TOKENS, MODELS

# models whose tokenizer emits the label as the first token (no leading BOS/space token)
_LABEL_FIRST_TOKEN_MODELS = ("gpt", "bloom", "falcon")


class LanguageModel:
    def __init__(self, name: str, model, tokenizer,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._label_token_idx = (
            0 if any(tag in name for tag in _LABEL_FIRST_TOKEN_MODELS) else 1
        )

    @classmethod
    def load(cls, name: str, max_new_tokens: int = MAX_NEW_TOKENS) -> LanguageModel:
        dtype = torch.float32 if "llama2-7b" in name else torch.float16
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                MODELS[name], torch_dtype=dtype, device_map="auto", token=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODELS[name], use_fast=False, padding_side="left")

        for cfg in (model.generation_config, model.config):  # model.config helps gpt2
            cfg.is_decoder = True
            cfg.max_new_tokens = max_new_tokens
            cfg.min_new_tokens = 1
        return cls(name, model, tokenizer, max_new_tokens)

    def generate(self, text: str, max_new_tokens: int | None = None,
                 padding: bool = False, repeat_input: bool = True) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        input_ids = self.tokenizer([text], return_tensors="pt",
                                   padding=padding).input_ids.cuda()
        generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        if not repeat_input:
            generated_ids = generated_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def classify(self, text: str, labels: list[str], padding: bool = False) -> str:
        """Pick the label token the model scores highest as the next token."""
        input_ids = self.tokenizer([text], padding=padding,
                                   return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids, do_sample=False, output_scores=True,
            return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)

        label_scores = np.zeros(len(labels))
        for i, label in enumerate(labels):
            label_id = self.tokenizer.encode(label)[self._label_token_idx]
            label_scores[i] = generated_ids.scores[0][0, label_id]
        return labels[np.argmax(label_scores)]
