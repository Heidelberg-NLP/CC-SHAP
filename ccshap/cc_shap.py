"""CC-SHAP: measure a model's self-consistency between prediction and explanation.

CC-SHAP (Parcalabescu & Frank) attributes, with SHAP, how much each input token drives
(a) the model's answer and (b) the model's explanation of that answer, then compares the
two attribution profiles. If a model answers and explains for the same reasons, the two
profiles line up; the score is ``1 - cosine_distance`` between them, in ``[-1, 1]``
where higher means more self-consistent (faithful).

Both attribution profiles are turned into per-input-token *ratios* (percent of the total
attribution mass), after marginalising the prompt scaffolding tokens (the answer/"Why?"
suffix) into SHAP's base value so only the actual input tokens are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import spatial, special, stats
from sklearn import metrics

import shap

from .config import MAX_NEW_TOKENS
from .model import LanguageModel
from .prompts import Prompts


@dataclass
class CCShapResult:
    score: float          # 1 - cosine distance: the headline CC-SHAP score
    dist_correl: float
    mse: float
    var: float
    kl_div: float
    js_div: float
    plot_info: dict


def _aggregate_explanation(shap_values, tokenizer,
                           to_marginalize: str) -> tuple[np.ndarray, int]:
    """Per-input-token ratios for an explanation, dropping the scaffolding tokens.

    The trailing ``to_marginalize`` tokens (e.g. the answer + "Why?" suffix) are folded
    into the base value via SHAP's additivity so only genuine input tokens remain.
    """
    len_to_marginalize = tokenizer(
        [to_marginalize], return_tensors="pt", padding=False,
        add_special_tokens=False).input_ids.shape[1]
    add_to_base = np.abs(shap_values.values[:, -len_to_marginalize:]).sum(axis=1)
    ratios = (shap_values.values
              / (np.abs(shap_values.values).sum(axis=1) - add_to_base) * 100)
    return np.mean(ratios, axis=2)[0, :-len_to_marginalize], len_to_marginalize


def _divergence_scores(ratios_prediction: np.ndarray,
                       ratios_explanation: np.ndarray) -> tuple[float, ...]:
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    dist_correl = spatial.distance.correlation(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = (np.sum(((ratios_prediction - ratios_explanation) ** 2 - mse) ** 2)
           / ratios_prediction.shape[0])
    kl_div = stats.entropy(special.softmax(ratios_explanation),
                           special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction),
                                            special.softmax(ratios_explanation))
    return cosine, dist_correl, mse, var, kl_div, js_div


class CCShap:
    def __init__(self, model: LanguageModel, prompts: Prompts):
        self.model = model
        self.prompts = prompts
        self.explainer = shap.Explainer(model.model, model.tokenizer, silent=True)

    def _explain(self, text: str, max_new_tokens: int):
        self.model.model.generation_config.max_new_tokens = max_new_tokens
        self.model.model.config.max_new_tokens = max_new_tokens
        return self.explainer([text])

    def measure(self, inputt: str, labels: list[str],
                expl_type: str = "post_hoc") -> CCShapResult:
        prompt_prediction = self.prompts.prediction(inputt)
        predicted_label = self.model.classify(prompt_prediction, labels)
        values_prediction = self._explain(prompt_prediction, max_new_tokens=1)

        answer_and_prompt = self.prompts.explanation_suffix(predicted_label, expl_type)
        explanation_input = self.prompts.explanation_body(inputt) + answer_and_prompt
        values_explanation = self._explain(explanation_input,
                                           max_new_tokens=MAX_NEW_TOKENS)

        marg_pred = self.prompts.prediction_margin(expl_type)
        tokenizer = self.model.tokenizer
        ratios_prediction, len_marg_pred = _aggregate_explanation(
            values_prediction, tokenizer, marg_pred)
        ratios_explanation, len_marg_expl = _aggregate_explanation(
            values_explanation, tokenizer, answer_and_prompt)

        cosine, dist_correl, mse, var, kl_div, js_div = _divergence_scores(
            ratios_prediction, ratios_explanation)

        plot_info = {
            "ratios_prediction": ratios_prediction.astype(float).round(2)
                                 .astype(str).tolist(),
            "ratios_explanation": ratios_explanation.astype(float).round(2)
                                  .astype(str).tolist(),
            "input_tokens": values_prediction.data[0].tolist(),
            "expl_input_tokens": values_explanation.data[0].tolist(),
            "len_marg_pred": len_marg_pred,
            "len_marg_expl": len_marg_expl,
        }
        return CCShapResult(1 - cosine, 1 - dist_correl, 1 - mse, 1 - var,
                            1 - kl_div, 1 - js_div, plot_info)
