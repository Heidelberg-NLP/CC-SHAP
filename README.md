# CC-SHAP

This is the implementation of the paper "On Measuring Faithfulness of Natural Language Explanations" https://arxiv.org/abs/2311.07466 .

## Cite
```bibtex
@misc{parcalabescu2023measuring,
      title={On Measuring Faithfulness of Natural Language Explanations}, 
      author={Letitia Parcalabescu and Anette Frank},
      year={2023},
      eprint={2311.07466},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      abstract = "Large language models (LLMs) can explain their own predictions, through post-hoc or Chain-of-Thought (CoT) explanations. However the LLM could make up reasonably sounding explanations that are unfaithful to its underlying reasoning. Recent work has designed tests that aim to judge the faithfulness of either post-hoc or CoT explanations. In this paper we argue that existing faithfulness tests are not actually measuring faithfulness in terms of the models' inner workings, but only evaluate their self-consistency on the output level. The aims of our work are two-fold. i) We aim to clarify the status of existing faithfulness tests in terms of model explainability, characterising them as self-consistency tests instead. This assessment we underline by constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open-source LLMs and 5 datasets -- including ii) our own proposed self-consistency measure CC-SHAP. CC-SHAP is a new fine-grained measure (not test) of LLM self-consistency that compares a model's input contributions to answer prediction and generated explanation. With CC-SHAP, we aim to take a step further towards measuring faithfulness with a more interpretable and fine-grained method. Code available at https://github.com/Heidelberg-NLP/CC-SHAP", 
}
```

## Usage
To reproduce the experiments:
1. Install the `requirements.txt`
1. Download the data by cloning the following repos.
  - e-SNLI `git clone https://github.com/OanaMariaCamburu/e-SNLI`
  - ComVE: `git clone https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation`
  - BBH samples: `git clone https://github.com/milesaturpin/cot-unfaithfulness`
1. Make sure you have enough compute resources for the largest models. We ran our experiments with 4x NVIDIA A40 (48 GB).
1. Run `python faithfulness.py [TASK] [MODEL] 100`.

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .

## Disclaimer
This is work in progress. Code and paper will be revised and improved for conference submissions (submission which is subject of anonimity period unfortunately).
