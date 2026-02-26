# DLM-Uncertainty

Diffusion Language Model uncertainty estimation toolkit.

## test run
```bash
uv run python -m main dataset=mmlu rounds=1 limit=10 task=lnll_gen qa_model.name=meta-llama/Llama-3.1-8B-Instruct 
```