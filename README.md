# GPT Homework

This directory contains the student-facing code for the language-model part of HW-4. You should only fill in the TODO sections needed for the assignment.

## Tasks

Complete the TODOs for:

1. Gradient accumulation in `train.py`.
2. Causal multi-head self-attention and top-p sampling in `model.py`.
3. RoPE attention in `model_RoPE.py`.
4. LoRA layers and LoRA attention in `model_lora.py`.

The surrounding model, training, sampling, optimizer, and data-loading code is already provided.

## Choose a model file

By default, `train.py` and `sample.py` import from `model.py`.

For the RoPE experiment, change the import to:

```python
from model_RoPE import ModelConfig, GPT
```

For the LoRA experiment, change the import to:

```python
from model_lora import ModelConfig, GPT
```

Do not resume a checkpoint across different architectures. A checkpoint trained with `model.py` should resume with `model.py` or `model_lora.py`; a checkpoint trained with `model_RoPE.py` should resume with `model_RoPE.py`.

## Training

Run:

```bash
python train.py config/train_wikitext.py
```

You can reduce model size, batch size, evaluation iterations, or training iterations if your device is slow. Do not submit checkpoints or datasets.

## LoRA Finetuning

When finetuning a checkpoint trained with `model.py`, change the model import to use `model_lora.py` and set the `load_optimizer_state=False`, since the optimizer state of LoRA is different and we only need to load the model. You can convert a LoRA finetuned checkpoint back into a normal model checkpoint by using `model_lora.save_merged_lora_checkpoint(lora_checkpoint_path, output_original_checkpoint_path)`.

## Sampling

After training a checkpoint, run:

```bash
python sample.py --out_dir=YOUR_MODEL_DIR_PATH
```

Use `generate` for top-k sampling and `generate_with_top_p` for top-p sampling in your report comparison.

## Report

Your report should include:

- training and validation loss trends for the base attention model,
- training and validation loss trends for the RoPE model,
- generated text examples from top-k and top-p sampling,
- a short explanation of your LoRA implementation,
- the memory-use comparison requested in the homework.
