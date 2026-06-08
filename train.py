import os
import time
import math
import inspect
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import matplotlib.pyplot as plt

from model_RoPE import ModelConfig, GPT

# ----------------------------- Configuration ----------------------------------
# Default values for GPT-2 (124M) training on WikiText
# I/O
out_dir = 'out-wikitext'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
load_optimizer_state = False

# Data
dataset = 'wikitext_large'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 512

# Model Architecture
n_layer = 4
n_head = 8
n_embd = 512
dropout = 0.2
bias = True

# Optimizer Settings
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# System
device = 'cuda' # you can set device to 'cuda' if you are using a gpu
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # TODO: set to True after fixing slow Triton compile on Windows

# Override config from CLI/config file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----------------------------- Initialization ----------------------------------
seed_offset = 0

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ----------------------------- Dataset Loader ----------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(split):
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# ----------------------------- Model Initialization ----------------------------
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch (using RoPE positional encoding)")
    gptconf = ModelConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    # Extract n_layer, n_embd from state dict shapes
    n_layers_from_ckpt = max(int(k.split('.')[2]) for k in state_dict if k.startswith('transformer.h.'))
    model_args['n_layer'] = n_layers_from_ckpt + 1
    model_args['n_embd'] = state_dict['transformer.wte.weight'].shape[1]
    # RoPE model doesn't have wpe; infer block_size from cos_cached buffer
    if 'transformer.wpe.weight' in state_dict:
        model_args['block_size'] = state_dict['transformer.wpe.weight'].shape[0]
    else:
        # RoPE model: try to get block_size from cos_cached or use config default
        for k in state_dict:
            if 'cos_cached' in k:
                model_args['block_size'] = state_dict[k].shape[0]
                break
    model_args['vocab_size'] = state_dict['transformer.wte.weight'].shape[0]
    model_args['bias'] = 'transformer.h.0.attn.c_attn.bias' in state_dict
    print(f"Inferred from checkpoint: n_layer={model_args['n_layer']}, n_embd={model_args['n_embd']}, block_size={model_args['block_size']}, vocab_size={model_args['vocab_size']}, bias={model_args['bias']}")
    n_layer = model_args['n_layer']
    block_size = model_args['block_size']
    model = GPT(ModelConfig(**model_args))
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
    # Also update RoPE cos/sin caches if they exist
    for block in model.transformer.h:
        if hasattr(block.attn, 'cos_cached'):
            block.attn.cos_cached = block.attn.cos_cached[:block_size]
            block.attn.sin_cached = block.attn.sin_cached[:block_size]

model.to(device)
raw_model = model

# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# you could use mixed precision training if you are familar with it
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and load_optimizer_state:
    try:
        # Filter optimizer state: skip params whose shape doesn't match the current model
        # (e.g. when block_size was changed via crop_block_size)
        ckpt_opt = checkpoint['optimizer']
        ckpt_state = ckpt_opt['state']
        cleaned_state = {}
        skipped = 0
        for param_id, state in ckpt_state.items():
            # Check if this state entry matches any current optimizer param
            matched = False
            for pg in optimizer.param_groups:
                for p in pg['params']:
                    if 'exp_avg' in state and state['exp_avg'].shape == p.shape:
                        cleaned_state[param_id] = state
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                skipped += 1
        if skipped > 0:
            print(f"Skipped {skipped} optimizer state entries due to shape mismatch (e.g. cropped block_size)")
            ckpt_opt['state'] = cleaned_state
        optimizer.load_state_dict(ckpt_opt)
        # Ensure fused flag is consistent
        for pg in optimizer.param_groups:
            pg['fused'] = False  # safer to use non-fused after loading state
        optimizer.defaults['fused'] = False
        print("Optimizer state loaded successfully")
    except (ValueError, RuntimeError) as e:
        print(f"Warning: could not load optimizer state ({e}), using fresh optimizer")
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ----------------------------- Evaluation and Training --------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

# 记录 loss 用于画图
train_losses = []    # (iter_num, loss)
val_losses = []      # (iter_num, loss)

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # every eval_interval evaluate the model on train and val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append((iter_num, losses['train']))
        val_losses.append((iter_num, losses['val']))
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        _, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {(t1 - t0)*1000:.2f}ms")
    t0 = t1

    iter_num += 1
    local_iter_num += 1

# ----------------------------- Plot Loss Curves ----------------------------------
print("\nTraining complete! Plotting loss curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：eval 时的 train/val loss
if val_losses:
    train_iters, train_vals = zip(*train_losses)
    val_iters, val_vals = zip(*val_losses)
    axes[0].plot(train_iters, train_vals, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_iters, val_vals, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(out_dir, 'loss_curve.png')
plt.savefig(plot_path, dpi=150)
print(f"Loss curve saved to {plot_path}")
plt.show()
