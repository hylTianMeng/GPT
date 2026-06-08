"""
使用 LoRA (Low-Rank Adaptation) 进行训练，并汇报显存变化。
LoRA 只训练低秩适配矩阵 A 和 B，冻结原始模型权重。
"""
import os
import time
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_lora import ModelConfig, GPT, GPTLoRA, merge_lora_state_dict

# ----------------------------- 配置 ----------------------------------
out_dir = 'out-wikitext-lora'
eval_interval = 250
log_interval = 10
eval_iters = 50
always_save_checkpoint = True
init_from = 'scratch'

dataset = 'wikitext_large'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 512

# 模型架构（与 baseline 相同）
n_layer = 4
n_head = 8
n_embd = 512
dropout = 0.2
bias = True

# LoRA 参数
lora_r = 8
lora_alpha = 16

# 优化器设置
learning_rate = 6e-4
max_iters = 2000  # LoRA 训练步数（演示用，可调整）
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# CLI 覆盖
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----------------------------- GPU 显存工具 ----------------------------------
def get_gpu_memory_info():
    """获取当前 GPU 显存使用情况 (MB)"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    free = total - reserved
    return {
        "allocated": allocated,
        "reserved": reserved,
        "total": total,
        "free": free
    }

def print_memory(label=""):
    """打印当前 GPU 显存状态"""
    mem = get_gpu_memory_info()
    print(f"[MEM] {label}: allocated={mem['allocated']:.1f}MB, reserved={mem['reserved']:.1f}MB, "
          f"total={mem['total']:.1f}MB, free={mem['free']:.1f}MB")
    return mem

def count_trainable_params(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    """统计总参数量"""
    return sum(p.numel() for p in model.parameters())

# ----------------------------- 初始化 ----------------------------------
seed_offset = 0
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")
print(f"Using device: {device}")
print(f"LoRA config: r={lora_r}, alpha={lora_alpha}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ----------------------------- 数据集 ----------------------------------
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

# ----------------------------- Phase 1: Baseline 模型显存测量 ------------------------------
print("\n" + "=" * 70)
print("Phase 1: Baseline 模型显存测量")
print("=" * 70)

if device_type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_baseline = get_gpu_memory_info()
    print_memory("baseline 加载前")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)

# 创建 baseline model (GPTLoRA 在合并前和普通 GPT 大小相同)
gptconf = ModelConfig(**model_args)
baseline_model = GPTLoRA(gptconf)
baseline_model.to(device)

if device_type == 'cuda':
    mem_after_baseline = get_gpu_memory_info()
    print_memory("baseline 加载后（全部参数在 GPU）")
    baseline_vram = mem_after_baseline["allocated"] - mem_before_baseline["allocated"]
    print(f"  => Baseline 模型占用显存: {baseline_vram:.1f} MB")

total_params_baseline = count_total_params(baseline_model)
trainable_params_baseline = count_trainable_params(baseline_model)
print(f"  Baseline 总参数量: {total_params_baseline/1e6:.2f}M")
print(f"  Baseline 可训练参数量: {trainable_params_baseline/1e6:.2f}M")

# 清理 baseline
del baseline_model
if device_type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ----------------------------- Phase 2: LoRA 模型显存与训练 ------------------------------
print("\n" + "=" * 70)
print("Phase 2: LoRA 模型显存测量与训练")
print("=" * 70)

if device_type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_lora = get_gpu_memory_info()
    print_memory("LoRA 模型创建前")

# 创建 LoRA 模型
model = GPTLoRA(gptconf)
model.to(device)

if device_type == 'cuda':
    mem_after_lora = get_gpu_memory_info()
    print_memory("LoRA 模型加载后")
    lora_model_vram = mem_after_lora["allocated"] - mem_before_lora["allocated"]
    print(f"  => LoRA 模型占用显存: {lora_model_vram:.1f} MB")

total_params_lora = count_total_params(model)
trainable_params_lora = count_trainable_params(model)
print(f"  LoRA 总参数量: {total_params_lora/1e6:.2f}M")
print(f"  LoRA 可训练参数量: {trainable_params_lora/1e6:.2f}M")
print(f"  可训练参数减少: {(1 - trainable_params_lora/trainable_params_baseline)*100:.1f}%")

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if device_type == 'cuda':
    mem_after_optimizer = get_gpu_memory_info()
    print_memory("LoRA 模型 + 优化器创建后")
    optimizer_vram = mem_after_optimizer["allocated"] - mem_after_lora["allocated"]
    print(f"  => 优化器状态占用显存: {optimizer_vram:.1f} MB")
    total_lora_vram = mem_after_optimizer["allocated"] - mem_before_lora["allocated"]
    print(f"  => LoRA 模型 + 优化器总占用: {total_lora_vram:.1f} MB")

# ----------------------------- 训练循环 ----------------------------------
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

print("\n" + "=" * 70)
print("Phase 3: 开始 LoRA 训练")
print("=" * 70)

X, Y = get_batch('train')
t0 = time.time()
iter_num = 0
local_iter_num = 0

train_losses = []
val_losses = []
memory_records = []  # (iter_num, allocated_mb, reserved_mb)

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append((iter_num, losses['train']))
        val_losses.append((iter_num, losses['val']))
        # 记录显存
        mem = get_gpu_memory_info()
        memory_records.append((iter_num, mem['allocated'], mem['reserved']))
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}  |  "
              f"GPU allocated: {mem['allocated']:.1f}MB, reserved: {mem['reserved']:.1f}MB")
        if losses['val'] < 1e9:
            best_val_loss = losses['val']
            print(f"saving checkpoint to {out_dir}")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }, os.path.join(out_dir, 'ckpt.pt'))

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

# ----------------------------- 显存汇总报告 ----------------------------------
print("\n" + "=" * 70)
print("显存使用汇总报告")
print("=" * 70)

if device_type == 'cuda':
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    print(f"  峰值分配显存 (peak allocated):  {peak_allocated:.1f} MB")
    print(f"  峰值保留显存 (peak reserved):   {peak_reserved:.1f} MB")
    print(f"  总 GPU 显存:                    {mem['total']:.1f} MB")
    print()
    print(f"  Baseline 模型占用:              {baseline_vram:.1f} MB  ({trainable_params_baseline/1e6:.2f}M 可训练参数)")
    print(f"  LoRA 模型占用:                  {lora_model_vram:.1f} MB  ({trainable_params_lora/1e6:.2f}M 可训练参数)")
    print(f"  LoRA 优化器状态占用:            {optimizer_vram:.1f} MB")
    print(f"  LoRA 模型+优化器总占用:         {total_lora_vram:.1f} MB")
    print()
    # 可训练参数显存节省 (粗略估算：每参数 ~4 bytes fp32 优化器状态 + 模型权重)
    # 实际上优化器存 2 个动量 (fp32)，所以大约 8 bytes/param for optimizer states
    param_mem_saved = (trainable_params_baseline - trainable_params_lora) * (4 + 8) / 1024**2  # weights + optimizer
    print(f"  可训练参数减少:                 {(trainable_params_baseline - trainable_params_lora)/1e6:.2f}M")
    print(f"  可训练参数占比:                 {trainable_params_lora/trainable_params_baseline*100:.1f}%")
    print(f"  预估优化器显存节省:             {param_mem_saved:.1f} MB")

# ----------------------------- 画 Loss 和显存曲线 ----------------------------------
print("\n绘制曲线...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Loss 曲线
if val_losses:
    train_iters, train_vals = zip(*train_losses)
    val_iters, val_vals = zip(*val_losses)
    axes[0].plot(train_iters, train_vals, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_iters, val_vals, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('LoRA Training: Train & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# 右图：显存曲线
if memory_records:
    mem_iters, mem_alloc, mem_resv = zip(*memory_records)
    axes[1].plot(mem_iters, mem_alloc, label='Allocated (MB)', marker='o', markersize=4, color='green')
    axes[1].plot(mem_iters, mem_resv, label='Reserved (MB)', marker='s', markersize=4, color='orange')
    axes[1].axhline(y=peak_allocated, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak_allocated:.0f}MB')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('GPU Memory (MB)')
    axes[1].set_title('GPU Memory Usage During LoRA Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(out_dir, 'lora_training_curves.png')
plt.savefig(plot_path, dpi=150)
print(f"曲线图已保存到 {plot_path}")

print("\nLoRA 训练完成！")
