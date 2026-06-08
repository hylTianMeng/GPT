"""
使用两种不同的文本生成策略生成文本：
  策略1: Top-k 采样 (top_k=200, temperature=0.8)
  策略2: Nucleus / Top-p 采样 (top_p=0.9, temperature=0.8)
"""
import os
import torch
import torch.nn.functional as F
import tiktoken
from contextlib import nullcontext
from model_RoPE import ModelConfig, GPT

# ----------------------------- 配置 ----------------------------------
out_dir = 'out-wikitext'
start = "\n"                          # 起始 prompt
num_samples = 3                       # 每种策略生成的样本数
max_new_tokens = 200                  # 每个样本最多生成的 token 数
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# 策略参数
top_k = 200                           # Top-k 采样的 k 值
top_p = 0.9                           # Nucleus 采样的 p 值
temperature = 0.8                     # 两个策略共用相同的 temperature

# ----------------------------- 加载模型 ----------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint from {ckpt_path} ...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = ModelConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
print(f"Model loaded. Iteration: {checkpoint.get('iter_num', 'N/A')}, Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

# ----------------------------- Tokenizer ----------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# ----------------------------- 自定义生成函数 ----------------------------------
@torch.no_grad()
def generate_top_k(model, idx, max_new_tokens, temperature=1.0, top_k=200):
    """策略1: Top-k 采样 —— 仅从概率最高的 k 个 token 中采样"""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def generate_top_p(model, idx, max_new_tokens, temperature=1.0, top_p=0.9):
    """策略2: Nucleus (Top-p) 采样 —— 从累积概率 ≥ p 的最小 token 集合中采样"""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        # 按概率降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 始终保留第一个 token（概率最高的）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # 将需要移除的 token 的 logit 设为 -inf
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ----------------------------- 生成文本 ----------------------------------
print("\n" + "=" * 80)
print("  策略1: Top-k 采样 (top_k=200, temperature=0.8)")
print("=" * 80)
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate_top_k(model, x.clone(), max_new_tokens, temperature=temperature, top_k=top_k)
            print(f"\n--- Sample {k+1} ---")
            print(decode(y[0].tolist()))
            print("")

print("\n" + "=" * 80)
print("  策略2: Nucleus / Top-p 采样 (top_p=0.9, temperature=0.8)")
print("=" * 80)
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate_top_p(model, x.clone(), max_new_tokens, temperature=temperature, top_p=top_p)
            print(f"\n--- Sample {k+1} ---")
            print(decode(y[0].tolist()))
            print("")
