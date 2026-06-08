"""精细诊断 Baseline vs LoRA 各阶段显存分配"""
import torch, gc, math
import torch.nn as nn
from model_RoPE import ModelConfig, GPT


class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, lora_alpha=16):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        in_f, out_f = linear.in_features, linear.out_features
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_f, r, device=device) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, device=device))

    def forward(self, x):
        y = self.linear(x)
        if self.r > 0:
            y = y + self.scaling * (x @ self.lora_A @ self.lora_B)
        return y


def apply_lora(model, r=8, lora_alpha=16):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                if child_name == 'lm_head':
                    for p in child.parameters():
                        p.requires_grad = False
                    continue
                setattr(module, child_name, LoRALinear(child, r=r, lora_alpha=lora_alpha))
                count += 1
    print(f"Applied LoRA to {count} linear layers")
    return model


def detailed_measure(model, optimizer, batch, label):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated() / 1024 ** 2

    x, y = batch
    logits, loss = model(x, y)
    mem_fwd = torch.cuda.memory_allocated() / 1024 ** 2

    loss.backward()
    mem_bwd = torch.cuda.memory_allocated() / 1024 ** 2

    optimizer.step()
    optimizer.zero_grad()
    mem_opt = torch.cuda.memory_allocated() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f"\n--- {label} ---")
    print(f"  After model load : {mem0:.0f} MB")
    print(f"  After forward     : {mem_fwd:.0f} MB (+{mem_fwd - mem0:.0f} MB activations)")
    print(f"  After backward    : {mem_bwd:.0f} MB (+{mem_bwd - mem_fwd:.0f} MB gradients)")
    print(f"  After optimizer   : {mem_opt:.0f} MB (+{mem_opt - mem_bwd:.0f} MB opt states)")
    print(f"  PEAK              : {peak:.0f} MB")
    return peak


model_args = dict(n_layer=4, n_head=8, n_embd=512, block_size=512, bias=True,
                  vocab_size=50304, dropout=0.2)

batch = (torch.randint(0, 50304, (4, 512)).cuda(),
         torch.randint(0, 50304, (4, 512)).cuda())

# ======== Baseline ========
print("=" * 60)
print("  Detailed GPU Memory Breakdown")
print("=" * 60)

m1 = GPT(ModelConfig(**model_args)).cuda()
opt1 = torch.optim.AdamW(m1.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1,
                         fused=True)
p1 = detailed_measure(m1, opt1, batch, "BASELINE (full training)")

del m1, opt1
torch.cuda.empty_cache()
gc.collect()

# ======== LoRA ========
m2 = GPT(ModelConfig(**model_args)).cuda()
m2 = apply_lora(m2, r=8, lora_alpha=16)
m2.transformer.wte.weight.requires_grad = False
lora_params = [p for p in m2.parameters() if p.requires_grad]
print(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,} ({sum(p.numel() for p in lora_params)/1e6:.2f}M)")
opt2 = torch.optim.AdamW(lora_params, lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1,
                         fused=True)
p2 = detailed_measure(m2, opt2, batch, "LORA (r=8, all 16 Linear layers)")

print(f"\n{'=' * 60}")
print(f"  SUMMARY:")
print(f"    Baseline peak : {p1:.0f} MB")
print(f"    LoRA peak     : {p2:.0f} MB")
print(f"    Memory saved  : {p1 - p2:.0f} MB ({(1 - p2 / p1) * 100:.1f}%)")
print(f"{'=' * 60}")
