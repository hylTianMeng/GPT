import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import ModelConfig, MLP

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        
        # TODO: 1. Initialize self.linear as a standard nn.Linear and freeze its weights (requires_grad = False)
        # TODO: 2. If r > 0, register self.lora_A and self.lora_B as learnable parameters (nn.Parameter)
        # TODO: 3. Initialize lora_A with random normal distribution (std = 1/sqrt(r)) and lora_B with zeros
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for param in self.linear.parameters():
            param.requires_grad = False
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r) / math.sqrt(r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x):
        # TODO: Implement the forward pass incorporating LoRA
        # Formula: output = linear(x) + scaling * (x @ lora_A @ lora_B)
        y = self.linear(x)
        if self.r > 0:
            y = y + self.scaling * (x @ self.lora_A @ self.lora_B)
        return y

class CausalSelfAttentionLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 核心改动：用 LoRALinear 替换标准的 nn.Linear
        self.c_attn = LoRALinear(config.n_embd, 3 * config.n_embd, r=8, lora_alpha=16, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # TODO: Implement the CausalSelfAttention process (similar to Task 4.2)
        # Make sure to use self.c_attn(x) to get q, k, v and apply the rest of attention mechanisms
        B, L, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:L,:L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class BlockLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionLoRA(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockLoRA(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of trainable parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: Tensor of shape (B, T)
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (int)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def merge_lora_state_dict(state_dict, r=8, lora_alpha=16):
    """
    Convert a GPTLoRA state_dict into a plain GPT-compatible state_dict.

    The merged QKV weight is:
        linear.weight + scaling * (lora_A @ lora_B).T

    This mirrors common LoRA export behavior: keep LoRA checkpoints during
    finetuning, then explicitly merge for inference or plain-model resume.
    """
    merged = {}
    consumed_lora_keys = set()

    for key, value in state_dict.items():
        if key.endswith(".attn.c_attn.linear.weight"):
            prefix = key[: -len(".linear.weight")]
            lora_A_key = prefix + ".lora_A"
            lora_B_key = prefix + ".lora_B"

            if lora_A_key in state_dict and lora_B_key in state_dict:
                lora_A = state_dict[lora_A_key]
                lora_B = state_dict[lora_B_key]
                scaling = lora_alpha / r
                value = value + (lora_A @ lora_B).transpose(0, 1) * scaling
                consumed_lora_keys.update({lora_A_key, lora_B_key})

            merged[key.replace(".linear.weight", ".weight")] = value
            continue

        if key.endswith(".attn.c_attn.linear.bias"):
            merged[key.replace(".linear.bias", ".bias")] = value
            continue

        if key.endswith(".attn.c_attn.lora_A") or key.endswith(".attn.c_attn.lora_B"):
            consumed_lora_keys.add(key)
            continue

        merged[key] = value

    return merged


def convert_lora_checkpoint_to_gpt_checkpoint(checkpoint, r=8, lora_alpha=16):
    """
    Return a copy of a training checkpoint whose `model` entry can be loaded by
    the plain GPT class from model.py.
    """
    converted = dict(checkpoint)
    converted["model"] = merge_lora_state_dict(checkpoint["model"], r=r, lora_alpha=lora_alpha)
    converted["optimizer"] = None
    return converted


def save_merged_lora_checkpoint(input_path, output_path, map_location="cpu", r=8, lora_alpha=16):
    """
    Load a GPTLoRA checkpoint, merge the LoRA weights into the base QKV weights,
    and save a plain GPT-compatible checkpoint.

    The optimizer state is set to None because LoRA optimizer states are not
    compatible with the parameter set of the plain GPT model.
    """
    checkpoint = torch.load(input_path, map_location=map_location)
    converted = convert_lora_checkpoint_to_gpt_checkpoint(
        checkpoint,
        r=r,
        lora_alpha=lora_alpha,
    )
    torch.save(converted, output_path)
    return converted

GPT = GPTLoRA
