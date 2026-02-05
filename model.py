##########################################################

##########################################################

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

multipler = 1/4

class LightConfig(PretrainedConfig):
    model_type = "LightDLM"
    def __init__(
        self,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 576,
        num_attention_heads: int = 12,
        intermediate_size: int = 1280,
        max_position_embeddings: int = 32768,
        step_hidden_size: int = 0,
        encoder_layers: int = 8,
        decoder_layers: int = 8,
        vocab_size: int = 6410,
        rms_norm_eps: float = 1e-05,
        rope_dim: int = 16,
        rope_theta: int = 1000000.0,
        flash_attn: bool = True,
        # tobe deleted
        num_key_value_heads = 4,
        # moe config
        use_moe: bool = False,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 10,
        n_shared_experts: int = 2,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.vocab_size = vocab_size
        self.rope_dim = rope_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # 
        self.num_key_value_heads = num_key_value_heads
        # moe config
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).to(x.device)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x).to(x.device)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # print(q.shape,cos.shape)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    return q_embed

def apply_rotary_pos_emb2(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class FeedForward(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.intermediate_size = 64 * ((int(config.hidden_size * multipler) + 64 - 1) // 64)

        self.w_gate = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.w_out = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.w_in = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        x_score = self.w_in(x)
        x_gate = self.w_gate(x)
        out = self.w_out(x_score * self.act_fn(x_gate))
        return out


class MoEGate(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for l in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for l in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class MOEFeedForward2(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.config = config
        self.topk = config.num_experts_per_tok

        self.n_experts = config.n_routed_experts + config.n_shared_experts
        self.n_shared = config.n_shared_experts
        self.interm_dim = int(config.hidden_size * multipler) 
        # print(config.hidden_size, self.interm_dim)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.w_gate = nn.Linear(config.hidden_size, self.n_experts * self.interm_dim, bias=False)
        self.w_out = nn.Linear(self.n_experts * self.interm_dim, config.hidden_size, bias=False)
        self.w_in = nn.Linear(config.hidden_size, self.n_experts * self.interm_dim, bias=False)
        # self.w = nn.Linear(self.n_experts * self.interm_dim, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        bsz, seq_len, _ = x.shape

        logits = self.gate(x)  # [B, S, N]
        scores = logits.softmax(dim=-1)
        # topk routing
        topk_val, topk_idx = torch.topk(scores, self.topk, dim=-1)  # [B, T, K]
        denominator = topk_val.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_val / denominator
        topk_idx = topk_idx + self.n_shared

        mask = torch.zeros(bsz, seq_len, self.n_experts, device=x.device, dtype=x.dtype)
        mask.scatter_(-1, topk_idx, topk_weight)
        mask[:,:,:self.n_shared] = 1.0

        x_gate = self.act_fn(self.w_gate(x))
        x_score = self.w_in(x)  # [B, T, N*D]x @ self.w.weight #
        x_score = x_score.view(bsz, seq_len, self.n_experts, self.interm_dim)  # [B, T, N, D]
        x_score = x_score * mask.unsqueeze(-1)  # [B, T, N, D]
        x_score = x_score.reshape(bsz, seq_len, self.n_experts * self.interm_dim)

        out = self.w_out(x_score * x_gate)
        return out

class EncoderAttention(nn.Module):
    def __init__(self, args: LightConfig, sq, sk, sv):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.sq, self.sk, self.sv = sq, sk, sv

        self.gate_w = nn.Linear(args.hidden_size, self.n_local_heads, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # self.head_gate = nn.Parameter(torch.zeros(self.n_local_heads))
        self.g_proj = nn.Linear(self.head_dim, 1, bias=False)
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        

    def block_causal_mask(self, seq_len, block_size, device):
        idx = torch.arange(seq_len, device=device)
        remainder = seq_len % block_size
        if remainder == 0:
            block_id = idx // block_size
        else:
            # 前 remainder 个 token 归为 block 0
            # 后面的 token 从 block 1 开始，每 block_size 个一组
            block_id = torch.zeros_like(idx)
            block_id[remainder:] = 1 + (idx[remainder:] - remainder) // block_size

        mask = block_id[:, None] >= block_id[None, :]
        
        return mask

    def local_prev1_mask(self, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        
        # j <= i 且 j >= i-1
        mask = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] <= 1)
        
        return mask  # [L, L] bool, True = allow

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
                attention_mask: Optional[torch.Tensor] = None,
                block_size = 1, semi_bid = False):
        bsz, seq_len, _ = x.shape
        gate = self.gate_w(x).reshape(bsz, seq_len, self.n_local_heads, -1)
        xq, xk, xv = self.sq(x), self.sk(x), self.sv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb2(xq, xk, cos[:seq_len], sin[:seq_len])

        xq, xk, xv = (
            xq.transpose(1, 2),
            # xk.transpose(1, 2),
            # xv.transpose(1, 2)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            # g = torch.sigmoid(self.head_gate).view(1, self.n_local_heads, 1, 1)
            g = torch.sigmoid(self.g_proj(xk))  # [B,H,S,1]
            xk_new = xk.clone()
            xk_new[:,:,1:,:] = xk[:,:,1:,:] + (2*g[:,:,:-1,:] - 1) * xk[:,:,:-1,:]
            xk = xk_new
            # xk[:,:4,1:,:] = (xk[:,:4,1:,:] - xk[:,:4,:-1,:]) 
            # xk[:,4:8,1:,:] = (xk[:,4:8,1:,:] + xk[:,4:8,:-1,:]) / 2
            causal_idx = (block_size == 1 and semi_bid == False)
            block_mask = None if (causal_idx or semi_bid) else self.block_causal_mask(seq_len, block_size, x.device)
            atten_mask = attention_mask if semi_bid else block_mask

            # atten_mask = self.local_prev1_mask(seq_len, x.device)
            # causal_idx = False

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=atten_mask, is_causal=causal_idx)
        else:
            g = torch.sigmoid(self.g_proj(xk))  # [B,H,S,1]
            # xk_new = xk.clone()
            # xk_new[:,:,1:,:] = xk[:,:,1:,:] + (2*g[:,:,:-1,:] - 1) * xk[:,:,:-1,:]
            # xk = xk_new
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = F.sigmoid(scores)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask
            # shift = 1
            # first_elems = scores[..., :1].repeat(1, 1, 1, shift)
            # rest = scores[..., :-shift]
            # shifted_score = torch.cat([first_elems, rest], dim=-1)
            # scores = (shifted_score + scores)/2
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = (scores * mask) @ xv

        output = ((output.transpose(1, 2)) * F.sigmoid(gate)).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output

class BlockAttention(nn.Module):
    def __init__(self, args: LightConfig, sq, sk, sv):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.sq, self.sk, self.sv = sq, sk, sv

        self.gate_w = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def block_mask(self, seq_len, block_size, device):
        idx = torch.arange(seq_len, device=device)
        remainder = seq_len % block_size
        if remainder == 0:
            block_id = idx // block_size
        else:
            block_id = torch.zeros_like(idx)
            block_id[remainder:] = 1 + (idx[remainder:] - remainder) // block_size

        mask = block_id[:, None] == block_id[None, :]
        # if remainder > 0:
        #     # 所有 block 可以看 prefix
        #     mask[:, block_id == 0] = True
        #     # prefix 不能看后面（保险栓）
        #     # mask[block_id == 0, block_id != 0] = False
        # # print(mask)
        return mask

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
                attention_mask: Optional[torch.Tensor] = None,
                block_size = 1, semi_bid = False):
        bsz, seq_len, _ = x.shape
        gate = self.gate_w(x).reshape(bsz, seq_len, self.n_local_heads, -1)
        xq, xk, xv = self.sq(x), self.sk(x), self.sv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb2(xq, xk, cos[:seq_len], sin[:seq_len])

        xq, xk, xv = (
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            block_mask =  self.block_mask(seq_len, block_size, x.device)
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=block_mask, is_causal=False)

        output = ((output.transpose(1, 2)) * F.sigmoid(gate)).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output

class DecoderAttention(nn.Module):
    def __init__(self, args: LightConfig, sq, sk, sv, n_heads = None):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = n_heads if n_heads else args.num_attention_heads
        self.n_local_kv_heads = n_heads if n_heads else self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads
        self.sq, self.sk, self.sv = sq, sk, sv

        self.gate_w = nn.Linear(args.hidden_size, self.n_local_heads, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # self.head_gate = nn.Parameter(torch.zeros(self.n_local_heads))
        self.g_proj = nn.Linear(self.head_dim, 1, bias=False)
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.intermediate_size = 64 * ((int(self.head_dim * multipler) + 64 - 1) // 64) 

    
    def block_causal_mask(self, seq_len, block_size, device):
        idx = torch.arange(seq_len, device=device)
        remainder = seq_len % block_size
        if remainder == 0:
            block_id = idx // block_size
        else:
            # 前 remainder 个 token 归为 block 0
            # 后面的 token 从 block 1 开始，每 block_size 个一组
            block_id = torch.zeros_like(idx)
            block_id[remainder:] = 1 + (idx[remainder:] - remainder) // block_size

        mask = block_id[:, None] >= block_id[None, :]
        return mask

    def block_summary_mask(self, seq_len, block_size, device):
        idx = torch.arange(seq_len, device=device)
        # i, j 网格
        i = idx[:, None]   # [L,1]
        j = idx[None, :]   # [1,L]
        # 1. causal: 只能看过去和自己
        causal = j <= i
        # 2. summary 条件：j % block_size == 0
        is_summary = ((j+1) % block_size) == 0
        self_visible = (i == j)
        # (j 是 summary 且 j < i) 或 (j == i)
        mask = (is_summary & (j < i)) | self_visible
        # 再加一道 causal 保险
        mask = mask & causal
        return mask  # bool, True = 可以 attention

    def local_prev1_mask(self, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        
        # j <= i 且 j >= i-1
        mask = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] <= 1)
        
        return mask  # [L, L] bool, True = allow

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None, 
                block_size = 1, semi_bid = False):
        bsz, seq_len, _ = x.shape
        gate = self.gate_w(x).reshape(bsz, seq_len, self.n_local_heads, -1)
        xq, xk, xv = self.sq(x), self.sk(x), self.sv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb2(xq, xk, cos[:seq_len], sin[:seq_len])

        xq, xk, xv = (
            xq.transpose(1, 2),
            # xk.transpose(1, 2),
            # xv.transpose(1, 2)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1: 
            # g = torch.sigmoid(self.head_gate).view(1, self.n_local_heads, 1, 1)
            g = torch.sigmoid(self.g_proj(xk))  # [B,H,S,1]
            xk_new = xk.clone()
            xk_new[:,:,1:,:] = xk[:,:,1:,:] + (2*g[:,:,:-1,:] - 1) * xk[:,:,:-1,:]
            xk = xk_new

            causal_idx = (block_size == 1 and semi_bid == False)
            block_mask = None if (causal_idx or semi_bid) else self.block_causal_mask(seq_len, block_size, x.device)
            atten_mask = attention_mask if semi_bid else block_mask
            atten_mask = self.block_summary_mask(seq_len, 4, x.device)
            # atten_mask = self.local_prev1_mask(seq_len, x.device)
            # causal_idx = False

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=atten_mask, is_causal=causal_idx)
        else:
            g = torch.sigmoid(self.g_proj(xk))  # [B,H,S,1]
            xk_new = xk.clone()
            xk_new[:,:,1:,:] = xk[:,:,1:,:] + (2*g[:,:,:-1,:] - 1) * xk[:,:,:-1,:]
            xk = xk_new
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = F.silu(scores)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask
            # shift = 1
            # first_elems = scores[..., :1].repeat(1, 1, 1, shift)
            # rest = scores[..., :-shift]
            # shifted_score = torch.cat([first_elems, rest], dim=-1)
            # scores = (shifted_score + scores)/2
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = (scores * mask) @ xv
        
        

        output = ((output.transpose(1, 2)) * F.sigmoid(gate)).reshape(bsz, seq_len, -1)
        output = self.o_proj(output) #+ ffn_out.reshape(bsz, seq_len, -1)
        return output

class ScaleLastDim(nn.Module):
    def __init__(self, d):
        super().__init__()
        # 无约束参数
        # self.log_scale = nn.Parameter(torch.zeros(d))
        self.logits = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (b, s, d)
        # scale = F.softplus(self.log_scale)  # 保证 > 0
        w = F.softmax(self.logits, dim=0)
        return x * w

class EncLayer(nn.Module):
    def __init__(self, layer_id: int, config: LightConfig, sq, sk, sv):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # if self.layer_id % 4 == 0 :
        self.self_attn = EncoderAttention(config, sq, sk, sv)
        # else:
        #     self.self_attn = BlockAttention(config, sq, sk, sv)
        self.mlp = MOEFeedForward2(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.hc = ScaleLastDim(config.hidden_size)

    def forward(self, hidden_states, position_embeddings, attention_mask = None, 
            block_size = 1, semi_bid = False):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, block_size, semi_bid
        )
        hidden_states = hidden_states + residual #+ self.hc(residual)
        denoise_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = denoise_out + hidden_states #+ self.hc(hidden_states) 
        return hidden_states, denoise_out

class DecLayer(nn.Module):
    def __init__(self, layer_id: int, config: LightConfig, sq, sk, sv, n_heads = None):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        # if self.layer_id % 4 == 0:
        self.self_attn = DecoderAttention(config, sq, sk, sv, n_heads)
        # else:
        #     self.self_attn = BlockAttention(config, sq, sk, sv)
        self.mlp = MOEFeedForward2(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.hc = ScaleLastDim(config.hidden_size)
        # self.hc2 = ScaleLastDim(config.hidden_size)

    def forward(self, hidden_states, position_embeddings, attention_mask = None, 
            block_size = 1, semi_bid=False):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, block_size, semi_bid
        )
        hidden_states = hidden_states + residual #+ self.hc(residual)
        denoise_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = denoise_out + hidden_states #+ self.hc(hidden_states) 
        return hidden_states, denoise_out

class BlockTransformer(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.n_layer = config.encoder_layers + config.decoder_layers
        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.mask_emb = nn.Parameter(torch.randn(config.hidden_size))
        
        group_size = 1
        n_rep = 1 #config.num_attention_heads // config.num_key_value_heads
        n_part = (self.encoder_layers + self.decoder_layers) // group_size
        self.sqs = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(n_part)]
        self.sks = [nn.Linear(config.hidden_size, config.hidden_size//n_rep, bias=False) for _ in range(n_part)]
        self.svs = [nn.Linear(config.hidden_size, config.hidden_size//n_rep, bias=False) for _ in range(n_part)]
        

        # 构建所有层
        self.e_layers = nn.ModuleList([
            EncLayer(l, config, self.sqs[l//group_size], self.sks[l//group_size], self.svs[l//group_size])
            for l in range(config.encoder_layers)
        ])
        self.d_layers = nn.ModuleList([
            DecLayer(l, config, self.sqs[(config.encoder_layers+l)//group_size], 
                self.sks[(config.encoder_layers+l)//group_size], 
                self.svs[(config.encoder_layers+l)//group_size]) #, 8 if l % 4 == 0 else 6)
            for l in range(config.decoder_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, latent, position_embeddings, position_embeddings1, position_embeddings2, attention_mask = None, 
            block_size = 1, semi_bid = False):
        blk_sizes = [8,8,8,8] #[1,2,4,8]

        bsz, seq_len, _ = latent.shape

        for layer_idx, layer in enumerate(self.e_layers):
            blk_idx = layer_idx % 4
            latent, _ = layer(
                latent,
                position_embeddings,
                attention_mask,
                min(block_size, blk_sizes[blk_idx]), semi_bid
            )
        # if self.training:
        # hidden_states = self.mask_emb.unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1).clone()
        # comp_size = block_size
        # blk_latent_idx = torch.arange(seq_len//comp_size, device=hidden_states.device) * comp_size + (comp_size - 1) +seq_len % comp_size
        # # blk_latent_idx = torch.randperm(seq_len)[:seq_len//block_size].sort().values
        # hidden_states[:,blk_latent_idx,:] = latent[:,blk_latent_idx,:]
        # hidden_states[:,:min(seq_len,block_size+seq_len % comp_size),:] = latent[:,:min(seq_len,block_size+seq_len % comp_size),:]
        # else:
        #     hidden_states = latent
        hidden_states = latent

        for layer_idx, layer in enumerate(self.d_layers):
            blk_idx = layer_idx % 4
            hidden_states, denoise_out = layer(
                hidden_states,
                position_embeddings1 if blk_idx == 0 else position_embeddings2,
                attention_mask,
                min(block_size, blk_sizes[blk_idx]), semi_bid
            )
        hidden_states = self.norm(denoise_out)

        return hidden_states

# Main model
class LightModel(nn.Module):
    def __init__(self, config: LightConfig):
        super().__init__()
        self.config = config
        self.model = BlockTransformer(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size//config.num_attention_heads, end=config.max_position_embeddings, theta=config.rope_theta)
        freqs_cos1, freqs_sin1 = precompute_freqs_cis(dim=config.hidden_size//12, end=config.max_position_embeddings, theta=config.rope_theta)
        freqs_cos2, freqs_sin2 = precompute_freqs_cis(dim=config.hidden_size//12, end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.register_buffer("freqs_cos1", freqs_cos1, persistent=False)
        self.register_buffer("freqs_sin1", freqs_sin1, persistent=False)
        self.register_buffer("freqs_cos2", freqs_cos2, persistent=False)
        self.register_buffer("freqs_sin2", freqs_sin2, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                n_tokens: int = 1, semi_bid: bool  = False,
                **kwargs):
        batch_size, seq_len = input_ids.shape
        # n_blocks = seq_len // n_tokens
        position_embeddings = (
            self.freqs_cos[0:seq_len],
            self.freqs_sin[0:seq_len]
        )
        position_embeddings1 = (
            self.freqs_cos1[0:seq_len],
            self.freqs_sin1[0:seq_len]
        )
        position_embeddings2 = (
            self.freqs_cos2[0:seq_len],
            self.freqs_sin2[0:seq_len]
        )

        hidden_states = self.embed_tokens(input_ids)
        token_latents = self.model(hidden_states, position_embeddings, position_embeddings1,
            position_embeddings2, attention_mask, n_tokens, semi_bid)
        logits = self.out_proj(token_latents)
        return logits


class LightForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LightConfig

    def __init__(self, config: LightConfig = None):
        self.config = config or LightConfig()
        super().__init__(self.config)
        self.model = LightModel(self.config)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                attention_mask: Optional[torch.Tensor] = None,
                n_tokens: int = 1, semi_bid: bool  = False,
                **args):
        h = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_tokens=n_tokens, semi_bid=semi_bid,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # if self.training or n_tokens == 1:
        logits = h[:, slice_indices, :]
        # else:
        #logits = h[:, :-n_tokens+1, :]
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        return self.OUT
