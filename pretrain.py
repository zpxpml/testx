#coding: utf-8
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
import math
import random
import time
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from traindata import BlockNIDataset
from llm.model_s7 import LightConfig, LightForCausalLM

# 1. 参数配置增强
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="./configs/config.json", help="模型配置文件路径")
    parser.add_argument("--dataset", default="../dataset/pretrain_hq.jsonl", help="训练数据路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=16) # ***
    parser.add_argument("--lr", type=float, default=1e-3) # ***
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_tokens", type=int, default=1)
    parser.add_argument("--diff_tokens", type=bool, default=False)
    parser.add_argument("--save_dir", default="./checkpoints")
    return parser.parse_args()

# 2. 设备管理优化
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

# 3. 模型初始化改进
def init_model(is_load : bool = False, n_tokens: int = 8, freeze_encdec: bool = False, device: torch.device = 'cuda0'):
    lm_config = LightConfig(hidden_size=576, use_moe=False)
    model = LightForCausalLM(lm_config).to(device)

    print(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    if is_load:
        model.load_state_dict(torch.load("./checkpoints/model_output_b1.pth"))
        
    if freeze_encdec:
        freeze_layers = ['encoder','decoder','embed_tokens','out_proj']
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
    return model

# learning rate schedule
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def block_causal_mask(seq_len, block_size, pred_blocks, device):
    idx = torch.arange(seq_len, device=device)
    remainder = seq_len % block_size

    if remainder == 0:
        block_id = idx // block_size
    else:
        block_id = torch.zeros_like(idx)
        block_id[remainder:] = 1 + (idx[remainder:] - remainder) // block_size

    # 1. 原始 causal block mask
    mask = block_id[:, None] >= block_id[None, :]   # True = allow

    if pred_blocks is None or len(pred_blocks) == 0:
        return mask

    # 2. 构造“未知块” = pred_blocks + 1
    unknown_blocks = set(int(b) + 1 for b in pred_blocks)
    # 3. 所有不在 unknown_blocks 的，都是“已知块”
    known_block = torch.ones_like(block_id, dtype=torch.bool)
    for b in unknown_blocks:
        if remainder == 0:
            known_block &= (block_id != b)
        else:
            known_block &= (block_id != (b + 1))

    # 4. 打开所有指向“已知 block”的 attention
    mask = mask | known_block[None, :]

    return mask

def mask_out_padding(block_mask, atten_mask):
    B, L = atten_mask.shape # bool: [L,L]

    # 扩展 block_mask 到 batch
    block_mask = block_mask[None, :, :].expand(B, -1, -1)  # [B, L, L]
    key_mask   = atten_mask[:, None, :]   # [B, 1, L]
    query_mask = atten_mask[:, :, None]   # [B, L, 1]

    full_mask = block_mask & key_mask # & query_mask   # [B, L, L]
    return full_mask[:,None,:,:]

# 4. 训练循环重构
def train_epoch(model, dataloader, dataset, criterion, scaler, optimizer, epoch, args, device):
    # strat trianing
    model.train()
    total_loss = 0
    start_time = time.time()
    accumulation_steps = args.accumulation_steps
    iter_per_epoch = len(dataloader)

    # 初始化滑动窗口，用于计算 moving loss
    moving_window_size = 100
    recent_losses1 = deque(maxlen=moving_window_size)
    recent_losses2 = deque(maxlen=moving_window_size)
    # 
    block_size = args.n_tokens
    block_sizes = [1,2,3,4]
    block_mask = None
    semi_bid = False
    dataset.set_block_size(block_size)
    dataset.set_predict_blocks(semi_bid, None)
    for batch_idx, batch in enumerate(dataloader):
        # 数据迁移到设备
        inputs = batch["block_tokens"].to(device) # [b, seq]
        labels = batch["target_block_tokens"].to(device) # [b, seq]
        loss_mask = batch["block_loss_mask"].to(device) # [b, seq]
        atten_mask = batch["atten_mask"].to(device)

        full_mask = atten_mask if block_mask == None else mask_out_padding(block_mask, atten_mask)
        
        # lr设置
        lr_now = get_lr(epoch * iter_per_epoch + batch_idx, args.epochs * iter_per_epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        # 前向传播（添加注意力掩码）
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(inputs, attention_mask = full_mask, 
                                n_tokens = block_size, semi_bid = semi_bid)
            # 损失计算
            loss_ce = criterion(out['logits'], labels, loss_mask)
            loss = loss_ce
            loss = loss/accumulation_steps

        # 构造下一个batch的blocksize
        if args.diff_tokens:
            block_size = random.choice(block_sizes)
            dataset.set_block_size(block_size)

            semi_bid = False #random.choice([True, False])
            dataset.set_predict_blocks(semi_bid, None)
            # 构建待预测block
            if semi_bid:
                _ , seq_len = inputs.shape
                num_blocks = seq_len // block_size
                num_pred_blocks = random.choice(list(range(num_blocks // 2, num_blocks -10))) 
                start = random.choice(list(range(0, num_blocks-1-num_pred_blocks+1)))
                pred_blocks = list(range(start,start+num_pred_blocks))
                # candidates = list(range(1, num_blocks-1)) # 注意 range(1, n-1) 不包含 n-1
                # pred_blocks = np.random.choice(candidates, size=num_pred_blocks, replace=False)
                dataset.set_predict_blocks(True, pred_blocks)
                block_mask = block_causal_mask(seq_len, block_size, pred_blocks, device)

            
        # 反向传播计算梯度
        scaler.scale(loss).backward()
        # 梯度下降更新参数
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        recent_losses1.append(loss.item()*accumulation_steps)
        recent_losses2.append(loss.item()*accumulation_steps)
        spend_time = time.time() - start_time

        if batch_idx % 100 == 0:
            moving_loss = sum(recent_losses1) / len(recent_losses1)
            moving_celoss = sum(recent_losses2) / len(recent_losses2)
            print(f"Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item() * accumulation_steps:.4f}  "
                  f"(mov_avg: {moving_loss:.4f}) | "
                  f"(mov_mse: {moving_celoss:.4f}) | "
                  f"Time: {spend_time / (batch_idx + 1) * iter_per_epoch // 60 - spend_time // 60}")
        # save model
        if (batch_idx + 1) % args.save_interval == 0 or (batch_idx + 1) % iter_per_epoch == 0:
            model.eval()
            torch.save(model.state_dict(), f"{args.save_dir}/model_output_{epoch}_{batch_idx}.pth")
            model.train()
    return total_loss / len(dataloader)


class BlockCriterion(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps  # 防止 log(0) 的极小值

    def forward(self, logits, labels, loss_mask):
        """
        logits:     [B*nSeq, V]
        labels:     [B, nSeq]
        loss_mask:  [B, N]
        """
        B, nSeq = labels.shape
        V = logits.size(-1)
        
        
        # CE loss (compute per-token loss, no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # logits: [B, nSeq, V], labels: [BN*K]
        logits_flat = logits.reshape(B*nSeq, V)
        labels_flat = labels.reshape(B*nSeq)
        
        token_loss = loss_fct(logits_flat, labels_flat)   # [B*nSeq]

        # mask out invalid tokens
        token_loss = token_loss * loss_mask.reshape(B*nSeq)
        
        # normalization = number of valid tokens
        denom = loss_mask.sum() + self.eps
        
        return token_loss.sum() / denom #* K

# 5. 主流程优化
def main():
    # 初始化参数
    args = get_args()
    device = setup_device()
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer2/")
    # 数据集加载（添加错误处理）
    try:
        max_length = 512
        dataset = BlockNIDataset(args.dataset, tokenizer, max_length, args.n_tokens)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # ***
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return
    # 数据集信息
    lr = args.lr
    epochs = args.epochs
    # 随机性设置
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    # 模型初始化
    model = init_model(False, args.n_tokens, False, device)
    # 模型精度设置
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # lr规划器
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*len(dataloader))
    # 损失函数
    criterion = BlockCriterion()
    # 迭代学习（强制退出保存）
    try:
        for epoch in range(epochs):
            avg_loss = train_epoch(model, dataloader, dataset, criterion, scaler, optimizer, epoch, args, device)
        # 模型保存
        torch.save(model.state_dict(), f"{args.save_dir}/model_output.pth")
    except KeyboardInterrupt:
        print("\n检测到中断信号（Ctrl+C）")
        user_input = input("是否保存模型？输入 1 保存，其他键退出: ")
        if user_input.strip() == "1":
            print("正在保存模型...")
            torch.save(model.state_dict(), f"{args.save_dir}/model_output_interp.pth")
        else:
            print("未保存模型，直接退出")

if __name__ == "__main__":
    main()