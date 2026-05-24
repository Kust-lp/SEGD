# -*- coding: utf-8 -*-
import argparse
import os
import sys
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.utils import ModelEmaV2
import cv2
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from model.DENet import *


@torch.no_grad()
def Validate(model, val_loader, args, loss_fn):
    model.eval()
    avg_loss = defaultdict(float)
    pre = []
    gt = []
    for lIr, _, label, _ in val_loader:
        lIr = lIr.to(args.device)
        label = label.to(args.device)
        pre_label, alpha, beta, p_hat, S = model(lIr)
        Loss, info = loss_fn(pre_label, alpha, beta, p_hat, label)
        pre.append(pre_label)
        gt.append(label[:,:3])

        for k, v in info.items():
            avg_loss[k] += v
    avg_loss = {k: v / len(val_loader) for k, v in avg_loss.items()}
    pre = torch.cat(pre, dim=0)
    gt = torch.cat(gt, dim=0)
    score = eval_Discriminator(pre, gt)

    return avg_loss, score

def Evimain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../../datasets/HM-TIR/")
    parser.add_argument('--max_patience', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--tau_init', type=float, default=1e-5)
    parser.add_argument('--tau_final', type=float, default=0.1 help="if normal: 0.1 if beta: 1")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../../ckpts")
    args = parser.parse_args()

    train_dataset = MainDataset(args.dataset, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    val_dataset = MainDataset(args.dataset, split='test')
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    model = DENet().to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_MB = total_params * 4 / (1024 * 1024)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M ({size_in_MB:.2f} MB)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1000, len(train_loader) * 1)
    warmup = LinearLR(opt, start_factor=0.001, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    ema = ModelEmaV2(model, decay=0.999)

    loss_fn = EvidentialBetaLoss()
    os.makedirs(args.checkpoint, exist_ok=True)

    best_score = 1e10
    patience = 0
    best_epoch = 0
    warmup_steps_tau = int(total_steps * args.warmup_ratio)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        avg_loss = defaultdict(float)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for lIr, _, label, _ in train_loader:
                lIr = lIr.to(args.device)
                label = label.to(args.device)

                opt.zero_grad(set_to_none=True)
                pre_label, alpha, beta, p_hat, S = model(lIr)

                factor = min(1.0, global_step / warmup_steps_tau)
                tau = args.tau_init + (args.tau_final - args.tau_init) * factor
                Loss, info = loss_fn(pre_label, alpha, beta, p_hat, label, tau)

                Loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                scheduler.step()

                for k, v in info.items():
                    avg_loss[k] += v
                pshow = {k: v / (pbar.n + 1) for k, v in avg_loss.items()}
                pshow["tau"] = tau
                pbar.set_postfix(pshow)
                pbar.update(1)
                global_step += 1

        # Validation
        eval_model = ema.module
        avg_loss, score = Validate(eval_model, val_loader, args, loss_fn)
        print(f"Val: \nLoss: {avg_loss} \n Score: {score}")

        current_score = avg_loss["l1"] * 10 - score["ACC"]
        if current_score < best_score:
            best_score = current_score
            torch.save(eval_model.state_dict(), f"{args.checkpoint}/DENet.pth")
            patience = 0
            best_epoch = epoch
        else:
            patience += 1
            if patience >= args.max_patience:
                print(f"Early stopping, best epoch: {best_epoch}")
                break



if __name__ == "__main__":
    Evimain()
