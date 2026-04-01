import random
from collections import defaultdict

import cv2
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
from sklearn.metrics import accuracy_score, f1_score
import lpips
from typing import List, Tuple

def eval_Discriminator(logits: torch.Tensor,
                                   targets: torch.Tensor,
                                   thresholds=0.45):
    """
    logits: (N, C) raw logits
    targets: (N, C) in {0,1}
    thresholds: float or 1D array-like of length C; default 0.5
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    C = probs.shape[1]
    if thresholds is None:
        thresholds = np.full(C, 0.5)
    elif np.isscalar(thresholds):
        thresholds = np.full(C, float(thresholds))
    thresholds = np.asarray(thresholds).reshape(1, C)

    y_pred = (probs > thresholds).astype(int)

    subset_acc = accuracy_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "ACC": round(float(subset_acc),5),
        "F1": round(float(f1_macro),5),
    }

def eval_Discriminator1(pred,gt):
    probs = pred.cpu().numpy()
    y_true = gt.cpu().numpy().astype(int)
    subset_acc = accuracy_score(y_true, probs)
    f1_macro = f1_score(y_true, probs, average="macro", zero_division=0)

    return {
        "ACC": round(float(subset_acc),5),
        "F1": round(float(f1_macro),5),
    }

class augumentation(object):
    def __call__(self, inputs, target):
        if not isinstance(inputs, List):
            inputs = [inputs]

        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][:, ::-1]
            target = target[:, ::-1]
        return inputs, target
class MainDataset(Dataset):
    def __init__(self, root_dir, split='test', auguse=False):

        self.root_dir = Path(root_dir)
        self.split = split
        self.auguse = auguse
        if self.auguse:
            self.aug = augumentation()
        if split == "train":
            self.src_dir = self.root_dir / split / "patches" / "src"
            self.tgt_dir = self.root_dir / split /  "patches" / "tgt"
            self.label_dir = self.root_dir / split / "patches" / "labels.csv"
        else:
            self.src_dir = self.root_dir / split/  "src"
            self.tgt_dir = self.root_dir / split/ "tgt"
            self.label_dir = self.root_dir / split/ "labels.csv"


        self.src_files = sorted(list(self.src_dir.glob("*.png")))
        self.tgt_files = sorted(list(self.tgt_dir.glob("*.png")))
        self.labels = pd.read_csv(self.label_dir, index_col="name", dtype={"name": str})
        self.labels.index = self.labels.index.astype(str)

        assert len(self.src_files) == len(self.tgt_files)

    def __len__(self):
        return len(self.tgt_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        hIr_p = self.tgt_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
        hIr = np.array(cv2.imread(hIr_p, cv2.IMREAD_GRAYSCALE))
        label = np.array(self.labels.loc[lIr_p.name].tolist())
        if self.split == "train" and self.auguse:
            lIr, hIr = self.aug(lIr, hIr)
            lIr_tensor = torch.from_numpy(lIr[0].copy()).float().div(255.0).unsqueeze(0)
            hIr_tensor = torch.from_numpy(hIr.copy()).float().div(255.0).unsqueeze(0)
        else:
            lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
            hIr_tensor = torch.from_numpy(hIr).float().div(255.0).unsqueeze(0)
        label_tensor = torch.from_numpy(label).float()

        return lIr_tensor, hIr_tensor, label_tensor, name

class AWDataset(Dataset):
    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir /"test"/ "src"
        self.src_files = sorted(list(self.src_dir.glob("*.png")))


    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
        return lIr_tensor, name

class ExpertDataset(Dataset):
    def __init__(self, root_dir, split='train', expert_name=None):
        self.root_dir = Path(root_dir)
        self.split = split
        if split == "train":
            self.src_dir = self.root_dir / split / "patches" / expert_name
            self.tgt_dir = self.root_dir / split /  "patches" / "tgt"

        else:
            self.src_dir = self.root_dir / split / expert_name
            self.tgt_dir = self.root_dir / split / "tgt"


        self.src_files = sorted(list(self.src_dir.glob("*.png")))
        self.tgt_files = sorted(list(self.tgt_dir.glob("*.png")))

        assert len(self.src_files) == len(self.tgt_files)

    def __len__(self):
        return len(self.tgt_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        hIr_p = self.tgt_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
        hIr = np.array(cv2.imread(hIr_p, cv2.IMREAD_GRAYSCALE))

        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
        hIr_tensor = torch.from_numpy(hIr).float().div(255.0).unsqueeze(0)

        return lIr_tensor, hIr_tensor, name


class MainLosses(nn.Module):
    def __init__(self, alpha=1, beta=0.5, gamma=0.1, mode=1):
        super(MainLosses, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode


    def grad(self, x, y):
        def sobel(z):
            sobel_x = torch.tensor([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]], device=x.device).view(1, 1, 3, 3) / 8.0
            sobel_y = torch.tensor([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]], device=x.device).view(1, 1, 3, 3) / 8.0
            grad_x = F.conv2d(z, sobel_x, padding=1)
            grad_y = F.conv2d(z, sobel_y, padding=1)
            return grad_x, grad_y

        x_gx, x_gy = sobel(x)
        y_gx, y_gy = sobel(y)


        grad_diff = 0.5 * (F.l1_loss(x_gx, y_gx) + F.l1_loss(x_gy, y_gy))

        return grad_diff

    def forward(self, pred, target):
        info = defaultdict(float)
        if self.mode:
            loss = F.l1_loss(pred, target)
            info['Total'] = loss.item()
        else:
            l1 = F.l1_loss(pred, target)
            info['l1'] = l1.item()

            ssimn = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)
            info['SSIM'] = ssimn.item()

            grad = self.grad(pred, target)
            info['Grad'] = grad.item()

            loss = self.alpha * l1 + self.beta * ssimn + self.gamma * grad
            info['Total'] = loss.item()
        return loss, info

class EvidentialBetaLoss(nn.Module):

    def __init__(self, mode = "normal"):
        super().__init__()
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _expected_log_loss(target: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

        S = alpha + beta
        digamma_S = torch.digamma(S)
        loss = target * (digamma_S - torch.digamma(alpha)) + (1.0 - target) * (digamma_S - torch.digamma(beta))
        return loss  # (B,C)

    @staticmethod
    def beta_nll(target, alpha, beta, eps=1e-4):
        t = target.clamp(eps, 1.0 - eps)
        S = alpha + beta
        # -log BetaPDF(t | alpha, beta)
        nll = -((alpha - 1) * torch.log(t) + (beta - 1) * torch.log(1.0 - t)
                - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(S)))
        return nll

    @staticmethod
    def _kl_beta_uniform(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

        S = alpha + beta
        t1 = - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(S))
        t2 = (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))
        t3 = (beta  - 1.0) * (torch.digamma(beta)  - torch.digamma(S))
        return t1 + t2 + t3  # (B,C)

    def forward(self, plabel, alpha, beta, p_hat, target, anneal: float = 1.0) :
        info = defaultdict(float)
        gt_label = target[:,:3]

        loss_degrade = self.bce(plabel, gt_label)
        info['degrade'] = loss_degrade.item()

        if self.mode == "normal":
            # expected log-loss
            logloss = self._expected_log_loss(target[:,-1], alpha, beta)  # (B,C)
        else:
            logloss = self.beta_nll(target[:,-1], alpha, beta, eps=1e-4)

        kl = self._kl_beta_uniform(alpha, beta)                  # (B,C)

        S = alpha + beta
        info['S'] = S.mean().item()
        evi_error = torch.abs((target[:, -1] - p_hat) ** 2)
        evi_penalty = evi_error * (alpha + beta)
        loss_edl = torch.mean(logloss + anneal * kl + 0.1*evi_penalty)  # (B,C)
        info['edl'] = loss_edl.item()
        loss_l1 = F.l1_loss(p_hat, target[:,-1])
        info['l1'] = loss_l1.item()

        # loss_strength =  loss_l1 * 0.1 + loss_edl
        # info['strength'] = loss_strength.item()

        # all_loss = loss_degrade + loss_strength
        all_loss = loss_degrade + loss_edl
        info['total'] = all_loss.item()

        return all_loss, info




