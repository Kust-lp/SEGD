import os

import cv2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pyiqa
import torch
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
metrics = {
        'psnr': pyiqa.create_metric('psnr', device=device),
        'ssim': pyiqa.create_metric('ssim', device=device),
        # 'lpips': pyiqa.create_metric('lpips', device=device),
        # 'niqe': pyiqa.create_metric('niqe', device=device),
        # 'nima': pyiqa.create_metric('nima', device=device),
        # 'musiq': pyiqa.create_metric('musiq', device=device),


    }

def evaluate(HRs, SRs=None):

    results = {}


    with torch.no_grad():
        for name, metric in metrics.items():
            if name in ['psnr', 'ssim']:
                score = metric(HRs, SRs)
            elif name == 'lpips':
                HRs_lpips = HRs.repeat(1, 3, 1, 1)
                SRs_lpips = SRs.repeat(1, 3, 1, 1)
                score = metric(HRs_lpips, SRs_lpips)
            else:
                HRs_lpips = HRs.repeat(1, 3, 1, 1)
                score = metric(HRs_lpips)
            results[name] = round(float(score.item()),4)
    return results




