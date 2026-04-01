# -*- coding: utf-8 -*-
import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from utils.evaluate import evaluate
from model.model import IRRestorationBackbone
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="../datasets")
    parser.add_argument('--datasetname', type=str, default="HM-TIR", help="HM-TIR,Night-TIR, AWMM")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../ckpts")
    args = parser.parse_args()

    data_path = os.path.join(args.datasets, args.datasetname)
    if args.datasetname == "AWMM":
        val_dataset = AWDataset(data_path)
    else:
        val_dataset = MainDataset(data_path, split='test')
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    model = IRRestorationBackbone().to(args.device)
    model.load_state_dict(torch.load(f"{args.checkpoint}/model.pth"))
    model.eval()
    model.avgResdegarde = False

    score = defaultdict(float)
    single_score = defaultdict(float)
    snum = 0
    double_score = defaultdict(float)
    dnum = 0
    three_score = defaultdict(float)
    tnum = 0
    if args.datasetname == "AWMM":
        for lIr, name in tqdm(val_loader, desc="Val"):
            lIr = lIr.to(args.device)

            feat = model(lIr)

            feat = feat.clamp(0.0, 1.0)
            res = evaluate(feat)
            for k, v in res.items():
                score[k] += v

    else:
        for lIr, hIr, label, name in tqdm(val_loader, desc="Val"):
            lIr = lIr.to(args.device)
            hIr = hIr.to(args.device)
            label = label.to(args.device)

            feat = model(lIr, label)

            feat = feat.clamp(0.0, 1.0)
            res = evaluate(feat, hIr)
            for k, v in res.items():
                score[k] += v

            if label[0,:3].sum().item() == 1.:
                for k, v in res.items():
                    single_score[k] += v
                snum += 1
            if label[0,:3].sum().item() == 2.:
                for k, v in res.items():
                    double_score[k] += v
                dnum += 1
            if label[0,:3].sum().item() == 3.:
                for k, v in res.items():
                    three_score[k] += v
                tnum += 1

    single_score = {k: v / snum for k, v in single_score.items()} if snum >0 else None
    double_score = {k: v / dnum for k, v in double_score.items()} if dnum >0 else None
    three_score = {k: v / tnum for k, v in three_score.items()} if tnum >0 else None
    score = {k: round((v+double_score[k]+three_score[k]) / 3,5) for k, v in single_score.items()}

    print(f"Val:\nAvg Metrics: {score} \nSingle Metrics: {single_score} \nDouble Metrics: {double_score} \nThree Metrics: {three_score}")



if __name__ == "__main__":
    main()
