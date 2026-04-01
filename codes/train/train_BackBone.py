
import argparse
import os
import sys
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.utils import ModelEmaV2
import torch
torch.backends.cudnn.benchmark = True
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
def Validate(model, val_loader, args, loss_fn):
    model.eval()
    avg_loss = defaultdict(float)
    score = defaultdict(float)
    for lIr, hIr, label, name  in tqdm(val_loader, desc="Val"):
        lIr = lIr.to(args.device)
        hIr = hIr.to(args.device)
        label = label.to(args.device)

        feat = model(lIr, label)
        L, info = loss_fn(feat, hIr)

        for k, v in info.items():
            avg_loss[k] += v

        feat = feat.clamp(0.0, 1.0)
        res = evaluate(hIr, feat)
        for k, v in res.items():
            score[k] += v

        # pre = feat.squeeze().cpu().numpy()
        # pre = (pre * 255 + 0.5).astype(np.uint8)
        # name = name[0]
        # cv2.imwrite(str(os.path.join(args.pred_image, f"{name}.png")), pre)
    score = {k: v / len(val_loader) for k, v in score.items()}
    avg_loss = {k: v / len(val_loader) for k, v in avg_loss.items()}
    return avg_loss, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../../datasets/HM-TIR")
    parser.add_argument('--max_patience', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--SEwarmup', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../../ckpts")
    parser.add_argument('--pred_image', type=str, default="../../datasets/HM-TIR/SEGD")
    args = parser.parse_args()

    train_dataset =  MainDataset(args.dataset, split="train")
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

    model = IRRestorationBackbone().to(args.device)
    model.evi.load_state_dict(torch.load(f"{args.checkpoint}/DENet.pth"))
    for p in model.evi.parameters(): p.requires_grad = False
    if os.path.exists(f"{args.checkpoint}/CDRM.pth"):
        model.CDRM.load_state_dict(torch.load(f"{args.checkpoint}/CDRM.pth"))
        model.BDRM.load_state_dict(torch.load(f"{args.checkpoint}/BDRM.pth"))
        model.NDRM.load_state_dict(torch.load(f"{args.checkpoint}/NDRM.pth"))

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

    loss_fn = MainLosses().to(args.device)
    os.makedirs(args.checkpoint, exist_ok=True)
    # os.makedirs(args.pred_image, exist_ok=True)

    best_score = 1e10
    patience = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        avg_loss = defaultdict(float)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for lIr, hIr, label, _ in train_loader:
                lIr = lIr.to(args.device)
                hIr = hIr.to(args.device)
                label = label.to(args.device)
                if epoch > args.SEwarmup:
                    model.avgResdegarde = False
                opt.zero_grad(set_to_none=True)
                feat = model(lIr, label)
                L, info = loss_fn(feat, hIr)

                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                scheduler.step()

                for k, v in info.items():
                    avg_loss[k] += v
                pbar.set_postfix({k: v / (pbar.n + 1) for k, v in avg_loss.items()})
                pbar.update(1)


        eval_model = ema.module
        avg_loss, score = Validate(eval_model, val_loader, args, loss_fn)
        print(f"Val: \nLoss: {avg_loss} \nMetrics: {score}")
        current_score = (
                (1 - score["ssim"]) +
                (50 - score["psnr"]) / 50
        )
        if current_score < best_score:
            best_score = current_score
            torch.save(eval_model.state_dict(), f"{args.checkpoint}/model.pth")
            patience = 0
            best_epoch = epoch
        else:
            patience += 1
            if patience >= args.max_patience:
                print(f"Early stopping, best epoch: {best_epoch}")
                break



if __name__ == "__main__":
    main()
