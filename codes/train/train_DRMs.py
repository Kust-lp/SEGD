"""
Note: Warm-up/Pre-training DRMs is optional.
"""
import argparse
import os
import sys
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.utils import ModelEmaV2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from model.DRMs import *
from utils.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def Validate(model, val_loader, args, loss_fn):
    model.eval()
    avg_loss = defaultdict(float)
    score = defaultdict(float)
    for lIr, hIr, name  in tqdm(val_loader, desc="Val"):
        lIr = lIr.to(args.device)
        hIr = hIr.to(args.device)

        out = model(lIr)
        L, info = loss_fn(out, hIr)
        for k, v in info.items():
            avg_loss[k] += v

        feat = out.clamp(0.0, 1.0)
        res = evaluate(hIr, feat)
        for k, v in res.items():
            score[k] += v


    score = {k: v / len(val_loader) for k, v in score.items()}
    avg_loss = {k: v / len(val_loader) for k, v in avg_loss.items()}
    return avg_loss, score

def Emain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../../datasets/HM-TIR")
    parser.add_argument('--max_patience', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../../ckpts")
    parser.add_argument("--DRMs", type=list, default=["Contrast", "Blur", "Noise"])
    parser.add_argument('--pred_image', type=str, default="../../datasets/HM-TIR/Single/")
    args = parser.parse_args()


    for DRM in args.DRMs:
        args.pred_image = os.path.join(args.pred_image, DRM)
        os.makedirs(args.pred_image, exist_ok=True)
        print(f"Training {DRM} DRM...")

        train_dataset = ExpertDataset(args.dataset, split='train', expert_name=DRM)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
        val_dataset = ExpertDataset(args.dataset, split='test', expert_name=DRM)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )


        model = DRMsBackbone().to(args.device)
        model.evi.load_state_dict(torch.load(f"{args.checkpoint}/DENet.pth"))
        for p in model.encoder.parameters(): p.requires_grad = False
        for p in model.film.parameters(): p.requires_grad = False
        # for p in model.mlp.parameters(): p.requires_grad = False
        for p in model.evi.parameters(): p.requires_grad = False
        for p in model.decoder.parameters(): p.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_in_MB = total_params * 4 / (1024 * 1024)
        print(f"Trainable parameters: {total_params / 1e6:.2f}M ({size_in_MB:.2f} MB)")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = max(1000, len(train_loader) * 1)
        warmup = LinearLR(opt, start_factor=0.001, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=(total_steps - warmup_steps), eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
        ema = ModelEmaV2(model, decay=0.999)

        loss_fn =  MainLosses().to(args.device)
        os.makedirs(args.checkpoint, exist_ok=True)

        best_score = 1e10
        patience = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            model.train()
            avg_loss = defaultdict(float)

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
                for lIr, hIr, _ in train_loader:
                    lIr = lIr.to(args.device)
                    hIr = hIr.to(args.device)

                    opt.zero_grad(set_to_none=True)
                    out = model(lIr)
                    L, info = loss_fn(out, hIr)

                    L.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ema.update(model)
                    scheduler.step()

                    for k,v in info.items():
                        avg_loss[k] += v
                    pbar.set_postfix({k: v / (pbar.n + 1) for k, v in avg_loss.items()})
                    pbar.update(1)

            eval_model = ema.module
            avg_loss, score = Validate(eval_model, val_loader, args, loss_fn)
            print(f"Val: \nLoss: {avg_loss} \nMetrics: {score}")
            current_score = (
                        avg_loss["Total"] +
                        (1 - score["ssim"]) +
                        (50 - score["psnr"]) / 50
                )
            if current_score < best_score:
                    eval_model = ema.module
                    best_score = current_score
                    torch.save(eval_model.expert.state_dict(), f"{args.checkpoint}/{DRM[0]}DRM.pth")
                    patience = 0
                    best_epoch = epoch
            else:
                    patience += 1
                    if patience >= args.max_patience:
                        print(f"Early stopping, best epoch: {best_epoch}")
                        break



if __name__ == "__main__":
    Emain()
