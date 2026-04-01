
import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm



def compute_starts(L: int, patch: int, stride: int, cover_edges: bool):

    assert L >= patch, "Image size is too small!"

    starts = list(range(0, L - patch + 1, stride))
    if cover_edges:
        last = L - patch
        if starts[-1] != last:
            starts.append(last)
    return starts

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_patch(img, y, x, patch_size, out_dir, base):
    patch = img[y:y+patch_size, x:x+patch_size]
    out_name = f"{base}_y{y:05d}_x{x:05d}.png"
    cv2.imwrite(os.path.join(out_dir, out_name), patch)
    return out_name

def patchify_pair_dir(src, tgt,c_src,b_src,n_src, out_src, out_tgt,out_c_src, out_b_src, out_n_src, P, S, cover):

    ensure_dir(out_src)
    ensure_dir(out_tgt)
    ensure_dir(out_c_src)
    ensure_dir(out_b_src)
    ensure_dir(out_n_src)

    p_num = 0
    pairs = []
    img_list = sorted([f for f in os.listdir(tgt) if f.lower().endswith(".png")])
    for name in tqdm(img_list, desc=f"Patchifying {os.path.basename(os.path.dirname(out_src))}"):
        base = os.path.splitext(name)[0]
        src_path = os.path.join(src, name)
        tgt_path = os.path.join(tgt, name)
        c_src_path = os.path.join(c_src, name)
        b_src_path = os.path.join(b_src, name)
        n_src_path = os.path.join(n_src, name)
        if not (os.path.isfile(src_path) and os.path.isfile(tgt_path)):
            continue

        src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE)
        c_src_img = cv2.imread(c_src_path, cv2.IMREAD_GRAYSCALE)
        b_src_img = cv2.imread(b_src_path, cv2.IMREAD_GRAYSCALE)
        n_src_img = cv2.imread(n_src_path, cv2.IMREAD_GRAYSCALE)
        H, W = src_img.shape
        assert tgt_img.shape == (H, W), f"Size mismatch: {src_path} vs {tgt_path}"

        ys = compute_starts(H, P, S, cover)
        xs = compute_starts(W, P, S, cover)

        patch_names = []
        for y in ys:
            y = min(y, H - P) if H >= P else 0
            for x in xs:
                x = min(x, W - P) if W >= P else 0
                if y + P <= H and x + P <= W:
                    save_patch(src_img, y, x, P, out_src, base)
                    save_patch(c_src_img, y, x, P, out_c_src, base)
                    save_patch(b_src_img, y, x, P, out_b_src, base)
                    save_patch(n_src_img, y, x, P, out_n_src, base)
                    out_name = save_patch(tgt_img, y, x, P, out_tgt, base)
                    patch_names.append(out_name)
                    p_num += 1
        if patch_names:
            pairs.append((base, patch_names))
    return pairs, p_num




def propagate_labels_to_patches(labels_csv, patch_pairs, out_labels_csv):
    df = pd.read_csv(labels_csv, index_col="name", dtype={"name": str})
    df.index = df.index.astype(str)
    rows = {}
    for base, patch_list in patch_pairs:
        base = base+".png"
        if base not in df.index:
            continue
        onehot = df.loc[base].tolist()
        for fname in patch_list:
            stem = os.path.splitext(fname)[0]
            rows[f"{stem}.png"] = onehot
    if rows:
        out_df = pd.DataFrame.from_dict(rows, orient="index", columns=["contrast", "blur", "noise", "strength"])
        out_df.to_csv(out_labels_csv, index_label="name", index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="../../datasets/HM-TIR/train/")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--cover_edges", type=int, default=0, help="1 covers to the right/bottom edge; 0 discards incomplete blocks")
    args = ap.parse_args()

    P = args.patch_size
    S = args.stride
    cover = bool(args.cover_edges)
    src = os.path.join(args.root, "src")
    tgt = os.path.join(args.root, "tgt")
    c_src = os.path.join(args.root, "Contrast")
    b_src = os.path.join(args.root, "Blur")
    n_src = os.path.join(args.root, "Noise")
    out_src = os.path.join(args.root, "patches", "src")
    out_tgt = os.path.join(args.root, "patches", "tgt")
    out_c_src = os.path.join(args.root, "patches", "Contrast")
    out_b_src = os.path.join(args.root, "patches", "Blur")
    out_n_src = os.path.join(args.root, "patches", "Noise")
    train_labels_csv = os.path.join(args.root,  "labels.csv")

    pairs, p_num = patchify_pair_dir(src, tgt,c_src,b_src,n_src, out_src, out_tgt,out_c_src, out_b_src, out_n_src, P, S, cover)
    print(f"Patches for train: {p_num}")
    out_train_labels = os.path.join(args.root, "patches", "labels.csv")

    if os.path.isfile(train_labels_csv):
        propagate_labels_to_patches(train_labels_csv, pairs, out_train_labels)
    else:
        print(f"[Warn] train labels not found: {train_labels_csv}")








if __name__ == "__main__":

    main()
