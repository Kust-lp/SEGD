#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import glob
import json
import random
import shutil
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
from datetime import datetime

import pandas as pd
import pyiqa
import torch
from tqdm import tqdm
from PIL import Image
from albumentations import GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast, Compose
metric = pyiqa.create_metric('ssim', device='cuda')

def stripe_noise(image, **params):
    if random.random() < 0.5:
        # g = np.random.randn(1, image.shape[1]) * (np.random.uniform(0.03, 0.1))
        # b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
        g = np.random.randn(1, image.shape[1]) * (np.random.uniform(0.03, 0.07))
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 3)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.uniform(0.03, 0.07))
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 3)
        # g = np.random.randn(image.shape[0], 1) * (np.random.uniform(0.03, 0.1))
        # b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image

def nonuniformity_optical(image, **params):
    h, w = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 55 + 1)
    # delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image


def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(std_range=(5 / 255.0, 15 / 255.0), p=1),
        # GaussNoise(std_range=(5 / 255.0, 20 / 255.0), p=1),
    ], p=p)

def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(-0.8, -0.4), p=p)
    # return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)

def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 17), sigma_limit=(1, 2), p=1)
        # GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1)
    ], p=p)





def apply_degradations_random(img_list, degrad_names):
    op_map = {
        "contrast": LC,
        "blur": Blur,
        "noise": Noise,
    }

    order = degrad_names[:]
    random.shuffle(order)

    for degrad in order:
        img_list = op_map[degrad]()(**img_list)

    dweight = [
        1.0 if "contrast" in degrad_names else 0.0,
        1.0 if "blur" in degrad_names else 0.0,
        1.0 if "noise" in degrad_names else 0.0,
    ]

    return img_list, dweight, order


def HMTIR(args):
    random.seed(42)
    dataset_dir = args.input_dir
    all_imgs = glob.glob(os.path.join(dataset_dir, "imgs", "*.png"))
    random.shuffle(all_imgs)
    train_imgP = all_imgs[:int(len(all_imgs) * 0.8)]
    test_imgP = all_imgs[int(len(all_imgs) * 0.8):]

    for split, imgP in zip(['train', 'test'], [train_imgP, test_imgP]):
        op = os.path.join(args.output_dir, split)
        src_path = os.path.join(op, "src")
        os.makedirs(src_path, exist_ok=True)
        tgt_path = os.path.join(op, "tgt")
        os.makedirs(tgt_path, exist_ok=True)
        opc = os.path.join(op, "Contrast")
        os.makedirs(opc, exist_ok=True)
        opb = os.path.join(op, "Blur")
        os.makedirs(opb, exist_ok=True)
        opn = os.path.join(op, "Noise")
        os.makedirs(opn, exist_ok=True)

        labels = {}
        random.shuffle(imgP)
        tbar = tqdm(imgP, desc=f"HM-TIR ({split})")
        count = len(imgP)
        for i, imgp in enumerate(tbar):
            img_name = os.path.basename(imgp)
            img = Image.open(imgp).convert('L')
            img = np.array(img, dtype=np.uint8)
            img_list = {'image': img}
            img_c_copy = img_list.copy()
            img_b_copy = img_list.copy()
            img_n_copy = img_list.copy()

            if i <= count * 0.2:
                degrad = random.choice(args.D_type)
                img_list, dweight, order = apply_degradations_random(img_list, [degrad])

            elif i <= count * 0.5:
                degrad_pair = random.sample(args.D_type, 2)
                img_list, dweight, order = apply_degradations_random(img_list, degrad_pair)

            else:
                degrad_triple = args.D_type[:]
                img_list, dweight, order = apply_degradations_random(img_list, degrad_triple)


            gt = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)
            deg = torch.from_numpy(img_list['image']).float().div(255.0).unsqueeze(0).unsqueeze(0)
            dweight.append(round(1- metric(deg, gt).item(),3))
            cor_img = Image.fromarray(img_list['image']).convert('L')


            img_c_copy =  LC()(**img_c_copy)
            ic = Image.fromarray(img_c_copy['image']).convert('L')

            img_b_copy = Blur()(**img_b_copy)
            ib = Image.fromarray(img_b_copy['image']).convert('L')

            img_n_copy = Noise()(**img_n_copy)
            ino = Image.fromarray(img_n_copy['image']).convert('L')

            labels[f"{img_name}_{str(dweight)}.png"] = dweight
            cor_img.save(os.path.join(src_path, f"{img_name}_{str(dweight)}.png"))
            ic.save(os.path.join(opc, f"{img_name}_{str(dweight)}.png"))
            ib.save(os.path.join(opb, f"{img_name}_{str(dweight)}.png"))
            ino.save(os.path.join(opn, f"{img_name}_{str(dweight)}.png"))
            shutil.copy(os.path.join(args.input_dir,"imgs", img_name), os.path.join(tgt_path, f"{img_name}_{str(dweight)}.png"))

        df = pd.DataFrame.from_dict(labels, orient="index", columns=["contrast", "blur", "noise","strength"])
        df.index = df.index.astype(str)
        df.to_csv(os.path.join(op, "labels.csv"), index_label="name", index=True)


def NightTIR(args):
    random.seed(42)
    dataset_dir = args.input_dir
    all_imgs = glob.glob(os.path.join(dataset_dir, "imgs", "*.JPG"))
    random.shuffle(all_imgs)

    op = args.output_dir
    src_path = os.path.join(args.output_dir, "src")
    os.makedirs(src_path, exist_ok=True)
    tgt_path = os.path.join(args.output_dir, "tgt")
    os.makedirs(tgt_path, exist_ok=True)

    labels = {}
    tbar = tqdm(all_imgs, desc="Night-TIR")
    count = len(all_imgs)
    for i, imgp in enumerate(tbar):
        img_name = f"{i:03d}"
        img = Image.open(imgp).convert('L')
        img.save(os.path.join(tgt_path, f"{img_name}.png"))
        img = np.array(img, dtype=np.uint8)
        img_list = {'image': img}

        if i <= count * 0.2:
            degrad = random.choice(args.D_type)
            img_list, dweight, order = apply_degradations_random(img_list, [degrad])

        elif i <= count * 0.5:
            degrad_pair = random.sample(args.D_type, 2)
            img_list, dweight, order = apply_degradations_random(img_list, degrad_pair)

        else:
            degrad_triple = args.D_type[:]
            img_list, dweight, order = apply_degradations_random(img_list, degrad_triple)

        gt = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)
        deg = torch.from_numpy(img_list['image']).float().div(255.0).unsqueeze(0).unsqueeze(0)
        dweight.append(round(1 - metric(deg, gt).item(), 3))
        cor_img = Image.fromarray(img_list['image']).convert('L')

        labels[f"{img_name}.png"] = dweight
        cor_img.save(os.path.join(src_path, f"{img_name}.png"))

    df = pd.DataFrame.from_dict(labels, orient="index", columns=["contrast", "blur", "noise", "strength"])
    df.index = df.index.astype(str)
    df.to_csv(os.path.join(op, "labels.csv"), index_label="name", index=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../../datasets/")
    p.add_argument("--D_type", type=list, default=["contrast", "blur", "noise"])
    args = p.parse_args()
    for dataset in ["HM-TIR", "Night-TIR"]:

        args.input_dir = os.path.join(args.data_dir, dataset)
        if dataset == "HM-TIR":
            args.output_dir = os.path.join(args.data_dir, dataset)
            HMTIR(args)
        else:
            args.output_dir = os.path.join(args.data_dir, dataset, "test")
            NightTIR(args)
