#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

# Dataset
from src.dataset.ufgvc import UFGVCDataset
from src.bpm.model import TimmtBPM
from src.bpm.losses import (
    invariance_kl, embedding_l2, uniformity_kl, byol_loss, vicreg_variance_covariance
)
from src.bpm.ema import EMA
from src.bpm.prototype import batch_mean, MomentumPrototype

import wandb


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    dataset: str = 'cotton80'
    root: str = './data'
    model: str = 'resnet50'
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_ce: float = 1.0
    alpha_inv: float = 0.5
    beta_uni: float = 0.2
    gamma_sd: float = 1.0
    ema: float = 0.99
    proj_dim: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    proto_mode: str = 'mean'  # 'mean' | 'momentum'
    momentum_tau: float = 0.99


def build_transforms(model_name: str, img_size: int):
    # Create timm transforms per docs
    m = timm.create_model(model_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(m.pretrained_cfg)

    # Override input_size if requested
    if img_size:
        data_cfg = {**data_cfg, 'input_size': (3, img_size, img_size)}
    transform_train = timm.data.create_transform(**data_cfg, is_training=True)
    transform_val = timm.data.create_transform(**data_cfg, is_training=False)
    return transform_train, transform_val


def build_dataloaders(cfg: Config, transform_train, transform_val):
    train_ds = UFGVCDataset(cfg.dataset, cfg.root, 'train', transform_train, download=True)
    val_ds = UFGVCDataset(cfg.dataset, cfg.root, 'val', transform_val, download=True)
    test_ds = UFGVCDataset(cfg.dataset, cfg.root, 'test', transform_val, download=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, len(train_ds.classes)


def train_one_epoch(model_s: TimmtBPM, model_t: TimmtBPM, ema: EMA, loader: DataLoader, optimizer: torch.optim.Optimizer, cfg: Config, device: str, epoch: int, mproto: MomentumPrototype | None = None):
    model_s.train()
    model_t.eval()
    total = 0
    correct = 0
    running = {'ce': 0.0, 'inv': 0.0, 'uni': 0.0, 'sd': 0.0, 'varcov': 0.0, 'total': 0.0}

    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Build prototype and residuals
        x_bar = batch_mean(images)  # (1, C, H, W)
        if cfg.proto_mode == 'momentum':
            assert mproto is not None
            x_bar = mproto.update(x_bar)
        residual = images - x_bar  # broadcast to (B, C, H, W)

        # Two views with timm transforms already applied, so we simulate views by slight noise/dropout
        # For simplicity, reuse the tensors (augmentations already applied in dataset)
        vA = images
        vB = residual
        vP = x_bar.expand_as(images)  # prototype view per-sample

        # Forward passes
        logits_A, emb_A = model_s.forward_with_head(vA)
        logits_B, emb_B = model_s.forward_with_head(vB)
        with torch.no_grad():
            _, emb_A_t = model_t.forward_with_head(vA)
            z_A_t = model_t.proj(emb_A_t)

        # Losses
        ce = F.cross_entropy(logits_A, labels) + cfg.lambda_ce * F.cross_entropy(logits_B, labels)
        inv = invariance_kl(logits_A, logits_B)
        uni = uniformity_kl(model_s(vP).mean(dim=0, keepdim=True))  # encourage high entropy on prototype
        sd = byol_loss(model_s.proj(emb_B), z_A_t)
        varcov = vicreg_variance_covariance(emb_A, emb_B)

        loss = ce + cfg.alpha_inv * inv + cfg.beta_uni * uni + cfg.gamma_sd * sd + varcov

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model_s)

        # Metrics
        total += labels.size(0)
        preds = logits_A.argmax(dim=-1)
        correct += (preds == labels).sum().item()

        running['ce'] += ce.item()
        running['inv'] += inv.item()
        running['uni'] += uni.item()
        running['sd'] += sd.item()
        running['varcov'] += varcov.item()
        running['total'] += loss.item()

    for k in running:
        running[k] /= len(loader)
    acc = 100.0 * correct / total if total else 0.0

    wandb.log({f'train/{k}': v for k, v in running.items()} | {'train/acc': acc, 'epoch': epoch})
    return acc, running


def evaluate(model: TimmtBPM, loader: DataLoader, device: str, split_name: str = 'val', epoch: int = 0):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss_sum += loss.item()
            total += labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
    acc = 100.0 * correct / total if total else 0.0
    loss_avg = loss_sum / max(1, len(loader))
    wandb.log({f'{split_name}/acc': acc, f'{split_name}/loss': loss_avg, 'epoch': epoch})
    return acc, loss_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cotton80')
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda-ce', type=float, default=1.0)
    parser.add_argument('--alpha-inv', type=float, default=0.5)
    parser.add_argument('--beta-uni', type=float, default=0.2)
    parser.add_argument('--gamma-sd', type=float, default=1.0)
    parser.add_argument('--ema', type=float, default=0.99)
    parser.add_argument('--epochs-test-only', action='store_true')
    parser.add_argument('--proto-mode', type=str, default='mean', choices=['mean','momentum'])
    parser.add_argument('--momentum-tau', type=float, default=0.99)
    args = parser.parse_args()

    # wandb offline mode
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project='BPM', config=vars(args), mode='offline')

    cfg = Config(
        dataset=args.dataset, root=args.root, model=args.model, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, lambda_ce=args.lambda_ce,
        alpha_inv=args.alpha_inv, beta_uni=args.beta_uni, gamma_sd=args.gamma_sd,
    ema=args.ema,
    proto_mode=args.proto_mode,
    momentum_tau=args.momentum_tau
    )

    set_seed(42)

    device = cfg.device
    transform_train, transform_val = build_transforms(cfg.model, cfg.img_size)
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg, transform_train, transform_val)

    # Models
    model_s = TimmtBPM(cfg.model, num_classes=num_classes, pretrained=True, proj_dim=cfg.proj_dim).to(device)
    model_t = TimmtBPM(cfg.model, num_classes=num_classes, pretrained=True, proj_dim=cfg.proj_dim).to(device)
    ema = EMA(cfg.ema)
    ema._init(model_s)
    ema.copy_to(model_t)

    optimizer = torch.optim.AdamW(model_s.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = 0.0
    best_ckpt_path = None

    if args.epochs_test_only:
        val_acc, _ = evaluate(model_s, val_loader, device, 'val', 0)
        test_acc, _ = evaluate(model_s, test_loader, device, 'test', 0)
        print(f'[EvalOnly] val_acc={val_acc:.2f} test_acc={test_acc:.2f}')
        return

    mproto = MomentumPrototype(cfg.momentum_tau) if cfg.proto_mode == 'momentum' else None

    for epoch in range(1, cfg.epochs + 1):
        train_acc, train_logs = train_one_epoch(model_s, model_t, ema, train_loader, optimizer, cfg, device, epoch, mproto)
        ema.copy_to(model_t)  # refresh teacher params from EMA shadow
        val_acc, _ = evaluate(model_s, val_loader, device, 'val', epoch)
        scheduler.step()
        print(f'Epoch {epoch}: train_acc={train_acc:.2f} val_acc={val_acc:.2f}')
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                'model': model_s.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'classes': num_classes,
                'config': vars(cfg)
            }
            os.makedirs('checkpoints', exist_ok=True)
            best_ckpt_path = f'checkpoints/best_{cfg.dataset}_{cfg.model}.pth'
            torch.save(ckpt, best_ckpt_path)
    # Evaluate with best checkpoint (if any)
    if best_ckpt_path is not None and os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model_s.load_state_dict(ckpt['model'])
        test_acc, _ = evaluate(model_s, test_loader, device, 'test', ckpt.get('epoch', cfg.epochs))
        print(f"[Best @ epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0.0):.2f}] Test acc: {test_acc:.2f}")
    else:
        test_acc, _ = evaluate(model_s, test_loader, device, 'test', cfg.epochs)
        print(f'Test acc (last): {test_acc:.2f}')


if __name__ == '__main__':
    main()
