#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader
import timm

from src.dataset.ufgvc import UFGVCDataset
from src.bpm.model import TimmtBPM

import wandb


def build_transforms(model_name: str, img_size: int):
    m = timm.create_model(model_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(m.pretrained_cfg)
    if img_size:
        data_cfg = {**data_cfg, 'input_size': (3, img_size, img_size)}
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cotton80')
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project='BPM', config=vars(args), mode='offline')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = build_transforms(args.model, args.img_size)
    val_ds = UFGVCDataset(args.dataset, args.root, 'val', transform, download=True)
    test_ds = UFGVCDataset(args.dataset, args.root, 'test', transform, download=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get('classes', len(val_ds.classes))
    model = TimmtBPM(args.model, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    def eval_split(loader, name: str):
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss_sum += loss.item()
                total += labels.size(0)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
        acc = 100.0 * correct / total if total else 0.0
        loss_avg = loss_sum / max(1, len(loader))
        wandb.log({f'{name}/acc': acc, f'{name}/loss': loss_avg})
        print(f'{name}: acc={acc:.2f} loss={loss_avg:.4f}')

    eval_split(val_loader, 'val')
    eval_split(test_loader, 'test')


if __name__ == '__main__':
    main()
