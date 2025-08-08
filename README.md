# BPM — Batch Prototype Matching (Ultra-FGVC)

Quick start (wandb is forced to offline):

Training

- python train.py --dataset cotton80 --model resnet50 --epochs 5

Evaluation

- python evaluate.py --checkpoint checkpoints/best_cotton80_resnet50.pth --dataset cotton80

Notes

- Uses timm for models and transforms.
- Dataset splits train/val/test supported via parquet metadata.
