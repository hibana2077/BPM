#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=32GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -N J0001

module load cuda/12.6.2
source /scratch/rp06/sl5952/BPM/.venv/bin/activate
cd ../..
python3 train.py --dataset cotton80 --model tiny_vit_21m_384.dist_in22k_ft_in1k --epochs 100 --img-size 384 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 --ema 0.99 --alpha-inv 0.0 --beta-uni 0.0 --gamma-sd 0.0 --proto-mode mean >> results/cotton80_J0001.log
