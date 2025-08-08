#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=00:30:00
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info-v100.txt
source /scratch/rp06/sl5952/BPM/.venv/bin/activate

cd ..
# Run training with BPM models
python3 train.py --dataset cotton80 --model convnextv2_tiny.fcmae_ft_in22k_in1k_384 --epochs 50 --img-size 384 >> results/cotton80_convnextv2_tiny_384.log

