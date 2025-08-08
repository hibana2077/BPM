#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=00:30:00
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info-a100.txt
source /scratch/rp06/sl5952/BPM/.venv/bin/activate

cd ..
# Run training with BPM models
python3 train.py --dataset cotton80 --model efficientnet_b5.sw_in12k_ft_in1k --epochs 50 --img-size 448 >> results/cotton80_efficientnet_b5_448.log