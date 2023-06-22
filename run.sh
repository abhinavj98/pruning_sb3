#!/bin/bash
#SBATCH -J ppo
#SBATCH -A eecs
#SBATCH -p dgx
#SBATCH -o ppo.out
#SBATCH -e ppo.err
#SBATCH -t 7-00:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:1
module load cuda/11.6
cd /nfs/stak/users/jainab/hpc-share/codes/pruning_sb3
python train_ppo_test.py