#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu-a100 --gres=gpu:1

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate /home/finn.vamosi/software/miniconda3/envs/DiT
python /home/finn.vamosi/3Brain/train.py $1
