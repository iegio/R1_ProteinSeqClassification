#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=5:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=4

python multi_esm.py
