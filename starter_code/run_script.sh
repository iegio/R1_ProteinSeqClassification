#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=4

python seq_training.py