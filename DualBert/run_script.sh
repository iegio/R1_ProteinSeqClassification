#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=4

python multi_bert_oct.py
