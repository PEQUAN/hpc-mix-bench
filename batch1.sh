#!/bin/bash

#SBATCH --job-name=exemple
#SBATCH --nodes=1
#SBATCH --constraint=amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=5
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

