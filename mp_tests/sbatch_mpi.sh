#!/bin/bash
#SBATCH --job-name=benchmarks-mpi
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /software/python/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "======================================"
echo "Job started      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node             : $(hostname)"
echo "Workdir          : $(pwd)"
echo "Python           : $(which python)"
python3 --version
echo "SLURM_JOB_ID     : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NAME   : ${SLURM_JOB_NAME:-N/A}"
echo "SLURM_NTASKS     : ${SLURM_NTASKS:-N/A}"
echo "OMP_NUM_THREADS  : $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS  : $MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS : $OPENBLAS_NUM_THREADS"
echo "======================================"

mpirun -np ${SLURM_NTASKS} python3 mpi_runner.py true true false \
    sparse_lu \
    dense_lu \
    backprop \
    hotspot \
    particle_filter \
    srad_v2

echo "======================================"
echo "Job finished     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"