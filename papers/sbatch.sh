#!/bin/bash
#SBATCH --job-name=benchmarks
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"

source /software/python/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

export JOBS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

bash run_benchmarks.sh true true false --parallel \
    sparse_lu \
    dense_lu \
    backprop \
    hotspot \
    particle_filter \
    srad_v2