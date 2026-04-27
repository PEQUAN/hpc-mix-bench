#!/bin/bash
#SBATCH --job-name=benchmarks
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"

source /software/python/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python3 --version

export JOBS=6
echo "JOBS=$JOBS"

bash run_benchmarks.sh false false true \
    sparse_lu \
    dense_lu \
    backprop \
    hotspot \
    particle_filter \
    srad_v2/


