#!/bin/bash
#SBATCH --job-name=benchmarks
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"


echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python3 --version
python3 -c "import cadnaPromise; print('cadnaPromise OK')"

export JOBS=6
echo "JOBS=$JOBS"

bash run_benchmarks.sh true true false --parallel \
    sparse_lu \
    dense_lu \
    backprop \
    hotspot \
    particle_filter \
    srad_v2