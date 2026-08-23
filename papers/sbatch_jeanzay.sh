#!/bin/bash

#SBATCH --job-name=benchmarks-jeanzay
#SBATCH --constraint=jeanzay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_jeanzay-t3
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 如果需要指定项目，请加：
# #SBATCH --account=YOUR_PROJECT@jeanzay

set -euo pipefail

###############################################################################
# 1. 必须确认这是 Slurm compute job
###############################################################################

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: This script must be submitted with sbatch."
    echo "Do NOT run it directly with bash on a frontend node."
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "SLURM job:    $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "============================================================"

###############################################################################
# 2. Jean Zay environment
###############################################################################

module purge

# module load arch/h100
# module load cuda/12.4.1
###############################################################################
# 3. JOBSCRATCH is mandatory
#
#    Do NOT silently fall back to /tmp.
###############################################################################

if [[ -z "${JOBSCRATCH:-}" ]]; then
    echo "ERROR: JOBSCRATCH is not defined."
    echo "This job does not appear to have the expected Jean Zay job scratch."
    exit 1
fi

# One private temp root for this job.
PROMISE_TMPDIR="${JOBSCRATCH}/promise_${SLURM_JOB_ID}"

mkdir -p "$PROMISE_TMPDIR"
chmod 700 "$PROMISE_TMPDIR"

###############################################################################
# 4. Redirect all common temporary/cache locations
###############################################################################

export PROMISE_TMPDIR

export TMPDIR="$PROMISE_TMPDIR"
export TMP="$PROMISE_TMPDIR"
export TEMP="$PROMISE_TMPDIR"

export XDG_CACHE_HOME="$PROMISE_TMPDIR/xdg_cache"
export MPLCONFIGDIR="$PROMISE_TMPDIR/matplotlib"
export CUDA_CACHE_PATH="$PROMISE_TMPDIR/cuda_cache"
export PYTHONPYCACHEPREFIX="$PROMISE_TMPDIR/pycache"

mkdir -p \
    "$XDG_CACHE_HOME" \
    "$MPLCONFIGDIR" \
    "$CUDA_CACHE_PATH" \
    "$PYTHONPYCACHEPREFIX"

###############################################################################
# 5. Clean only OUR job temp directory
###############################################################################

cleanup_tmp() {
    echo
    echo "Cleaning PROMISE temporary directory:"
    echo "  $PROMISE_TMPDIR"

    # Safety guard: never rm an unexpected path.
    if [[ "$PROMISE_TMPDIR" == "$JOBSCRATCH"/promise_"$SLURM_JOB_ID" ]]; then
        rm -rf -- "$PROMISE_TMPDIR"
    else
        echo "WARNING: refusing to remove unexpected path: $PROMISE_TMPDIR"
    fi
}

trap cleanup_tmp EXIT TERM INT

###############################################################################
# 6. Python environment
###############################################################################

USER_BASE="$(python3 -m site --user-base)"
USER_SITE="$(python3 -m site --user-site)"
REPO_CADNAPROMISE="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)/cadnaPromise"

export PATH="$USER_BASE/bin:$PATH"
export PYTHONPATH="$REPO_CADNAPROMISE:$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"

###############################################################################
# 7. Parallelism
###############################################################################

export JOBS="${JOBS:-6}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

###############################################################################
# 8. Diagnostics
###############################################################################

echo
echo "================ Environment ================="
echo "Node:            $(hostname)"
echo "Workdir:         $(pwd)"
echo "Python:          $(command -v python3)"
python3 --version

echo "User base:       $USER_BASE"
echo "User site:       $USER_SITE"
echo "Repo PROMISE:    $REPO_CADNAPROMISE"

echo "JOBSCRATCH:      $JOBSCRATCH"
echo "PROMISE_TMPDIR:  $PROMISE_TMPDIR"
echo "TMPDIR:          $TMPDIR"
echo "JOBS:            $JOBS"

echo
echo "Filesystem:"
df -h "$JOBSCRATCH" || true

echo
echo "Checking CLI tools..."

command -v promise
command -v activate-promise

###############################################################################
# 9. IMPORTANT: verify Python tempfile really uses JOBSCRATCH
###############################################################################

echo
echo "Checking Python tempfile..."

python3 - <<'PY'
import os
import tempfile
import shutil

tmpdir = tempfile.gettempdir()

print("TMPDIR environment :", os.environ.get("TMPDIR"))
print("tempfile.gettempdir:", tmpdir)

expected = os.path.realpath(os.environ["TMPDIR"])
actual = os.path.realpath(tmpdir)

if actual != expected:
    raise RuntimeError(
        f"Python tempfile is NOT using TMPDIR: "
        f"expected={expected}, actual={actual}"
    )

testdir = tempfile.mkdtemp(prefix="promise_test_")
print("Created test temp :", testdir)

if not os.path.realpath(testdir).startswith(expected + os.sep):
    raise RuntimeError(
        f"Temporary directory escaped JOBSCRATCH: {testdir}"
    )

shutil.rmtree(testdir)

print("Python tempfile check: OK")
PY

###############################################################################
# 10. Check Python dependencies
###############################################################################

echo
echo "Checking Python dependencies..."

python3 - <<'PY'
import cadnaPromise
import colorama
import colorlog
import matplotlib
import numpy
import packaging
import regex
import tqdm
import yaml

try:
    import docopt
except ImportError:
    import docopt_ng as docopt

import tempfile

print("Python deps OK")
print("cadnaPromise:", cadnaPromise.__file__)
print("temp directory:", tempfile.gettempdir())
PY

###############################################################################
# 11. Activate/install PROMISE/CADNA
###############################################################################

echo
echo "Activating CADNA/PROMISE..."

activate-promise

###############################################################################
# 12. Run benchmarks on the allocated compute node
###############################################################################

echo
echo "Starting benchmarks..."
echo "JOBS=$JOBS"

srun bash run_benchmarks.sh \
    true \
    true \
    true \
    sparse_lu \
    dense_lu \
    backprop \
    hotspot \
    particle_filter \
    srad_v2 \
    --parallel \
    --jobs "$JOBS"

echo
echo "Benchmarks finished successfully."
