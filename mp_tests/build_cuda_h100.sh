#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARKS="${BENCHMARKS:-backprop dense_lu hotspot}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"

for bench in $BENCHMARKS; do
  if [ -f "$ROOT_DIR/$bench/Makefile.cuda" ]; then
    echo "==> building ${bench} double-precision baseline"
    make -C "$ROOT_DIR/$bench" -f Makefile.cuda CUDA_ARCH="$CUDA_ARCH"
  fi

  for dir in "$ROOT_DIR/$bench"/digit*_*; do
    [ -d "$dir" ] || continue
    echo "==> building ${dir#$ROOT_DIR/}"
    make -C "$dir" -f Makefile.cuda CUDA_ARCH="$CUDA_ARCH"
  done
done
