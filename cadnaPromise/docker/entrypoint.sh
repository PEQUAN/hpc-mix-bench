#!/bin/bash
set -e

<<<<<<< HEAD
if ! command -v g++-13 >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: g++-13 not found in container"
    exit 1
fi

export CC=gcc-13
export CXX=g++-13

echo "[entrypoint] Using CC=$CC ($(command -v $CC))"
echo "[entrypoint] Using CXX=$CXX ($(command -v $CXX))"

if command -v activate-promise >/dev/null 2>&1; then
    activate-promise --CC="$CC" --CXX="$CXX"
=======
# 确保是 bash
if [ -f "$(which activate-promise)" ]; then
    echo "[entrypoint] Activating promise environment"
    activate-promise
>>>>>>> 9134ceec922a3fa36a38f034607a04c945a008fe
else
    echo "[entrypoint] activate-promise not found"
fi

<<<<<<< HEAD
=======
# 执行传进来的命令
>>>>>>> 9134ceec922a3fa36a38f034607a04c945a008fe
exec "$@"
