#!/bin/bash
set -e

# 确保是 bash
if [ -f "$(which activate-promise)" ]; then
    echo "[entrypoint] Activating promise environment"
    activate-promise
else
    echo "[entrypoint] activate-promise not found"
fi

# 执行传进来的命令
exec "$@"
