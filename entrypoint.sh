#!/bin/bash
set -e

if command -v activate-promise &> /dev/null; then
    echo ">> Activating cadnaPromise environment..."
    activate-promise || true
else
    echo "Warning: activate-promise not found in PATH"
fi

echo ">> Checking Compiler Version:"
g++ --version | head -n 1

exec "$@"