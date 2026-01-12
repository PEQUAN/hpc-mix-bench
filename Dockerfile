FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

# ---------- build arg（install matplotlib or not，default yes） ----------
ARG WITH_PLOT=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    rsync \
    python3 \
    python3-pip \
    python3-venv \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VENV_PATH}

RUN ${VENV_PATH}/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

WORKDIR /app/cadnaPromise
RUN ${VENV_PATH}/bin/python -m pip install .

RUN if [ "$WITH_PLOT" = "1" ]; then \
        ${VENV_PATH}/bin/python -m pip install matplotlib; \
    else \
        echo "Skipping matplotlib installation"; \
    fi

RUN ${VENV_PATH}/bin/python - << 'EOF'
from cadnaPromise.run import runPromise
print("runPromise OK")
try:
    import matplotlib
    print("matplotlib OK")
except ImportError:
    print("matplotlib NOT installed")
EOF

ENV PATH="${VENV_PATH}/bin:${PATH}"
WORKDIR /app

CMD ["/bin/bash"]
