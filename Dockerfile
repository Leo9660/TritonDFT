FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ---------- OS-level deps ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libopenmpi-dev \
    openmpi-bin \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip

# ---------- Python venv ----------
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip
ENV PATH="/opt/venv/bin:${PATH}"

# ---------- Python deps (cached layer) ----------
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /workspace/requirements.txt

# ---------- Build Quantum ESPRESSO (CPU) ----------
ARG QE_VERSION=qe-7.3.1
ARG QE_BUILD_JOBS=8
RUN git clone --depth 1 --branch ${QE_VERSION} https://gitlab.com/QEF/q-e.git /workspace/QuantumE \
 && cd /workspace/QuantumE \
 && ./configure \
 && make -j${QE_BUILD_JOBS} pw pp \
 && find /workspace/QuantumE -type d \( -name ".git" -o -name "test-suite" -o -name "Doc" \) -prune -exec rm -rf {} + || true

# ---------- Late-added Python deps (avoid invalidating QE build cache) ----------
RUN pip install --no-cache-dir slowapi

# ---------- App source ----------
COPY . /workspace
RUN rm -rf /workspace/QuantumE.bak 2>/dev/null || true

# ---------- Runtime ----------
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
