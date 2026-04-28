FROM python:3.12-slim

# System deps for numpy/scipy/cma + openfhe native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install fhe-oracle + libraries. openfhe wheel is Linux-only,
# resolves correctly here.
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    cma \
    scikit-learn \
    tenseal \
    openfhe \
    fhe-oracle

# Copy the benchmark harness (will be mounted at runtime in practice).
# The harness imports fhe_oracle's AutoOracle + tenseal_circuits.
COPY benchmarks/ ./benchmarks/

CMD ["python", "benchmarks/library_comparison.py"]
