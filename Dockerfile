FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /work
RUN pip install --no-cache-dir \
    numpy scipy cma scikit-learn \
    tenseal openfhe Pyfhel \
    fhe-oracle
COPY benchmarks/ ./benchmarks/
CMD ["python", "benchmarks/library_comparison.py"]
