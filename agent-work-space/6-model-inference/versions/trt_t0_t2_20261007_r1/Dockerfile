FROM pi05_quant:20261006-r1
# Base image must match sha256:972eacb4cb7ded436467d8cf95e3ccfd2bd65583ae866151ae96235ca540a4c9.
# --no-config is essential: /app/pyproject.toml otherwise overrides ml-dtypes.
RUN uv --no-config pip install --python /.venv/bin/python --index-url https://pypi.org/simple --no-deps \
    tensorrt-cu12==10.13.3.9 tensorrt-cu12-bindings==10.13.3.9 tensorrt-cu12-libs==10.13.3.9 \
    onnx==1.19.0 onnxruntime-gpu==1.22.0 onnx-graphsurgeon==0.5.8 \
    nvidia-modelopt==0.33.1 nvidia-modelopt-core==0.33.1 cupy-cuda12x==13.6.0 \
    ml-dtypes==0.5.3 coloredlogs==15.0.1 humanfriendly==10.0 flatbuffers==25.9.23 \
    onnxconverter-common==1.16.0 ninja==1.13.0 torchprofile==0.0.4 pulp==3.2.2 fastrlock==0.8.3
ENV PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
