# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . .
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip wheel --wheel-dir /wheels \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        "torch==2.8.0+cpu" "torchvision==0.23.0+cpu" \
    && python -m pip wheel --wheel-dir /wheels \
        "numpy>=1.26,<3.0" \
        "pillow>=10.0,<13.0" \
        "pandas>=2.3.1,<3.0.0" \
        "scikit-learn>=1.7.1,<2.0.0" \
        "matplotlib>=3.10.5,<4.0.0" \
        "seaborn>=0.13.2,<0.14.0" \
        "tqdm>=4.67.1,<5.0.0" \
        "albumentations>=2.0.8,<3.0.0" \
    && python -m pip wheel --no-deps --wheel-dir /wheels .


FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
        /wheels/famous_deep_learning_cnns-*.whl \
    && rm -rf /wheels

WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/outputs && chown -R app:app /workspace
USER app

ENTRYPOINT ["famous-cnns"]
CMD ["list"]
