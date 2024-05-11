FROM python:3.11-slim as base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    WORKDIR=/code

RUN mkdir ${WORKDIR}
WORKDIR ${WORKDIR}
RUN pip install uv && \
    uv venv .venv --prompt alpha-cc
ENV PATH="$WORKDIR/.venv/bin:$PATH"


FROM base as rust-base
# Install build deps
RUN apt-get update &&                                                           \
    apt-get install -y curl clang libssl-dev make pkg-config gcc python3-dev && \
    apt-get clean &&                                                            \
    rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    pip install uv &&                                 \
    uv venv .venv --prompt alpha-cc
ENV PATH="/root/.cargo/bin:$WORKDIR/.venv/bin:$PATH"
RUN mkdir -p $WORKDIR/alpha_cc/engine/backend
COPY ./alpha_cc/engine/backend $WORKDIR/alpha_cc/engine/backend
RUN bash -c ' \
    uv pip install pip maturin && \
    cd $WORKDIR/alpha_cc/engine/backend && \
    maturin develop --release \
'

FROM rust-base as training
RUN mkdir -p $WORKDIR/alpha_cc/
COPY ./alpha_cc $WORKDIR/alpha_cc
COPY ./setup.* $WORKDIR
RUN uv pip install . 
CMD ["bash"]


FROM base as api-backend
COPY . .
RUN uv pip install ".[api]"
CMD ["bash"]
