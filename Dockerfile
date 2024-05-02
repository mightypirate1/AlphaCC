FROM python:3.11.9-bullseye

# rust tool-chain
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

ENV WORKDIR=/code
RUN mkdir $WORKDIR
WORKDIR $WORKDIR
COPY setup.py $WORKDIR
COPY pyproject.toml $WORKDIR
COPY Makefile $WORKDIR

COPY alpha_cc $WORKDIR/alpha_cc
RUN bash -c '                              \
    pip install uv &&                      \
    uv pip install --system -e ".[all]" && \
    cd alpha_cc/engine/backend &&          \
    maturin develop --release              \
'

CMD ["bash"]
# CMD ["python" "-c" "'from alpha_cc.engine import Board'"]