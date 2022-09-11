FROM tensorflow/tensorflow:2.10.0-gpu as base

#######
### Rust compiler:
#####
RUN apt-get install -y \
    rustc \
    cargo

#######
### Setup environment
#####

RUN mkdir -p /AlphaCC
WORKDIR /AlphaCC
RUN apt install python3.8-venv
COPY requirements.txt .
COPY make.sh .
COPY Makefile .
COPY . .
RUN make install

ENTRYPOINT ["bash"]
CMD ["-i"]
