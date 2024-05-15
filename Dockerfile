FROM rust:latest
COPY . /hercules
WORKDIR /hercules
RUN apt-get update && \
    apt-get -y install python3 python3-pip curl python3-venv && \
    python3 -m venv .venv
RUN . .venv/bin/activate && \
    pip3 install maturin==1.5.1 && \
    maturin develop
ENV PATH=/hercules/.venv/bin:$PATH
