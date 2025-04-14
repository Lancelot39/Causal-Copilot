# comment out the following line to use cpu
# FROM --platform=linux/amd64 pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime AS cpu-base

# comment the following line when gpu is not available
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

USER root

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y tzdata
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    perl \
    graphviz \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install TinyTeX
RUN rm -rf ~/.TinyTeX && \
    wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh && \
    echo 'export PATH="$PATH:$HOME/.TinyTeX/bin/x86_64-linux"' >> ~/.bashrc && \
    . ~/.bashrc

# Install TeX packages
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip

# Copy requirements files
COPY requirements_gpu.txt requirements_cpu.txt ./

# Install CPU requirements
# RUN pip install --no-cache-dir -r requirements_cpu.txt --no-deps; 

# comment out the following line when gpu is not available
RUN pip install --no-cache-dir -r requirements_gpu.txt --no-deps;

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Setup GPU acceleration, comment the following line when gpu is not available
RUN apt-get update && apt-get install -y libboost-python-dev libboost-numpy-dev && \
    pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*" "dask-cudf-cu12==25.4.*" "cuml-cu12==25.4.*" \
    "cugraph-cu12==25.4.*" "nx-cugraph-cu12==25.4.*" "cuspatial-cu12==25.4.*" \
    "cuproj-cu12==25.4.*" "cuxfilter-cu12==25.4.*" "cucim-cu12==25.4.*" \
    "pylibraft-cu12==25.4.*" "raft-dask-cu12==25.4.*" "cuvs-cu12==25.4.*" \
    "nx-cugraph-cu12==25.4.*" 

# comment the following line when gpu is not available
RUN cd externals/pc_adjacency_search && make

# Set up bash as default shell with useful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias python="python3"' >> ~/.bashrc

# Start interactive bash shell by default
CMD ["/bin/bash"]
