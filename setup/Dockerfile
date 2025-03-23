ARG USE_GPU=false

FROM --platform=linux/amd64 pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

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
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
# COPY requirements_${USE_GPU:+gpu}${USE_GPU:-cpu}.txt requirements.txt
COPY requirements_cpu.txt .
COPY requirements_gpu.txt .

# Install Python dependencies
# RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements_cpu.txt

# Check CUDA
RUN CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False") && \
    if [ "$CUDA_AVAILABLE" = "True" ]; then \
        pip install --no-cache-dir --no-deps -r requirements_gpu.txt --no-deps; \
    else \
        pip uninstall -y torch && \
        pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps; \
    fi

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create a non-root user
# RUN useradd -m -s /bin/bash developer
# RUN chown -R developer:developer /app

# Switch to non-root user
# USER developer

# Set up bash as default shell with useful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias python="python3"' >> ~/.bashrc

# Start interactive bash shell by default
CMD ["/bin/bash"]
