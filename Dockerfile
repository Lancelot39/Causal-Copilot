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
    && rm -rf /var/lib/apt/lists/*

RUN apt-get udpate && apt-get install latexmk

# Copy requirements file
# COPY requirements.txt .
COPY requirements_fastapi.txt .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_fastapi.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app
ENV PORT=8000

# Create a non-root user
RUN useradd -m -s /bin/bash developer
RUN chown -R developer:developer /app

# Switch to non-root user
USER developer

# Set up bash as default shell with useful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias python="python3"' >> ~/.bashrc

# Port
EXPOSE 8000

# Start interactive bash shell by default
# CMD ["/bin/bash"]
CMD ["uvicorn", "run_workflow_v1:app", "--host", "0.0.0.0", "--port", "8000"] 

