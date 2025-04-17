.PHONY: help setup-local setup-venv install-deps run-local \
        build-cpu build-gpu run-cpu run-gpu clean-docker clean-all

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Local Development:"
	@echo "  make setup-local    - Set up local development environment"
	@echo "  make setup-venv     - Create and activate virtual environment"
	@echo "  make install-deps   - Install Python dependencies"
	@echo "  make run-local      - Run the application locally"
	@echo ""
	@echo "Docker Development:"
	@echo "  make build-cpu      - Build CPU version of the Docker image"
	@echo "  make build-gpu      - Build GPU version of the Docker image"
	@echo "  make run-cpu        - Run CPU version of the container"
	@echo "  make run-gpu        - Run GPU version of the container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-docker   - Clean up Docker resources"
	@echo "  make clean-all      - Clean up all resources (local and Docker)"
	@echo ""
	@echo "  make help           - Show this help message"

# Local Development Commands
setup-local: setup-venv install-deps

setup-venv:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to activate it."

install-deps:
	pip install --upgrade pip
	pip install -r setup/requirements_cpu.txt
	pip install python-multipart itsdangerous python-jose passlib

run-local:
	python3 -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload

# Docker Commands
build-cpu:
	docker build --build-arg USE_GPU=false -t causal-copilot:cpu .

build-gpu:
	docker build --build-arg USE_GPU=true -t causal-copilot:gpu .

run-cpu:
	docker run -it --rm \
		-p 7860:7860 \
		-v $(PWD):/app \
		--name causal-copilot-cpu \
		causal-copilot:cpu

run-gpu:
	docker run -it --rm \
		-p 7860:7860 \
		-v $(PWD):/app \
		--gpus all \
		--name causal-copilot-gpu \
		causal-copilot:gpu

# Cleanup Commands
clean-docker:
	docker stop $$(docker ps -aq) 2>/dev/null || true
	docker rm $$(docker ps -aq) 2>/dev/null || true
	docker rmi $$(docker images -q causal-copilot*) 2>/dev/null || true

clean-all: clean-docker
	rm -rf venv
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage 