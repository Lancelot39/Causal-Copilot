FROM python:3.10.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install latexmk

# 复制项目文件
# COPY requirements.txt .
COPY requirements_fastapi.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PORT=8000

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "run_workflow_v1:app", "--host", "0.0.0.0", "--port", "8000"] 