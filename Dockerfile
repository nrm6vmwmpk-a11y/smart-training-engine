# Dockerfile for Smart Training Harness
FROM python:3.10-slim

# 设置工作目录
WORKDIR /workspace

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制任务数据和代码
COPY tasks/ ./tasks/
COPY harness.py .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令（运行任务A，10轮优化）
CMD ["python", "harness.py", "a", "10"]
