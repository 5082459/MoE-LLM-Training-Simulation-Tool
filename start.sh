#!/bin/bash

# 设置脚本在遇到错误时停止
set -e

echo "🚀 正在初始化 AI Infra Simulation Tool..."

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未检测到 Python3，请先安装 Python 3.9+。"
    exit 1
fi

# 1. 创建虚拟环境 (如果不存在)
if [ ! -d "venv" ]; then
    echo "📦 正在创建虚拟环境 (venv)..."
    python3 -m venv venv
else
    echo "✅ 检测到现有虚拟环境"
fi

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 安装依赖
if [ -f "requirements.txt" ]; then
    echo "⬇️  正在安装/更新依赖..."
    pip install -r requirements.txt -q
    echo "✅ 依赖安装完成"
else
    echo "⚠️ 警告: 未找到 requirements.txt"
fi

# 4. 启动服务
echo "🔥 正在启动服务..."

# 启动后端 (后台运行)
echo "   - 启动后端 API (Port 8000)..."
nohup uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 > backend.log 2>&1 &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端
echo "   - 启动前端 UI..."
echo "✅ 服务已就绪! 浏览器应会自动打开。"
echo "📝 后端日志保存在 backend.log"
echo "🛑 按 Ctrl+C 停止所有服务"

streamlit run frontend_app.py

# 退出时清理后台进程
kill $BACKEND_PID
