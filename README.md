# MoE-LLM 预训练资源需求分析工具

一款基于理论建模与仿真的轻量级可视化工具，帮助 AI 基础设施工程师评估大模型（尤其是 MoE 架构）在预训练阶段的计算、显存与通信需求，并给出 GPU 选型与并行策略参考。

## 核心功能
- **深度模型资源估算（Tab 1）**：解析模型张量图，计算总参数/激活参数、HBM 容量占用、单位 step FLOPs 与显存吞吐，并给出按 HBM/算力约束的 GPU 选型建议。
- **分布式通信仿真（Tab 2）**：支持 DP/PP/TP/EP 组合，估算 AllReduce/AllGather/AllToAll 等通信量，区分 scale-up/scale-out 带宽，输出单 step 耗时、总 GPU Hours 与训练天数。
- **时间线推演**：结合 pipeline bubble 与梯度更新，给出 step timeline 及计算/显存/通信占比。
- **可视化前端**：Streamlit UI，支持常用模型与训练配置输入，实时查看指标与图表。

## 目录结构
- `frontend_app.py`：Streamlit 前端界面与交互。
- `api.py`：FastAPI 后端，两类分析接口 `/tab1/analyze` 与 `/tab2/analyze`。
- `Compute_Graph_fork4.py`：模型张量/计算图构建与 FLOPs/显存估算。
- `Parallel_fork4.py`：并行策略与 runtime rank 拓扑生成。
- `GPU_Profiler_fork4.py`：GPU 硬件画像（H100/H800/H20/A100/Ascend 等）。
- `Precision_Policy_fork4.py`：精度与算子精度策略。
- `timeline.py`：分布式训练时间线仿真。
- `User_Config_fork4.py`：输入数据类与默认配置。
- `start.sh` / `start.bat`：一键启动脚本（后端 + 前端）。
- `.streamlit/config.toml`：前端主题配置。
- `requirements.txt`：Python 依赖。

## 环境要求
- Python 3.9+
- pip / virtualenv 或 conda

## 快速开始
1) 克隆或下载本仓库，进入 `ai_infra_simu_tool_portable`。
2) 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   # Windows:   venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```
3) 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4) 启动后端（默认 8000 端口）：
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
   ```
5) 启动前端（默认 8501 端口）：
   ```bash
   streamlit run frontend_app.py
   ```
6) 打开浏览器访问 `http://localhost:8501`，前端会调用本地后端 API。

> macOS/Linux 可执行 `./start.sh`，Windows 可执行 `start.bat`，自动安装依赖并先后启动后端与前端。

## 开发者提示
- 后端接口：`/tab1/analyze` 与 `/tab2/analyze`，请求/响应模型见 `api.py` 中的 Pydantic 数据类。
- 默认 GPU 画像定义在 `GPU_Profiler_fork4.py`，可按需新增或调整。
- 默认并行策略/模型配置可在 `User_Config_fork4.py` 里调整。
- 目前无自动化测试，修改后建议本地跑一遍后端（uvicorn）和前端（streamlit）进行冒烟验证。

## 许可证
本项目采用 MIT License，详见 `LICENSE` 文件。
