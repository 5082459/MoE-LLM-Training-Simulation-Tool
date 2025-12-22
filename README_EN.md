# MoE-LLM Pre-training Resource Analysis Tool

A lightweight visualization tool that combines theory-based estimation and simulation to help AI infra engineers size compute, memory, and communication needs for large-model pre-training (especially MoE architectures), and to suggest GPU choices and parallel strategies.

## Key Features
- **Deep model resource estimation (Tab 1)**: Parse tensor graphs to compute total/active params, HBM footprint, per-step FLOPs and memory throughput; suggest GPU choices under HBM/compute constraints.
- **Distributed comms simulation (Tab 2)**: Support DP/PP/TP/EP combinations, estimate AllReduce/AllGather/AllToAll traffic, separate scale-up vs scale-out bandwidth, output step time, total GPU hours, and training days.
- **Timeline projection**: Incorporate pipeline bubble and grad update to show step timeline and compute/memory/comm ratios.
- **Interactive UI**: Streamlit frontend for common model/train configs with real-time metrics and charts.

## Project Layout
- `frontend_app.py`: Streamlit UI and interactions.
- `api.py`: FastAPI backend; two endpoints `/tab1/analyze` and `/tab2/analyze`.
- `Compute_Graph_fork4.py`: Tensor/compute graph construction and FLOPs/memory estimation.
- `Parallel_fork4.py`: Parallel strategy and runtime rank topology.
- `GPU_Profiler_fork4.py`: GPU profiles (H100/H800/H20/A100/Ascend, etc.).
- `Precision_Policy_fork4.py`: Precision and operator policies.
- `timeline.py`: Distributed training timeline simulation.
- `User_Config_fork4.py`: Input data classes and defaults.
- `start.sh` / `start.bat`: One-click scripts to start backend + frontend.
- `.streamlit/config.toml`: UI theme.
- `requirements.txt`: Python dependencies.

## Requirements
- Python 3.9+
- pip with virtualenv or conda

## Quick Start
1) Clone or download, then enter `ai_infra_simu_tool_portable`.
2) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:   venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```
3) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4) Start backend (port 8000):
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
   ```
5) Start frontend (port 8501):
   ```bash
   streamlit run frontend_app.py
   ```
6) Open `http://localhost:8501` in your browser; the UI calls the local backend.

> macOS/Linux: `./start.sh`; Windows: `start.bat` to auto install deps and start backend + frontend sequentially.

## Developer Notes
- Backend APIs: `/tab1/analyze` and `/tab2/analyze`; request/response models are defined in `api.py` via Pydantic.
- GPU profiles live in `GPU_Profiler_fork4.py`; add or adjust as needed.
- Parallel and model defaults are in `User_Config_fork4.py`.
- No automated tests yet; after changes, run backend (uvicorn) and frontend (streamlit) locally for a smoke check.

## License
MIT License. See `LICENSE`.
