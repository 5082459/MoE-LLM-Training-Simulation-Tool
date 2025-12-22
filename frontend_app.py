import streamlit as st
import httpx
import json
from typing import Dict, Any, List

# API Base URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="LLM 训练资源需求分析工具",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for iOS-like style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');

    /* Global Background - Aesthetic Mesh Gradient with Breathing Animation */
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #ffffff;
        background-image: 
            radial-gradient(at 40% 20%, hsla(28,100%,74%,0.25) 0px, transparent 50%),
            radial-gradient(at 80% 0%, hsla(189,100%,56%,0.25) 0px, transparent 50%),
            radial-gradient(at 0% 50%, hsla(340,100%,76%,0.25) 0px, transparent 50%),
            radial-gradient(at 80% 50%, hsla(240,100%,70%,0.25) 0px, transparent 50%),
            radial-gradient(at 0% 100%, hsla(22,100%,77%,0.25) 0px, transparent 50%),
            radial-gradient(at 80% 100%, hsla(242,100%,70%,0.25) 0px, transparent 50%),
            radial-gradient(at 0% 0%, hsla(343,100%,76%,0.25) 0px, transparent 50%);
        background-size: 180% 180%;
        animation: gradient-animation 10s ease infinite;
        background-attachment: fixed;
    }
    
    /* Card/Container Style - Premium Frosted Glass */
    .stCard, div[data-testid="stMetric"], div[data-testid="stExpander"], div[data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.6);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        background-color: rgba(255, 255, 255, 0.8);
        transition: all 0.2s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #007AFF;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15);
    }
    
    /* Buttons - Gradient & Glow */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        background: linear-gradient(135deg, #007AFF 0%, #00C6FF 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 122, 255, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        padding: 0.6rem 1.2rem;
    }
    .stButton button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 122, 255, 0.5);
    }
    .stButton button:active {
        transform: translateY(1px);
    }
    
    /* Scrollbar Styling (Webkit) */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.2);
    }

    /* Headings */
    h1, h2, h3 {
        color: #1C1C1E;
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: -0.5px;
    }
    
    /* Custom Metric Card Style */
    .metric-card {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.8);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8E8E93;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1C1C1E 0%, #3A3A3C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .metric-value.highlight {
        background: linear-gradient(135deg, #007AFF 0%, #00C6FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* GPU Card Style */
    .gpu-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .gpu-card:hover {
        background: rgba(255, 255, 255, 0.9);
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .gpu-info {
        display: flex;
        flex-direction: column;
    }
    .gpu-name {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1C1C1E;
    }
    .gpu-meta {
        font-size: 0.9rem;
        color: #8E8E93;
    }
    .gpu-badge {
        background: rgba(0, 122, 255, 0.1);
        color: #007AFF;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Animation: Fade In Up */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    .animate-enter {
        animation: fadeInUp 0.6s ease-out forwards;
    }

    /* Hide Streamlit Branding & Deploy Button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stBaseButton-header"] {display: none;}
    div[data-testid="stMetric"] { display: none; } /* Hide default metrics if using custom */

    /* Glassmorphism Tabs */
    button[data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px 16px 0 0 !important;
        margin-right: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    /* Selected Tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.85) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border-bottom: none !important;
    }
    
    /* Tab Hover */
    button[data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-2px);
    }
    
    /* Hide default tab bar bottom line */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-baseweb="tab-border"] {
        display: none;
    }

    /* Hero Header Style */
    .hero-container {
        text-align: center;
        padding: 40px 0 20px 0;
        position: relative;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1C1C1E 30%, #007AFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -1px;
        position: relative;
        z-index: 2;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #8E8E93;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .hero-glow {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 300px;
        height: 100px;
        background: radial-gradient(circle, rgba(0,122,255,0.2) 0%, rgba(255,255,255,0) 70%);
        filter: blur(40px);
        z-index: 1;
    }
    
    /* Enhanced Expander Header */
    .streamlit-expanderHeader {
        font-family: 'Poppins', sans-serif;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #1C1C1E !important;
        background-color: transparent !important;
    }
    
    /* Input Label Decoration */
    div[data-testid="stWidgetLabel"] {
        border-left: 3px solid #007AFF;
        padding-left: 8px;
        margin-bottom: 4px;
    }
    
    /* Footer Style */
    .footer {
        text-align: center;
        padding: 40px 0;
        color: #C7C7CC;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-glow"></div>
    <h1 class="hero-title">🚀 MoE-LLM 预训练资源需求分析工具</h1>
    <div class="hero-subtitle">基于Transformer架构分析与仿真推演，输出特定模型架构下预训练的计算、显存与通信需求(默认配置为Deepseek-V3.1)</div>
</div>
""", unsafe_allow_html=True)

# Create Tabs
tab1, tab2 = st.tabs(["📊 模型资源需求理论分析 (功能1)", "🔄 分布式训练资源推演 (功能2)"])

# Helper function to call API
def call_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Increased timeout to 600s (10 minutes) for heavy computations
        response = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=600.0)
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error(f"无法连接到后端服务 ({API_URL})。请确保 `python api.py` 或 `uvicorn api:app` 正在运行。")
        return None
    except Exception as e:
        st.error(f"API 请求失败: {str(e)}")
        if 'response' in locals():
            st.code(response.text)
        return None

# --- Tab 1 Implementation ---
with tab1:
    st.header("模型资源需求理论分析 (算力、HBM)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 1. Model Config
        with st.expander("🛠️ 模型结构配置 (ModelConfig)", expanded=True):
            hidden_size = st.number_input("Hidden Size", value=7168, step=128)
            num_attention_heads = st.number_input("Attention Heads", value=128, step=1)
            num_hidden_layers = st.number_input("Hidden Layers", value=61, step=1)
            num_dense_hidden_layers = st.number_input("Dense Layers", value=3, step=1)
            num_moe_hidden_layers = st.number_input("MoE Layers", value=58, step=1)
            dense_intermediate_size = st.number_input("Dense Intermediate Size", value=18432, step=128)
            moe_intermediate_size = st.number_input("MoE Intermediate Size", value=2048, step=64)
            vocab_size = st.number_input("Vocab Size", value=129280, step=1000)
            n_routed_experts = st.number_input("Routed Experts", value=256, step=1)
            n_shared_experts = st.number_input("Shared Experts", value=1, step=1)
            num_experts_per_tok = st.number_input("Experts per Token", value=8, step=1)
            
            # Constants (Hidden from UI)
            capacity_factor = 1.1
            alpha_silu = 6
            beta_bias = 1
            gamma_rms = 4
            delta_softmax = 5
            epsilon_topk = 3
            gamma_adam = 2

        # 2. Other Configs (Outside Scrollable Area)
        with st.expander("⚙️ 训练配置 (TrainConfig)", expanded=True):
            global_batch_size = st.number_input("Global Batch Size", value=4096, step=8)
            num_micro_batches = st.number_input("Micro Batches (Gradient Accumulation)", value=16, step=1)
            seq_len = st.number_input("Sequence Length", value=4096, step=128)
            total_tokens = st.number_input("Total Tokens (Billion)", value=14800.0, step=10.0) * 1e9
            
            # Default values for removed UI controls
            use_activation_checkpoint = True
            use_zero = True
            ckpt_ratio = 0.5
            zero_stage = 2

        with st.expander("🎯 精度配置 (PrecisionConfig)", expanded=True):
            policy_name = st.selectbox("Policy Name", ["BF16_F8_E4M3_F32"], index=0)
            
        # Default value for removed UI control
        system_overhead_ratio = 0.10

        analyze_btn_tab1 = st.button("开始分析", type="primary")

    with col2:
        if analyze_btn_tab1:
            # Construct Payload
            model_config = {
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "num_dense_hidden_layers": num_dense_hidden_layers,
                "num_moe_hidden_layers": num_moe_hidden_layers,
                "dense_intermediate_size": dense_intermediate_size,
                "moe_intermediate_size": moe_intermediate_size,
                "vocab_size": vocab_size,
                "n_routed_experts": n_routed_experts,
                "n_shared_experts": n_shared_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "capacity_factor": capacity_factor,
                "alpha_silu": alpha_silu,
                "beta_bias": beta_bias,
                "gamma_rms": gamma_rms,
                "delta_softmax": delta_softmax,
                "epsilon_topk": epsilon_topk,
                "gamma_adam": gamma_adam
            }
            
            train_config = {
                "global_batch_size": global_batch_size,
                "num_micro_batches": num_micro_batches,
                "seq_len": seq_len,
                "total_tokens": total_tokens,
                "use_activation_checkpoint": use_activation_checkpoint,
                "ckpt_ratio": ckpt_ratio,
                "use_zero_": use_zero, # Note: backend uses use_zero_ in User_Config but might expect use_zero in some places? checking User_Config_fork4.py again... it says use_zero_: bool = True
                "zero_stage": zero_stage
            }
            
            payload = {
                "sim": {
                    "model": model_config,
                    "train": train_config,
                    "precision": {"policy_name": policy_name}
                },
                "system_overhead_ratio": system_overhead_ratio
            }
            
            with st.spinner("正在分析中..."):
                result = call_api("/tab1/analyze", payload)
                if result:
                    st.session_state['tab1_result'] = result

        # Check for cached result
        if 'tab1_result' in st.session_state:
            result = st.session_state['tab1_result']
            
            # Wrap everything in an animation container
            st.markdown('<div class="animate-enter">', unsafe_allow_html=True)
            
            # 1. Parameter Card
            st.subheader("📦 参数与显存需求")
            m1, m2, m3, m4 = st.columns(4)
            
            def metric_html(label, value, icon=""):
                return f"""
                <div class="metric-card">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            with m1: st.markdown(metric_html("模型总参数量", f"{result['total_params_b']:.2f} B", "🧩"), unsafe_allow_html=True)
            with m2: st.markdown(metric_html("模型激活参数量", f"{result['active_params_b']:.2f} B", "⚡"), unsafe_allow_html=True)
            with m3: st.markdown(metric_html("HBM 容量需求", f"{result['hbm_total_need_tb']:.2f} TB", "💾"), unsafe_allow_html=True)
            with m4: st.markdown(metric_html("HBM 系统开销", f"{result['hbm_system_overhead_gb']:.2f} GB", "⚙️"), unsafe_allow_html=True)
            
            # 2. Compute Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("⚡ 计算与吞吐")
            c1, c2, c3 = st.columns(3)
            
            with c1: st.markdown(metric_html("单个Step 计算量", f"{result['step_flops_eflops']:.2f} EFLOPs", "🔢"), unsafe_allow_html=True)
            with c2: st.markdown(metric_html("HBM 吞吐/Step", f"{result['step_hbm_throughput_tb']:.2f} TB", "🚀"), unsafe_allow_html=True)
            with c3: st.markdown(metric_html("总训练 Steps", f"{result['total_steps']:,}", "⏱️"), unsafe_allow_html=True)

            # 3. GPU Comparison List (Card View)
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🖥️ GPU 资源部署方式寻优")
            
            gpu_rows = result.get("gpu_rows", [])
            if gpu_rows:
                for row in gpu_rows:
                    schemes = row.get('feasible_schemes', [])
                    best_scheme = schemes[0] if schemes else None
                    
                    if best_scheme:
                        min_gpus = best_scheme['total_gpus']
                        best_note = best_scheme['note']
                        per_gpu_mem = f"{best_scheme['per_gpu_hbm_gb']:.2f}"
                        util = f"{best_scheme['hbm_util']*100:.1f}%"
                    else:
                        min_gpus = "N/A"
                        best_note = "无法满足显存需求"
                        per_gpu_mem = "-"
                        util = "0%"

                    # Generate GPU Card HTML
                    st.markdown(f"""
                    <div class="gpu-card">
                        <div class="gpu-info">
                            <div class="gpu-name">{row['gpu_sku']} <span style="font-size:0.8em; color:#8E8E93; font-weight:400;">({row['hbm_gb']}GB)</span></div>
                            <div class="gpu-meta">
                                方案: {best_note} <br>
                                单卡占用: {per_gpu_mem} GB ({util})
                            </div>
                        </div>
                        <div class="gpu-badge">
                            {min_gpus} Cards
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("无 GPU 数据返回")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Placeholder for output area (Tab 1)
            st.markdown("""
            <div style="
                border: 2px dashed rgba(0, 122, 255, 0.2); 
                border-radius: 12px; 
                padding: 20px; 
                height: 100%; 
                min-height: 600px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: rgba(255, 255, 255, 0.4);
            " id="output-placeholder">
                <div style="color: #007AFF; font-weight: 600; font-size: 1.2em; margin-bottom: 10px;">📊 分析结果输出区</div>
                <div style="color: #8E8E93;">请在左侧配置参数并点击“开始分析”</div>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2 Implementation ---
with tab2:
    st.header("集合通信推演")
    
    col_t2_1, col_t2_2 = st.columns([1, 2])
    
    with col_t2_1:
        with st.expander("🖥️ GPU 资源配置", expanded=True):
            t2_gpu_sku = st.selectbox("GPU SKU", ["H100 80G", "H800 80G", "H20 96G"], index=1)
            t2_num_gpus = st.number_input("Total GPUs", value=2048, step=8)

        with st.expander("🧩 并行策略配置 (ParallelConfig)", expanded=True):
            dp_size = st.number_input("DP Size", value=8, step=1)
            pp_size = st.number_input("PP Size", value=4, step=1)
            ep_size = st.number_input("EP Size", value=64, step=1)
            tp_size = st.number_input("TP Size", value=1, step=1)
            
            total_parallel = dp_size * pp_size * ep_size * tp_size
            if total_parallel != t2_num_gpus:
                st.warning(f"⚠️ 并行度总乘积 ({total_parallel}) 与 GPU 总数 ({t2_num_gpus}) 不一致！")

        # Reuse configs from Tab 1 (Optional: could duplicate inputs if needed)
        st.info("💡 模型与训练配置将沿用功能1的设置")
        
        analyze_btn_tab2 = st.button("开始推演", type="primary")

    with col_t2_2:
        if analyze_btn_tab2:
            # Reconstruct Payload using Tab 1 inputs + Tab 2 inputs
            # Note: In a real app, we might want state management, but here direct access works
            model_config_t2 = {
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "num_dense_hidden_layers": num_dense_hidden_layers,
                "num_moe_hidden_layers": num_moe_hidden_layers,
                "dense_intermediate_size": dense_intermediate_size,
                "moe_intermediate_size": moe_intermediate_size,
                "vocab_size": vocab_size,
                "n_routed_experts": n_routed_experts,
                "n_shared_experts": n_shared_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "capacity_factor": capacity_factor,
                "alpha_silu": alpha_silu,
                "beta_bias": beta_bias,
                "gamma_rms": gamma_rms,
                "delta_softmax": delta_softmax,
                "epsilon_topk": epsilon_topk,
                "gamma_adam": gamma_adam
            }
            
            train_config_t2 = {
                "global_batch_size": global_batch_size,
                "num_micro_batches": num_micro_batches,
                "seq_len": seq_len,
                "total_tokens": total_tokens,
                "use_activation_checkpoint": use_activation_checkpoint,
                "ckpt_ratio": ckpt_ratio,
                "use_zero_": use_zero,
                "zero_stage": zero_stage
            }

            payload_t2 = {
                "sim": {
                    "model": model_config_t2,
                    "train": train_config_t2,
                    "precision": {"policy_name": policy_name}
                },
                "gpu": {
                    "num_gpus": t2_num_gpus,
                    "gpu_sku": t2_gpu_sku
                },
                "parallel": {
                    "dp_size": dp_size,
                    "pp_size": pp_size,
                    "ep_size": ep_size,
                    "tp_size": tp_size
                },
                "system_overhead_ratio": system_overhead_ratio
            }

            with st.spinner("正在推演中..."):
                result_t2 = call_api("/tab2/analyze", payload_t2)
                if result_t2:
                    st.session_state['tab2_result'] = result_t2
            
        if 'tab2_result' in st.session_state:
            result_t2 = st.session_state['tab2_result']
            
            # Wrap in animation container
            st.markdown('<div class="animate-enter">', unsafe_allow_html=True)
            
            # Helper for metrics (re-defined here or reused if moved to global scope, duplicating for safety)
            def metric_html(label, value, icon=""):
                return f"""
                <div class="metric-card">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """

            # 1. Communication Card
            st.subheader("📡 通信需求分析")
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.markdown(metric_html("单 GPU 通信量", f"{result_t2['comm_bytes_per_rank_tb']:.2f} TB", "📶"), unsafe_allow_html=True)
            with k2: st.markdown(metric_html("Scale-up 通信耗时", f"{result_t2['comm_time_scaleup_s']:.2f} s", "⬆️"), unsafe_allow_html=True)
            with k3: st.markdown(metric_html("Scale-out 通信耗时", f"{result_t2['comm_time_scaleout_s']:.2f} s", "🌐"), unsafe_allow_html=True)
            
            # Breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 通信量构成 (TB)")
            
            c_data = result_t2['comm_by_collective_tb']
            # Convert dict to DataFrame for Altair
            import pandas as pd
            import altair as alt
            
            df_comm = pd.DataFrame(list(c_data.items()), columns=['Collective', 'Traffic (TB)'])
            
            # Custom Altair Chart for aesthetic unity
            chart = alt.Chart(df_comm).mark_bar(cornerRadiusEnd=4, height=20).encode(
                x=alt.X('Traffic (TB)', title=None),
                y=alt.Y('Collective', sort='-x', title=None, axis=alt.Axis(labelFont='Inter', labelColor='#8E8E93')),
                color=alt.Color('Collective', legend=None, scale=alt.Scale(scheme='blues')),
                tooltip=['Collective', 'Traffic (TB)']
            ).properties(
                height=300,
                background='transparent'
            ).configure_axis(
                gridColor='rgba(0,0,0,0.05)',
                domain=False
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(chart, use_container_width=True)

            # 2. Timeline Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("⏱️ 训练耗时推演 (Timeline)")
            t1, t2, t3, t4 = st.columns(4) 
            with t1: st.markdown(metric_html("单Step 耗时", f"{result_t2['step_time_s']:.4f} s", "⏳"), unsafe_allow_html=True)
            with t2: st.markdown(metric_html("单Step GPU Hours", f"{result_t2['step_gpu_hours']:.4f}", "🔋"), unsafe_allow_html=True)
            with t3: st.markdown(metric_html("总 GPU Hours", f"{result_t2['Total_gpu_hours']:.2f}", "🔋"), unsafe_allow_html=True)
            with t4: st.markdown(metric_html("总训练天数", f"{result_t2['Total_days']:.2f}", "📅"), unsafe_allow_html=True) 
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Placeholder for output area in Tab 2
            st.markdown("""
            <div style="
                border: 2px dashed rgba(0, 122, 255, 0.2); 
                border-radius: 12px; 
                padding: 20px; 
                height: 100%; 
                min-height: 500px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: rgba(255, 255, 255, 0.4);
            " id="output-placeholder-tab2">
                <div style="color: #007AFF; font-weight: 600; font-size: 1.2em; margin-bottom: 10px;">🔄 分布式推演结果输出区</div>
                <div style="color: #8E8E93;">请在左侧配置分布式策略并点击“开始推演”</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Designed By Resource Operation · 2025
</div>
""", unsafe_allow_html=True)
