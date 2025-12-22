# backend/core/hw_profiles.py
from pydantic import BaseModel
from typing import Dict


class GPUHwProfile(BaseModel):
    """
    单种 GPU 的硬件规格：不同精度峰值算力、HBM 容量/带宽、互联带宽、dtype 字节数。
    """
    name: str

    peak_flops: Dict[str, float]        # 各 dtype 峰值算力（FLOPs/s）
    hbm_bandwidth: float                # bytes/s
    hbm_capacity: float                 # bytes
    scaleup_bandwidth: float            # TP域bytes/s，单卡有效双向带宽估计
    scaleout_bandwidth: float           # TP域外bytes/s，单卡有效双向带宽估计
    scaleup_boundary: int               # scaleup 边界，<=该规模使用 scaleup 带宽计算，>该规模使用 scaleout 带宽计算


GPU_PROFILES: Dict[str, GPUHwProfile] = {
    "H100_80G_SXM": GPUHwProfile(
        name="H100_80G_SXM",
        peak_flops={
            "fp32": 67e12,
            "tf32": 494e12,
            "bf16": 989e12,
            "fp16": 989e12,
            "fp8": 1980e12,
        },
        hbm_bandwidth=3.35e12,
        hbm_capacity=80 * 1024**3,
        scaleup_bandwidth=900e9,
        scaleout_bandwidth=100e9,
        scaleup_boundary=8,
    ),
    "A100_80G_SXM": GPUHwProfile(
        name="A100_80G_SXM",
        peak_flops={
            "fp32": 19.5e12,
            "tf32": 156e12,
            "bf16": 312e12,
            "fp16": 312e12,
            "fp8": 0.0,   # 没有 fp8
        },
        hbm_bandwidth=2.0e12,
        hbm_capacity=80 * 1024**3,
        scaleup_bandwidth=600e9,
        scaleout_bandwidth=100e9,
        scaleup_boundary=8,
    ),
    "Ascend_910B2": GPUHwProfile(
        name="Ascend_910B2",
        peak_flops={
            "fp32":99e12,
            "tf32": 0e12,
            "bf16": 364e12,
            "fp16": 376e12,
            "fp8": 0.0,   # 没有 fp8
        },
        hbm_bandwidth=1.6e12,
        hbm_capacity=64 * 1024**3,
        scaleup_bandwidth=392e9,
        scaleout_bandwidth=50e9,
        scaleup_boundary=8,
    ),
    "Ascend_910C2": GPUHwProfile(
        name="Ascend_910C2",
        peak_flops={
            "fp32":198e12,
            "tf32": 0e12,
            "bf16": 728e12,
            "fp16": 752e12,
            "fp8": 0.0,   # 没有 fp8
        },
        hbm_bandwidth=3.2e12,
        hbm_capacity=128 * 1024**3,
        scaleup_bandwidth=784e9,
        scaleout_bandwidth=100e9,
        scaleup_boundary=384,
    ),

    "H800_80G_SXM": GPUHwProfile(
        name="H800_80G_SXM",
        peak_flops={
            "fp32": 67e12,
            "tf32": 494e12,
            "bf16": 989e12,
            "fp16": 989e12,
            "fp8": 1980e12,
        },
        hbm_bandwidth=3.35e12,
        hbm_capacity=80 * 1024**3,
        scaleup_bandwidth=400e9,
        scaleout_bandwidth=100e9,
        scaleup_boundary=8,
    ),

    "H20_96G": GPUHwProfile(
        name="H20_96G",
        peak_flops={
            "fp32": 40e12,
            "tf32": 70e12,
            "bf16": 148e12,
            "fp16": 148e12,
            "fp8": 296e12,
        },
        hbm_bandwidth=4e12,
        hbm_capacity=96 * 1024**3,
        scaleup_bandwidth=900e9,
        scaleout_bandwidth=100e9,
        scaleup_boundary=8,
    ),
}


def get_gpu_profile(name: str) -> GPUHwProfile:
    if name not in GPU_PROFILES:
        raise ValueError(f"Unknown GPU profile: {name}")
    return GPU_PROFILES[name]