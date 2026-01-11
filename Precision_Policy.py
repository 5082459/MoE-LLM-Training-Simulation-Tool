# backend/core/precision.py
from pydantic import BaseModel
from typing import Dict


class PrecisionPolicy(BaseModel):
    """
    精度策略：按角色拆分（param/act/grad/opt/comm），支持混合精度。
    """
    name: str

    # 计算精度：算子内部 FMA 用什么 dtype
    compute_w_dtype: str = "FP8"
    compute_a_dtype: str = "BF16"
    compute_accum_dtype: str = "FP32"

    # 存储精度
    storage_param_dtype: str = "FP8"              # 训练时真正参与 matmul 的权重
    storage_master_param_dtype:str = "FP32"  # master weights，没有就 None
    storage_act_dtype: str = "BF16"
    storage_grad_param_dtype: str = "FP32"
    storage_grad_act_dtype: str = "BF16"
    storage_opt_state_dtype: str = "BF16"

    # 通信精度
    comm_act_dtype: str = "BF16"
    comm_grad_param_dtype: str = "BF16"
    comm_grad_act_dtype: str = "BF16"

    # 特定算子强制使用的 compute dtype，例如 LN/softmax 用 fp32
    # key = op_type（字符串），value = dtype
    op_compute_overrides: Dict[str, str] = {}


# 一些预设策略，可以后续慢慢扩展
PRESET_POLICIES: Dict[str, PrecisionPolicy] = {
    "BF16_F8_E4M3_F32": PrecisionPolicy(
        name="BF16_F8_E4M3_F32",
        compute_w_dtype="FP8",
        compute_a_dtype="BF16",
        compute_accum_dtype="FP32",
        storage_param_dtype="FP8",
        storage_master_param_dtype="FP32",
        storage_act_dtype="BF16",
        storage_grad_param_dtype="FP32",
        storage_grad_act_dtype="BF16",
        storage_opt_state_dtype="BF16",
        comm_act_dtype="BF16",
        comm_grad_param_dtype="FP32",
        comm_grad_act_dtype="BF16",
        op_compute_overrides={
            "layernorm": "FP32",
            "softmax": "FP32",
        },
    ),
    "FP16_F32": PrecisionPolicy(
        name="fp16_train_fp32_master",
        compute_w_dtype="FP16",
        compute_a_dtype="FP16",
        compute_accum_dtype="FP32",
        storage_param_dtype="FP16",
        storage_master_param_dtype="FP32",
        storage_act_dtype="FP16",
        storage_grad_param_dtype="FP16",
        storage_grad_act_dtype="FP16",
        storage_opt_state_dtype="FP32",
        comm_act_dtype="FP16",
        comm_grad_act_dtype="FP16",
        op_compute_overrides={
            "layernorm": "fp32",
            "softmax": "fp32",
        },
    ),
}


def get_policy(name: str) -> PrecisionPolicy:
    if name not in PRESET_POLICIES:
        raise ValueError(f"Unknown precision policy: {name}")
    return PRESET_POLICIES[name]