from pydantic import BaseModel


class ModelConfig(BaseModel):
    # 模型结构-默认MOE*Dense，当前业界主流模型都是MOE，前几层是通用Dense，如deepseek-v3.1,当前注意力组件采用原始MHA实现，后续补充MLA
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_dense_hidden_layers: int
    num_moe_hidden_layers: int
    dense_intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    
    n_routed_experts: int 
    n_shared_experts: int
    num_experts_per_tok: int

    #常量系数配置
    capacity_factor: float = 1.125  # token capacity factor for MoE
    alpha_silu: int = 6                 # Silu激活单元素数量
    beta_bias: int = 1                 #偏置项计算开关
    gamma_rms:  int = 4                 # RMSNorm归一化单元素系数
    delta_softmax: int =5               #Softmax单元素计算量
    epsilon_topk: int = 3                 #moe router 单token计算量/单专家
    gamma_adam: int = 2                 # Adam优化器状态系数
class TrainConfig(BaseModel):
    global_batch_size: int          # GBS = dp_size * micro_batch_size * #grad_accum
    num_micro_batches: int          # gradient accumulation steps
    seq_len: int                    # 实际训练 seq_len
    total_tokens: float             # 总训练 token 数（T_tokens）

    use_activation_checkpoint: bool = True
    ckpt_ratio: float = 0.5         # 被 checkpoint 覆盖的 layer 比例，粗略因子
    use_zero_: bool = True  # 是否开启 zero 优化
    zero_stage: int = 2     # zero 优化 stage
    system_overhead_ratio: float = 0.1  # 系统开销比例，建议设置为 0.05 到 0.1 之间

class PrecisionConfig(BaseModel):
    policy_name: str = "BF16_F8_E4M3_F32"

class ParallelConfig(BaseModel):
    dp_size: int                    # data parallel size
    pp_size: int                    # pipeline parallel size
    ep_size: int                    # expert parallel size
    tp_size: int                    # tensor parallel size

class GPU_Resource(BaseModel):
    num_gpus: int                   # GPU 数量
    gpu_sku: str                    # GPU 型号             

class SimulationInput(BaseModel):
    model: ModelConfig
    train: TrainConfig
    precision: PrecisionConfig          