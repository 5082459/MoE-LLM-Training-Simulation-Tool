from math import log
from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from Precision_Policy_fork4 import get_policy
from User_Config_fork4 import ModelConfig, PrecisionConfig, TrainConfig    

#--------------------定义数据精度层级------------------
dtype_bytes ={"FP32":4,"TF32":4,"FP16":2,"BF16":2,"FP8":1}
#--------------------定义组件张量类型与Spec--------------------
class globle_tensor_role:
    #定义组件张量类型
    PARAM = "param"
    MASTER_PARAM = "master_param"
    ACTIVATION = "activation"
    OPTIMIZER_STATE = "optimizer_state"
    PARAM_GRADIENT = "param_gradient"
    INPUT_GRADIENT = "input_gradient"

class tensor_spec(BaseModel):
    role: Literal[
        globle_tensor_role.PARAM,
        globle_tensor_role.MASTER_PARAM,
        globle_tensor_role.ACTIVATION,
        globle_tensor_role.OPTIMIZER_STATE,
        globle_tensor_role.PARAM_GRADIENT,
        globle_tensor_role.INPUT_GRADIENT,
    ]
    layer_id: int
    component_type: str
    shape: Dict[str, Any]
    dtype: str
    element_count: int
    hbm_occupy: int

#--------------------定义全局模型组件类型--------------------
class model_component:
    #定义全量可选的模型逻辑组件
    TOKENIZER = "tokenizer"
    EMBEDDING = "embedding"
    POS_EMBEDDING = "pos_embedding"
    RMSNORM = "rmsnorm"
    RESIDUAL = "residual"
    MHA_QKV = "mha_qkv"
    MHA_CORE = "mha_core"
    MHA_OUT = "mha_out"
    GATED_FFN_UP = "gated_ffn_up"
    GATED_FFN_NONLINEAR = "gated_ffn_nonlinear"
    GATED_FFN_DOWN = "gated_ffn_down"
    LOGITS = "logits"
    ENTROPYLOSS = "entropyloss"

    # MoE 特有组件
    MOE_ROUTER = "moe_router"
    MOE_ALL2ALL_DISPATCH = "moe_all2all_dispatch"
    MOE_EXPERT_FFN_UP = "moe_expert_ffn_up"
    MOE_EXPERT_FFN_NONLINEAR = "moe_expert_ffn_nonlinear"
    MOE_EXPERT_FFN_DOWN = "moe_expert_ffn_down"
    MOE_ALL2ALL_GATHER = "moe_all2all_gather"

#--------------------定义全局模型组件计算规格--------------------
class model_component_compute_spec(BaseModel):
    name: str
    component_type: Literal[
        model_component.TOKENIZER,
        model_component.EMBEDDING,
        model_component.POS_EMBEDDING,
        model_component.RMSNORM,
        model_component.RESIDUAL,
        model_component.MHA_QKV,
        model_component.MHA_CORE,
        model_component.MHA_OUT,
        model_component.GATED_FFN_UP,
        model_component.GATED_FFN_NONLINEAR,
        model_component.GATED_FFN_DOWN,
        model_component.LOGITS,
        model_component.ENTROPYLOSS,
        model_component.MOE_ROUTER,
        model_component.MOE_ALL2ALL_DISPATCH,
        model_component.MOE_EXPERT_FFN_UP,
        model_component.MOE_EXPERT_FFN_NONLINEAR,
        model_component.MOE_EXPERT_FFN_DOWN,
        model_component.MOE_ALL2ALL_GATHER,
    ]
    layer_id: int
    fwd_flops: Dict[str,float]
    bwd_flops: Dict[str,float]
    fwd_hbm_throughput: Dict[str,float]
    bwd_hbm_throughput: Dict[str,float]   

#--------------------定义模型层计算规格类------------------------    
class model_layer_compute_spec(BaseModel):
    lidx: int
    components: List[model_component_compute_spec]

#-------------------定义模型层张量规格类-----------------------
class model_layer_tensor_spec(BaseModel):
    layer_id: int
    tensors: List[tensor_spec]

#--------------------定义理论模型计算图结构类--------------------------
class model_compute_graph(BaseModel):
    layers: List[model_layer_compute_spec]
    embedding: model_component_compute_spec
    pos_embedding: model_component_compute_spec
    logits: model_component_compute_spec
    entropyloss: model_component_compute_spec

#--------------------定义理论模型张量结构类--------------------------
class model_tensor_graph(BaseModel):
    layers: List[model_layer_tensor_spec]

#--------------------定义理论模型计算&显存结构类--------------------------
class model_compute_and_memory_graph(BaseModel):
    compute_graph: model_compute_graph
    tensor_graph: model_tensor_graph

#--------------------解析用户输入，构建模型逻辑计算图--------------------------
def parse_model_compute_spec(
    model_config: ModelConfig,
    train_config: TrainConfig,
    precision_config: PrecisionConfig,
) -> model_compute_graph:   
    
    #根据用户变量输入配置
    GBS = train_config.global_batch_size
    S = train_config.seq_len
    d_model = model_config.hidden_size
    dff_dense = model_config.dense_intermediate_size
    dff_moe = model_config.moe_intermediate_size
    L = model_config.num_hidden_layers
    L_dense = model_config.num_dense_hidden_layers
    L_moe = L - L_dense
    E_router = model_config.n_routed_experts
    E_shared = model_config.n_shared_experts
    E_active = model_config.num_experts_per_tok
    E_total = E_router + E_shared
    V = model_config.vocab_size
    H = model_config.num_attention_heads
    precision_policy = get_policy(precision_config.policy_name)
    
    #定义常量系数
    alpha_silu = model_config.alpha_silu
    beta_bias = model_config.beta_bias
    gamma_rms = model_config.gamma_rms
    delta_softmax = model_config.delta_softmax
    epsilon_topk = model_config.epsilon_topk
    
    #初始化模型层图与组件图
    layers:List[model_layer_compute_spec] = []

    #Embedding 
    F_emb_fwd = GBS * S * (d_model + beta_bias)
    F_emb_bwd = GBS * S * (d_model + beta_bias)
    V_HBM_throughput_emb_fwd = F_emb_fwd * dtype_bytes[precision_policy.compute_accum_dtype]
    V_HBM_throughput_emb_bwd = F_emb_bwd * dtype_bytes[precision_policy.compute_accum_dtype]
    emb_comp = model_component_compute_spec(
        name="embedding",
        component_type=model_component.EMBEDDING,
        layer_id= 1,
        fwd_flops={precision_policy.compute_accum_dtype:F_emb_fwd},
        bwd_flops={precision_policy.compute_accum_dtype:F_emb_bwd},
        fwd_hbm_throughput={precision_policy.compute_accum_dtype:V_HBM_throughput_emb_fwd},
        bwd_hbm_throughput={precision_policy.compute_accum_dtype:V_HBM_throughput_emb_bwd},
    )

    #Pos Embedding
    F_pos_emb_fwd = GBS * S * (d_model + beta_bias)
    F_pos_emb_bwd = GBS * S * (d_model + beta_bias)
    V_HBM_throughput_pos_emb_fwd = F_pos_emb_fwd * dtype_bytes[precision_policy.compute_accum_dtype]
    V_HBM_throughput_pos_emb_bwd = F_pos_emb_bwd * dtype_bytes[precision_policy.compute_accum_dtype]
    pos_emb_comp = model_component_compute_spec(
        name="pos_embedding",
        component_type=model_component.EMBEDDING,
        layer_id= 1,
        fwd_flops={precision_policy.compute_accum_dtype:F_pos_emb_fwd},
        bwd_flops={precision_policy.compute_accum_dtype:F_pos_emb_bwd},
        fwd_hbm_throughput={precision_policy.compute_accum_dtype:V_HBM_throughput_pos_emb_fwd},
        bwd_hbm_throughput={precision_policy.compute_accum_dtype:V_HBM_throughput_pos_emb_bwd},
    )
    
    #Logits
    F_logits_fwd = GBS * S * (d_model * V + beta_bias * 4)
    F_logits_bwd = GBS * S * (2 * d_model * V + beta_bias * 4)
    V_HBM_throughput_logits_fwd = GBS * S * (V + beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
    V_HBM_throughput_logits_bwd = GBS * S * (V + beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
    logits_comp = model_component_compute_spec(
        name="logits",
        component_type=model_component.LOGITS,
        layer_id=L,
        fwd_flops={precision_policy.compute_w_dtype:F_logits_fwd},
        bwd_flops={precision_policy.compute_w_dtype:F_logits_bwd},
        fwd_hbm_throughput={precision_policy.compute_a_dtype:V_HBM_throughput_logits_fwd},
        bwd_hbm_throughput={precision_policy.compute_a_dtype:V_HBM_throughput_logits_bwd},
    )

    #Entropy Loss
    F_entropyloss_fwd = GBS * S * (delta_softmax * V + V * log(V) + 1)
    F_entropyloss_bwd = GBS * S * ((delta_softmax +2 ) * V + V * log(V) + 1)
    V_HBM_throughput_entropyloss_fwd = GBS * S * V * dtype_bytes[precision_policy.compute_a_dtype]
    V_HBM_throughput_entropyloss_bwd = GBS * S * V * dtype_bytes[precision_policy.compute_a_dtype]
    entropyloss_comp = model_component_compute_spec(
    name="entropyloss",
    component_type=model_component.ENTROPYLOSS,
    layer_id=L,
    fwd_flops={precision_policy.compute_accum_dtype:F_entropyloss_fwd},
    bwd_flops={precision_policy.compute_accum_dtype:F_entropyloss_bwd},
    fwd_hbm_throughput={precision_policy.compute_a_dtype:V_HBM_throughput_entropyloss_fwd},
    bwd_hbm_throughput={precision_policy.compute_a_dtype:V_HBM_throughput_entropyloss_bwd},
    )

#解析Transformer层
    for l in range(1, L + 1):
        components: List[model_component_compute_spec] = []
        #根据用户配置计算组件的spec属性
        #首先配置所有layers的通用组件
        #RMSNorm1
        Flops_RMSNORM1_FWD = GBS * S * d_model * gamma_rms + GBS * S * 2 * beta_bias
        Flops_RMSNORM1_BWD = GBS * S * d_model * (gamma_rms + 2) + GBS * S * 2 * beta_bias
        Through_HBM_RMSNORM1_FWD = (GBS * S * d_model + GBS * S * 2 * beta_bias)*dtype_bytes[precision_policy.compute_accum_dtype]
        Through_HBM_RMSNORM1_BWD = (GBS * S * d_model + GBS * S * 2 * beta_bias)*dtype_bytes[precision_policy.compute_accum_dtype]
        RMSNorm1_comp = model_component_compute_spec(
            name=f"lay{l}_RMSNorm1",
            component_type=model_component.RMSNORM,
            layer_id=l,
            fwd_flops={precision_policy.compute_accum_dtype:Flops_RMSNORM1_FWD},
            bwd_flops={precision_policy.compute_accum_dtype:Flops_RMSNORM1_BWD},
            fwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RMSNORM1_FWD},
            bwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RMSNORM1_BWD},
        )  
        components.append(RMSNorm1_comp)
    
        #MHA-QKV project
        Flops_MHAQKV_FWD = 3 * GBS * S * (d_model * d_model + beta_bias * 4)
        Flops_MHAQKV_BWD = 3 * GBS * S * (2 * d_model * d_model + beta_bias * 4)
        Through_HBM_MHAQKV_FWD = 3 * (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
        Through_HBM_MHAQKV_BWD = 3 * (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
        MHAQKV_comp = model_component_compute_spec(
            name=f"lay{l}_MHAQKV",
            component_type=model_component.MHA_QKV,
            layer_id=l,
            fwd_flops={precision_policy.compute_w_dtype:Flops_MHAQKV_FWD},
            bwd_flops={precision_policy.compute_w_dtype:Flops_MHAQKV_BWD},
            fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHAQKV_FWD},
            bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHAQKV_BWD},
        )
        components.append(MHAQKV_comp)

        #MHA-CORE QK^T + softmax + P*V #默认使用Flash attention
        Flops_MHACORE_FWD = H * GBS * S * (S * (d_model / H + delta_softmax) + (d_model / H) * S)
        Flops_MHACORE_BWD = H * GBS * S * (2 * S * (d_model / H + delta_softmax) + 2 * (d_model / H) * S)
        Through_HBM_MHACORE_FWD = 4 * GBS * S * d_model * dtype_bytes[precision_policy.compute_a_dtype]
        Through_HBM_MHACORE_BWD = 6 * GBS * S * d_model * dtype_bytes[precision_policy.compute_a_dtype]
        MHACORE_comp = model_component_compute_spec(
            name=f"lay{l}_MHACORE",
            component_type=model_component.MHA_CORE,
            layer_id=l,
            fwd_flops={precision_policy.compute_w_dtype:Flops_MHACORE_FWD},
            bwd_flops={precision_policy.compute_w_dtype:Flops_MHACORE_BWD},
            fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHACORE_FWD},
            bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHACORE_BWD},  
        )
        components.append(MHACORE_comp)

        #self-attention output project
        Flops_MHAOUT_FWD = GBS * S * (d_model * d_model + beta_bias * 4)
        Flops_MHAOUT_BWD = GBS * S * (2 * d_model * d_model + beta_bias * 4)
        Through_HBM_MHAOUT_FWD = (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
        Through_HBM_MHAOUT_BWD = (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype] 
        MHAOUT_comp = model_component_compute_spec(
            name=f"lay{l}_MHAOUT",
            component_type=model_component.MHA_OUT,
            layer_id=l,
            fwd_flops={precision_policy.compute_w_dtype:Flops_MHAOUT_FWD},
            bwd_flops={precision_policy.compute_w_dtype:Flops_MHAOUT_BWD},
            fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHAOUT_FWD},
            bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MHAOUT_BWD},
        )
        components.append(MHAOUT_comp)

        #Residual1
        Flops_RES1_FWD = GBS * S * d_model
        Flops_RES1_BWD = GBS * S * d_model
        Through_HBM_RES1_FWD = 2 * Flops_RES1_FWD * dtype_bytes[precision_policy.compute_accum_dtype]
        Through_HBM_RES1_BWD = 2 * Flops_RES1_BWD * dtype_bytes[precision_policy.compute_accum_dtype]
        RES1_comp = model_component_compute_spec(
            name=f"lay{l}_RES1",
            component_type=model_component.RESIDUAL,
            layer_id=l,
            fwd_flops={precision_policy.compute_accum_dtype:Flops_RES1_FWD},
            bwd_flops={precision_policy.compute_accum_dtype:Flops_RES1_BWD},
            fwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RES1_FWD},
            bwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RES1_BWD},
        )
        components.append(RES1_comp)

        #RMSNorm2
        Flops_RMSNORM2_FWD = GBS * S * d_model * gamma_rms + GBS * S * 2 * beta_bias
        Flops_RMSNORM2_BWD = GBS * S * d_model * (gamma_rms + 2) + GBS * S * 2 * beta_bias
        Through_HBM_RMSNORM2_FWD = (GBS * S * d_model + GBS * S * 2 * beta_bias)*dtype_bytes[precision_policy.compute_accum_dtype]
        Through_HBM_RMSNORM2_BWD = (GBS * S * d_model + GBS * S * 2 * beta_bias)*dtype_bytes[precision_policy.compute_accum_dtype]
        RMSNorm2_comp = model_component_compute_spec(
            name=f"lay{l}_RMSNorm2",
            component_type=model_component.RMSNORM,
            layer_id=l,
            fwd_flops={precision_policy.compute_accum_dtype:Flops_RMSNORM2_FWD},
            bwd_flops={precision_policy.compute_accum_dtype:Flops_RMSNORM2_BWD},
            fwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RMSNORM2_FWD},
            bwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RMSNORM2_BWD},
        )
        components.append(RMSNorm2_comp)

        #Residual2
        Flops_RES2_FWD = GBS * S * d_model
        Flops_RES2_BWD = GBS * S * d_model
        Through_HBM_RES2_FWD = 2 * (GBS * S * d_model)*dtype_bytes[precision_policy.compute_accum_dtype]
        Through_HBM_RES2_BWD = 2 * (GBS * S * d_model)*dtype_bytes[precision_policy.compute_accum_dtype]
        RES2_comp = model_component_compute_spec(
            name=f"lay{l}_RES2",
            component_type=model_component.RESIDUAL,
            layer_id=l,
            fwd_flops={precision_policy.compute_accum_dtype:Flops_RES2_FWD},
            bwd_flops={precision_policy.compute_accum_dtype:Flops_RES2_BWD},
            fwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RES2_FWD},
            bwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_RES2_BWD},
        )
        components.append(RES2_comp)

        #处理Dense Layers的FLOPs计算逻辑
        if l <= L_dense:
            #Dense GATED_FFN_up
            Flops_GATED_FFN_UP_FWD = 2 * GBS * S * (d_model * dff_dense + beta_bias*4)
            Flops_GATED_FFN_UP_BWD = 2 * GBS * S * (2 * d_model * dff_dense + beta_bias*4)
            Through_HBM_GATED_FFN_UP_FWD = 2 * (GBS * S * dff_dense + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_GATED_FFN_UP_BWD = 2 * (GBS * S * dff_dense + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            GATED_FFN_UP_comp = model_component_compute_spec(
                name=f"lay{l}_GATED_FFN_UP",
                component_type=model_component.GATED_FFN_UP,
                layer_id=l,
                fwd_flops={precision_policy.compute_w_dtype:Flops_GATED_FFN_UP_FWD},
                bwd_flops={precision_policy.compute_w_dtype:Flops_GATED_FFN_UP_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_GATED_FFN_UP_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_GATED_FFN_UP_BWD},
            )
            components.append(GATED_FFN_UP_comp)

            #Dense GATED_FFN_NONLINEAR
            Flops_GATED_FFN_NONLINEAR_FWD = 2 * GBS * S * dff_dense * alpha_silu
            Flops_GATED_FFN_NONLINEAR_BWD = 2 * GBS * S * dff_dense * (alpha_silu + 2)
            Through_HBM_GATED_FFN_NONLINEAR_FWD = 2 * GBS * S * dff_dense * dtype_bytes[precision_policy.compute_accum_dtype]
            Through_HBM_GATED_FFN_NONLINEAR_BWD = 2 * GBS * S * dff_dense * dtype_bytes[precision_policy.compute_accum_dtype]
            GATED_FFN_NONLINEAR_comp = model_component_compute_spec(
                name=f"lay{l}_GATED_FFN_NONLINEAR",
                component_type=model_component.GATED_FFN_NONLINEAR,
                layer_id=l,
                fwd_flops={precision_policy.compute_accum_dtype:Flops_GATED_FFN_NONLINEAR_FWD},
                bwd_flops={precision_policy.compute_accum_dtype:Flops_GATED_FFN_NONLINEAR_BWD},
                fwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_GATED_FFN_NONLINEAR_FWD},
                bwd_hbm_throughput={precision_policy.compute_accum_dtype:Through_HBM_GATED_FFN_NONLINEAR_BWD},
            )
            components.append(GATED_FFN_NONLINEAR_comp)

            #Dense GATED_FFN_DOWN
            Flops_GATED_FFN_DOWN_FWD = GBS * S * (d_model * dff_dense + beta_bias * 4)
            Flops_GATED_FFN_DOWN_BWD = GBS * S * (2 * d_model * dff_dense + beta_bias * 4)
            Through_HBM_GATED_FFN_DOWN_FWD = (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_GATED_FFN_DOWN_BWD = (GBS * S * d_model + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            GATED_FFN_DOWN_comp = model_component_compute_spec(
                name=f"lay{l}_GATED_FFN_DOWN",
                component_type=model_component.GATED_FFN_DOWN,
                layer_id=l,
                fwd_flops={precision_policy.compute_w_dtype:Flops_GATED_FFN_DOWN_FWD},
                bwd_flops={precision_policy.compute_w_dtype:Flops_GATED_FFN_DOWN_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_GATED_FFN_DOWN_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_GATED_FFN_DOWN_BWD},
            )
            components.append(GATED_FFN_DOWN_comp)
    
#------------------------------------------MoE层解析------------------------------------------
        elif l > L_dense:
            #Moe_FFN_router
            Flops_MoeFFNRouter_FWD = GBS * S * (d_model * E_total + E_total * delta_softmax * 2 + E_total * epsilon_topk * 2 + beta_bias * 4)
            Flops_MoeFFNRouter_BWD = GBS * S * (2 * d_model * E_total + E_total * (delta_softmax * 2 + 2) + E_total * (epsilon_topk * 2 + 1) + beta_bias * 4)
            Through_HBM_MoeFFNRouter_FWD = (GBS * S * E_total + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_MoeFFNRouter_BWD = (GBS * S * E_total + GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            MoeFFNRouter_comp = model_component_compute_spec(
                name=f"lay{l}_MoeFFNRouter",
                component_type=model_component.MOE_ROUTER,
                layer_id=l,
                fwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNRouter_FWD},
                bwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNRouter_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNRouter_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNRouter_BWD},
            )
            components.append(MoeFFNRouter_comp)

            #Moe_FFN_Expert_UP
            Flops_MoeFFNExpert_UP_FWD = E_active * 2 * GBS * S * (d_model * dff_moe + beta_bias * 4)
            Flops_MoeFFNExpert_UP_BWD = E_active * 2 * GBS * S * (2 * d_model * dff_moe + beta_bias * 4)
            Through_HBM_MoeFFNExpert_UP_FWD = 2 * (E_active * GBS * S * dff_moe + E_active * GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_MoeFFNExpert_UP_BWD = 2 * (E_active * GBS * S * dff_moe + E_active * GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            MoeFFNExpert_UP_comp = model_component_compute_spec(
                name=f"lay{l}_MoeFFNExpert_UP",
                component_type=model_component.MOE_EXPERT_FFN_UP,
                layer_id=l,
                fwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNExpert_UP_FWD},
                bwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNExpert_UP_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_UP_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_UP_BWD},
            )
            components.append(MoeFFNExpert_UP_comp)

            #Moe_FFN_Expert_NONLINEAR
            Flops_MoeFFNExpert_NONLINEAR_FWD = E_active * 2 * GBS * S * dff_moe * alpha_silu
            Flops_MoeFFNExpert_NONLINEAR_BWD = E_active * 2 * GBS * S * dff_moe * (alpha_silu + 2 )
            Through_HBM_MoeFFNExpert_NONLINEAR_FWD = 2 * (E_active * GBS * S * dff_moe) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_MoeFFNExpert_NONLINEAR_BWD = 2 * (E_active * GBS * S * dff_moe) * dtype_bytes[precision_policy.compute_a_dtype]
            MoeFFNExpert_NONLINEAR_comp = model_component_compute_spec(
                name=f"lay{l}_MoeFFNExpert_NONLINEAR",
                component_type=model_component.MOE_EXPERT_FFN_NONLINEAR,
                layer_id=l,
                fwd_flops={precision_policy.compute_accum_dtype:Flops_MoeFFNExpert_NONLINEAR_FWD},
                bwd_flops={precision_policy.compute_accum_dtype:Flops_MoeFFNExpert_NONLINEAR_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_NONLINEAR_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_NONLINEAR_BWD},
            )
            components.append(MoeFFNExpert_NONLINEAR_comp)

            #Moe_FFN_Expert_DOWN
            Flops_MoeFFNExpert_DOWN_FWD = E_active * GBS * S * (dff_moe * d_model + beta_bias * 4)
            Flops_MoeFFNExpert_DOWN_BWD = E_active * GBS * S * (2 * dff_moe * d_model + beta_bias * 4)
            Through_HBM_MoeFFNExpert_DOWN_FWD = (E_active * GBS * S * d_model + E_active * GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            Through_HBM_MoeFFNExpert_DOWN_BWD = (E_active * GBS * S * d_model + E_active * GBS * S * beta_bias) * dtype_bytes[precision_policy.compute_a_dtype]
            MoeFFNExpert_DOWN_comp = model_component_compute_spec(
                name=f"lay{l}_MoeFFNExpert_DOWN",
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                layer_id=l,
                fwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNExpert_DOWN_FWD},
                bwd_flops={precision_policy.compute_w_dtype:Flops_MoeFFNExpert_DOWN_BWD},
                fwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_DOWN_FWD},
                bwd_hbm_throughput={precision_policy.compute_a_dtype:Through_HBM_MoeFFNExpert_DOWN_BWD},
            )
            components.append(MoeFFNExpert_DOWN_comp)

        layers.append(model_layer_compute_spec(lidx=l,components=components))

    return model_compute_graph(
        layers=layers,
        embedding=emb_comp,
        pos_embedding=pos_emb_comp,
        logits=logits_comp,
        entropyloss=entropyloss_comp,
    )
#------------------------根据用户输入解析构建模型组件张量图------------------------
def parse_model_tensor_spec(
    model_config: ModelConfig,
    train_config: TrainConfig,
    precision_config: PrecisionConfig,
) -> model_tensor_graph:
    #解析用户配置输入
    GBS = train_config.global_batch_size
    S = train_config.seq_len
    d_model = model_config.hidden_size
    dff_dense = model_config.dense_intermediate_size
    dff_moe = model_config.moe_intermediate_size
    L = model_config.num_hidden_layers
    L_dense = model_config.num_dense_hidden_layers
    L_moe = L - L_dense
    E_router = model_config.n_routed_experts
    E_shared = model_config.n_shared_experts
    E_active = model_config.num_experts_per_tok
    E_total = E_router + E_shared
    V = model_config.vocab_size
    H = model_config.num_attention_heads
    precision_policy = get_policy(precision_config.policy_name)
    
    gamma_adam = model_config.gamma_adam
    beta_bias = model_config.beta_bias

    model_tensors:List[model_layer_tensor_spec] = []
    
    # Embedding层张量（独立层）
    layer_tensors:List[tensor_spec] = []
    #Embedding层张量
    emb_param_tensor = tensor_spec(
        role=globle_tensor_role.PARAM,
        layer_id=1,
        component_type=model_component.EMBEDDING,
        shape={"row":model_config.vocab_size,"column":d_model,"Bias":d_model},
        dtype=precision_policy.storage_param_dtype,
        element_count=model_config.vocab_size * d_model + d_model * beta_bias,
        hbm_occupy=(model_config.vocab_size * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
    )
    layer_tensors.append(emb_param_tensor)

    emb_master_param_tensor = tensor_spec(
        role=globle_tensor_role.MASTER_PARAM,
        layer_id=1,
        component_type=model_component.EMBEDDING,
        shape={"row":model_config.vocab_size,"column":d_model,"Bias":d_model},
        dtype=precision_policy.storage_master_param_dtype,
        element_count=model_config.vocab_size * d_model + d_model * beta_bias,
        hbm_occupy=(model_config.vocab_size * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
    )
    layer_tensors.append(emb_master_param_tensor)

    emb_activation_tensor = tensor_spec(
        role=globle_tensor_role.ACTIVATION,
        layer_id=1,
        component_type=model_component.EMBEDDING,
        shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
        dtype=precision_policy.storage_act_dtype,
        element_count=GBS * S * d_model,
        hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
    )
    layer_tensors.append(emb_activation_tensor)

    emb_optimizer_state_tensor = tensor_spec(
        role=globle_tensor_role.OPTIMIZER_STATE,
        layer_id=1,
        component_type=model_component.EMBEDDING,
        shape={"row":model_config.vocab_size,"column":d_model,"Bias":d_model,"state":int(gamma_adam)},
        dtype=precision_policy.storage_opt_state_dtype,
        element_count=(model_config.vocab_size * d_model + d_model * beta_bias) * gamma_adam,
        hbm_occupy=(model_config.vocab_size * d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
    )
    layer_tensors.append(emb_optimizer_state_tensor)

    emb_param_gradient_tensor = tensor_spec(
        role=globle_tensor_role.PARAM_GRADIENT,
        layer_id=1,
        component_type=model_component.EMBEDDING,
        shape={"row":model_config.vocab_size,"column":d_model,"Bias":d_model},
        dtype=precision_policy.storage_grad_param_dtype,
        element_count=model_config.vocab_size * d_model + d_model * beta_bias,
        hbm_occupy=(model_config.vocab_size * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype], 
    )
    layer_tensors.append(emb_param_gradient_tensor)

    #Positional Embedding层张量
    pos_emb_param_tensor = tensor_spec(
        role=globle_tensor_role.PARAM,
        layer_id=1,
        component_type=model_component.POS_EMBEDDING,
        shape={"seq_len":S,"hidden_dim":d_model,"Bias":d_model},
        dtype=precision_policy.storage_param_dtype,
        element_count=S * d_model + d_model * beta_bias,
        hbm_occupy=(S * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
    )
    layer_tensors.append(pos_emb_param_tensor)

    pos_emb_master_param_tensor = tensor_spec(
        role=globle_tensor_role.MASTER_PARAM,
        layer_id=1,
        component_type=model_component.POS_EMBEDDING,
        shape={"seq_len":S,"hidden_dim":d_model,"Bias":d_model},
        dtype=precision_policy.storage_master_param_dtype,
        element_count=S * d_model + d_model * beta_bias,
        hbm_occupy=(S * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
    )
    layer_tensors.append(pos_emb_master_param_tensor)

    pos_emb_activation_tensor = tensor_spec(
        role=globle_tensor_role.ACTIVATION,
        layer_id=1,
        component_type=model_component.POS_EMBEDDING,
        shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
        dtype=precision_policy.storage_act_dtype,
        element_count=GBS * S * d_model,
        hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
    )
    layer_tensors.append(pos_emb_activation_tensor)

    pos_emb_optimizer_state_tensor = tensor_spec(
        role=globle_tensor_role.OPTIMIZER_STATE,
        layer_id=1,
        component_type=model_component.POS_EMBEDDING,
        shape={"seq_len":S,"hidden_dim":d_model,"Bias":d_model,"state":int(gamma_adam)},
        dtype=precision_policy.storage_opt_state_dtype,
        element_count=(S * d_model + d_model * beta_bias) * gamma_adam,
        hbm_occupy=(S * d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
    )
    layer_tensors.append(pos_emb_optimizer_state_tensor)

    pos_emb_param_gradient_tensor = tensor_spec(
        role=globle_tensor_role.PARAM_GRADIENT,
        layer_id=1,
        component_type=model_component.POS_EMBEDDING,
        shape={"seq_len":S,"hidden_dim":d_model,"Bias":d_model},
        dtype=precision_policy.storage_grad_param_dtype,
        element_count=S * d_model + d_model * beta_bias,
        hbm_occupy=(S * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
    )
    layer_tensors.append(pos_emb_param_gradient_tensor)

    model_tensors.append(model_layer_tensor_spec(
        layer_id=1,
        tensors=layer_tensors
    ))

    # Logits和EntropyLoss层张量（独立层）
    layer_tensors:List[tensor_spec] = []
    #Logits层张量
    logits_param_tensor = tensor_spec(
        role=globle_tensor_role.PARAM,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"row":d_model,"column":V,"Bias":V},
        dtype=precision_policy.storage_param_dtype,
        element_count=d_model * V + V * beta_bias,
        hbm_occupy=(d_model * V + V * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
    )
    layer_tensors.append(logits_param_tensor)

    logits_master_param_tensor = tensor_spec(
        role=globle_tensor_role.MASTER_PARAM,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"row":d_model,"column":V,"Bias":V},
        dtype=precision_policy.storage_master_param_dtype,
        element_count=d_model * V + V * beta_bias,
        hbm_occupy=(d_model * V + V * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
    )
    layer_tensors.append(logits_master_param_tensor)

    logits_activation_tensor = tensor_spec(
        role=globle_tensor_role.ACTIVATION,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"batch":GBS,"seq_len":S,"vocab_size":V},
        dtype=precision_policy.storage_act_dtype,
        element_count=GBS * S * V,
        hbm_occupy=GBS * S * V * dtype_bytes[precision_policy.storage_act_dtype],
    )
    layer_tensors.append(logits_activation_tensor)

    logits_optimizer_state_tensor = tensor_spec(
        role=globle_tensor_role.OPTIMIZER_STATE,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"row":d_model,"column":V,"Bias":V,"state":int(gamma_adam)},
        dtype=precision_policy.storage_opt_state_dtype,
        element_count=(d_model * V + V * beta_bias) * gamma_adam,
        hbm_occupy=(d_model * V + V * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
    )
    layer_tensors.append(logits_optimizer_state_tensor)

    logits_param_gradient_tensor = tensor_spec(
        role=globle_tensor_role.PARAM_GRADIENT,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"row":d_model,"column":V,"Bias":V},
        dtype=precision_policy.storage_grad_param_dtype,
        element_count=d_model * V + V * beta_bias,
        hbm_occupy=(d_model * V + V * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
    )
    layer_tensors.append(logits_param_gradient_tensor)

    logits_input_gradient_tensor = tensor_spec(
        role=globle_tensor_role.INPUT_GRADIENT,
        layer_id=L,
        component_type=model_component.LOGITS,
        shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
        dtype=precision_policy.storage_grad_act_dtype,
        element_count=GBS * S * d_model,
        hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
    )
    layer_tensors.append(logits_input_gradient_tensor)

    #Entropy Loss层张量    
    entropyloss_activation_tensor = tensor_spec(
        role=globle_tensor_role.ACTIVATION,
        layer_id=L,
        component_type=model_component.ENTROPYLOSS,
        shape={"batch":GBS,"seq_len":S},
        dtype=precision_policy.storage_act_dtype,
        element_count=GBS * S,
        hbm_occupy=GBS * S * dtype_bytes[precision_policy.storage_act_dtype],
    )
    layer_tensors.append(entropyloss_activation_tensor)

    entropyloss_input_gradient_tensor = tensor_spec(
        role=globle_tensor_role.INPUT_GRADIENT,
        layer_id=L,
        component_type=model_component.ENTROPYLOSS,
        shape={"batch":GBS,"seq_len":S,"vocab_size":V},
        dtype=precision_policy.storage_grad_act_dtype,
        element_count=GBS * S * V,
        hbm_occupy=GBS * S * V * dtype_bytes[precision_policy.storage_grad_act_dtype],
    )
    layer_tensors.append(entropyloss_input_gradient_tensor)
    model_tensors.append(model_layer_tensor_spec(
        layer_id=L,
        tensors=layer_tensors
    ))

    #---------------------------------解析每层的张量--------------------------------------
    for l in range(1,L+1):  
        layer_tensors:List[tensor_spec] = []
        #配置每一层的通用组件的张量
        #RMSNorm1
        RMSNorm1_param_tensor = tensor_spec(
            role=globle_tensor_role.PARAM,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
        )
        layer_tensors.append(RMSNorm1_param_tensor)

        RMSNorm1_master_param_tensor = tensor_spec(
            role=globle_tensor_role.MASTER_PARAM,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_master_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
        )
        layer_tensors.append(RMSNorm1_master_param_tensor)

        RMSNorm1_activation_tensor = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(RMSNorm1_activation_tensor)

        RMSNorm1_optimizer_state_tensor = tensor_spec(
            role=globle_tensor_role.OPTIMIZER_STATE,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model,"state":int(gamma_adam)},
            dtype=precision_policy.storage_opt_state_dtype,
            element_count=(d_model + d_model * beta_bias) * gamma_adam,
            hbm_occupy=(d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
        )
        layer_tensors.append(RMSNorm1_optimizer_state_tensor)

        RMSNorm1_param_gradient_tensor = tensor_spec(
            role=globle_tensor_role.PARAM_GRADIENT,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_grad_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
        )
        layer_tensors.append(RMSNorm1_param_gradient_tensor)

        RMSNorm1_input_gradient_tensor = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.RMSNORM, 
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(RMSNorm1_input_gradient_tensor)

        #MHA-QKV project
        MHAQKV_param_tensor = tensor_spec(
            role=globle_tensor_role.PARAM,
            layer_id=l,
            component_type=model_component.MHA_QKV,
            shape={"Q":(d_model, d_model),"K":(d_model, d_model),"V":(d_model, d_model),
                   "bias_Q":(d_model,),"bias_K":(d_model,),"bias_V":(d_model,)},
            dtype=precision_policy.storage_param_dtype,
            element_count=3 * d_model * d_model + 3 * d_model * beta_bias,
            hbm_occupy=(3 * d_model * d_model + 3 * d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
        )
        layer_tensors.append(MHAQKV_param_tensor)

        MHAQKV_master_param_tensor = tensor_spec(
            role=globle_tensor_role.MASTER_PARAM,
            layer_id=l,
            component_type=model_component.MHA_QKV,
            shape={"Q":(d_model, d_model),"K":(d_model, d_model),"V":(d_model, d_model),
                   "bias_Q":(d_model,),"bias_K":(d_model,),"bias_V":(d_model,)},
            dtype=precision_policy.storage_master_param_dtype,
            element_count=3 * d_model * d_model + 3 * d_model * beta_bias,
            hbm_occupy=(3 * d_model * d_model + 3 * d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
        )
        layer_tensors.append(MHAQKV_master_param_tensor)

        #MHA-CORE
        MHA_CORE_Q_activation = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"d_head":int(d_model/H)},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * H * S * (d_model // H),
            hbm_occupy=GBS * H * S * (d_model // H) * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_CORE_Q_activation)

        MHA_CORE_K_activation = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"d_head":int(d_model/H)},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * H * S * (d_model // H),
            hbm_occupy=GBS * H * S * (d_model // H) * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_CORE_K_activation)

        MHA_CORE_V_activation = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"d_head":int(d_model/H)},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * H * S * (d_model // H),
            hbm_occupy=GBS * H * S * (d_model // H) * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_CORE_V_activation)

        MHA_CORE_attention_activation = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"seq_len":S},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * H * S * S,
            hbm_occupy=GBS * H * S * S * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_CORE_attention_activation)

        MHA_CORE_output_activation = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"d_head":int(d_model/H)},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * H * S * (d_model // H),
            hbm_occupy=GBS * H * S * (d_model // H) * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_CORE_output_activation)

        MHA_CORE_input_gradient = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.MHA_CORE,
            shape={"batch":GBS,"heads":H,"seq_len":S,"d_head":int(d_model/H)},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * H * S * (d_model // H) * 3,  # Q/K/V梯度总和
            hbm_occupy=GBS * H * S * (d_model // H) * 3 * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(MHA_CORE_input_gradient)

        #MHA-OUT
        MHA_OUT_param_tensor = tensor_spec(
            role=globle_tensor_role.PARAM,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"row":d_model,"column":d_model,"Bias":d_model},
            dtype=precision_policy.storage_param_dtype,
            element_count=d_model * d_model + d_model * beta_bias,
            hbm_occupy=(d_model * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
        )
        layer_tensors.append(MHA_OUT_param_tensor)

        MHA_OUT_master_param_tensor = tensor_spec(
            role=globle_tensor_role.MASTER_PARAM,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"row":d_model,"column":d_model,"Bias":d_model},
            dtype=precision_policy.storage_master_param_dtype,
            element_count=d_model * d_model + d_model * beta_bias,
            hbm_occupy=(d_model * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
        )
        layer_tensors.append(MHA_OUT_master_param_tensor)

        MHA_OUT_activation_tensor = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(MHA_OUT_activation_tensor)

        MHA_OUT_optimizer_state_tensor = tensor_spec(
            role=globle_tensor_role.OPTIMIZER_STATE,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"row":d_model,"column":d_model,"Bias":d_model,"state":int(gamma_adam)},
            dtype=precision_policy.storage_opt_state_dtype,
            element_count=(d_model * d_model + d_model * beta_bias) * gamma_adam,
            hbm_occupy=(d_model * d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
        )
        layer_tensors.append(MHA_OUT_optimizer_state_tensor)

        MHA_OUT_param_gradient_tensor = tensor_spec(
            role=globle_tensor_role.PARAM_GRADIENT,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"row":d_model,"column":d_model,"Bias":d_model},
            dtype=precision_policy.storage_grad_param_dtype,
            element_count=d_model * d_model + d_model * beta_bias,
            hbm_occupy=(d_model * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
        )
        layer_tensors.append(MHA_OUT_param_gradient_tensor)

        MHA_OUT_input_gradient_tensor = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.MHA_OUT,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(MHA_OUT_input_gradient_tensor)

        #Residual1
        RES1_activation_tensor = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.RESIDUAL,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(RES1_activation_tensor)

        RES1_input_gradient_tensor = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.RESIDUAL,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(RES1_input_gradient_tensor)

        #RMSNorm2
        RMSNorm2_param_tensor = tensor_spec(
            role=globle_tensor_role.PARAM,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
        )
        layer_tensors.append(RMSNorm2_param_tensor)

        RMSNorm2_master_param_tensor = tensor_spec(
            role=globle_tensor_role.MASTER_PARAM,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_master_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
        )
        layer_tensors.append(RMSNorm2_master_param_tensor)

        RMSNorm2_activation_tensor = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(RMSNorm2_activation_tensor)

        RMSNorm2_optimizer_state_tensor = tensor_spec(
            role=globle_tensor_role.OPTIMIZER_STATE,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model,"state":int(gamma_adam)},
            dtype=precision_policy.storage_opt_state_dtype,
            element_count=(d_model + d_model * beta_bias) * gamma_adam,
            hbm_occupy=(d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
        )
        layer_tensors.append(RMSNorm2_optimizer_state_tensor)

        RMSNorm2_param_gradient_tensor = tensor_spec(
            role=globle_tensor_role.PARAM_GRADIENT,
            layer_id=l,
            component_type=model_component.RMSNORM,
            shape={"scale":d_model,"bias":d_model},
            dtype=precision_policy.storage_grad_param_dtype,
            element_count=d_model + d_model * beta_bias,
            hbm_occupy=(d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
        )
        layer_tensors.append(RMSNorm2_param_gradient_tensor)

        RMSNorm2_input_gradient_tensor = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.RMSNORM, 
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(RMSNorm2_input_gradient_tensor)

        #Residual2
        RES2_activation_tensor = tensor_spec(
            role=globle_tensor_role.ACTIVATION,
            layer_id=l,
            component_type=model_component.RESIDUAL,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
        )
        layer_tensors.append(RES2_activation_tensor)

        RES2_input_gradient_tensor = tensor_spec(
            role=globle_tensor_role.INPUT_GRADIENT,
            layer_id=l,
            component_type=model_component.RESIDUAL,
            shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
            dtype=precision_policy.storage_grad_act_dtype,
            element_count=GBS * S * d_model,
            hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
        )
        layer_tensors.append(RES2_input_gradient_tensor)

        #处理Dense Layers的张量
        if l <= L_dense:
            #GATED_FFN_UP
            GATED_FFN_UP_param_tensor = tensor_spec(
                role=globle_tensor_role.PARAM,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"up":(d_model, dff_dense),"gate":(d_model, dff_dense),
                       "bias_up":(dff_dense,),"bias_gate":(dff_dense,)},
                dtype=precision_policy.storage_param_dtype,
                element_count=2 * d_model * dff_dense + 2 * dff_dense * beta_bias,
                hbm_occupy=(2 * d_model * dff_dense + 2 * dff_dense * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_param_tensor)

            GATED_FFN_UP_master_param_tensor = tensor_spec(
                role=globle_tensor_role.MASTER_PARAM,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"up":(d_model, dff_dense),"gate":(d_model, dff_dense),
                       "bias_up":(dff_dense,),"bias_gate":(dff_dense,)},
                dtype=precision_policy.storage_master_param_dtype,
                element_count=2 * d_model * dff_dense + 2 * dff_dense * beta_bias,
                hbm_occupy=(2 * d_model * dff_dense + 2 * dff_dense * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_master_param_tensor)

            GATED_FFN_UP_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_dense},
                dtype=precision_policy.storage_act_dtype,
                element_count=GBS * S * dff_dense * 2,  # up和gate的输出
                hbm_occupy=GBS * S * dff_dense * 2 * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_activation_tensor)

            GATED_FFN_UP_optimizer_state_tensor = tensor_spec(
                role=globle_tensor_role.OPTIMIZER_STATE,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"up":(d_model, dff_dense),"gate":(d_model, dff_dense),
                       "bias_up":(dff_dense,),"bias_gate":(dff_dense,),"state":int(gamma_adam)},
                dtype=precision_policy.storage_opt_state_dtype,
                element_count=(2 * d_model * dff_dense + 2 * dff_dense * beta_bias) * gamma_adam,
                hbm_occupy=(2 * d_model * dff_dense + 2 * dff_dense * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_optimizer_state_tensor)

            GATED_FFN_UP_param_gradient_tensor = tensor_spec(
                role=globle_tensor_role.PARAM_GRADIENT,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"up":(d_model, dff_dense),"gate":(d_model, dff_dense),
                       "bias_up":(dff_dense,),"bias_gate":(dff_dense,)},
                dtype=precision_policy.storage_grad_param_dtype,
                element_count=2 * d_model * dff_dense + 2 * dff_dense * beta_bias,
                hbm_occupy=(2 * d_model * dff_dense + 2 * dff_dense * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_param_gradient_tensor)

            GATED_FFN_UP_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.GATED_FFN_UP,
                shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=GBS * S * d_model,
                hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(GATED_FFN_UP_input_gradient_tensor)

            #GATED_FFN_NONLINEAR
            GATED_FFN_NONLINEAR_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.GATED_FFN_NONLINEAR,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_dense},
                dtype=precision_policy.storage_act_dtype,
                element_count=GBS * S * dff_dense,
                hbm_occupy=GBS * S * dff_dense * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(GATED_FFN_NONLINEAR_activation_tensor)

            GATED_FFN_NONLINEAR_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.GATED_FFN_NONLINEAR,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_dense * 2},  # up+gate的输入梯度
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=GBS * S * dff_dense * 2,
                hbm_occupy=GBS * S * dff_dense * 2 * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(GATED_FFN_NONLINEAR_input_gradient_tensor)

            #GATED_FFN_DOWN
            GATED_FFN_DOWN_param_tensor = tensor_spec(
                role=globle_tensor_role.PARAM,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"down":(dff_dense, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_param_dtype,
                element_count=dff_dense * d_model + d_model * beta_bias,
                hbm_occupy=(dff_dense * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_param_tensor)

            GATED_FFN_DOWN_master_param_tensor = tensor_spec(
                role=globle_tensor_role.MASTER_PARAM,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"down":(dff_dense, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_master_param_dtype,
                element_count=dff_dense * d_model + d_model * beta_bias,
                hbm_occupy=(dff_dense * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_master_param_tensor)

            GATED_FFN_DOWN_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
                dtype=precision_policy.storage_act_dtype,
                element_count=GBS * S * d_model,
                hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_activation_tensor)

            GATED_FFN_DOWN_optimizer_state_tensor = tensor_spec(
                role=globle_tensor_role.OPTIMIZER_STATE,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"down":(dff_dense, d_model),"bias_down":(d_model,),"state":int(gamma_adam)},
                dtype=precision_policy.storage_opt_state_dtype,
                element_count=(dff_dense * d_model + d_model * beta_bias) * gamma_adam,
                hbm_occupy=(dff_dense * d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_optimizer_state_tensor)

            GATED_FFN_DOWN_param_gradient_tensor = tensor_spec(
                role=globle_tensor_role.PARAM_GRADIENT,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"down":(dff_dense, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_grad_param_dtype,
                element_count=dff_dense * d_model + d_model * beta_bias,
                hbm_occupy=(dff_dense * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_param_gradient_tensor)

            GATED_FFN_DOWN_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.GATED_FFN_DOWN,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_dense},
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=GBS * S * dff_dense,
                hbm_occupy=GBS * S * dff_dense * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(GATED_FFN_DOWN_input_gradient_tensor)
        
        #处理MoE Layers的张量
        elif l > L_dense:
            #MOE_ROUTER
            MOE_ROUTER_param_tensor = tensor_spec(
                role=globle_tensor_role.PARAM,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"router":(d_model, E_total),"bias_router":(E_total,)},
                dtype=precision_policy.storage_param_dtype,
                element_count=d_model * E_total + E_total * beta_bias,
                hbm_occupy=(d_model * E_total + E_total * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
            )
            layer_tensors.append(MOE_ROUTER_param_tensor)

            MOE_ROUTER_master_param_tensor = tensor_spec(
                role=globle_tensor_role.MASTER_PARAM,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"router":(d_model, E_total),"bias_router":(E_total,)},
                dtype=precision_policy.storage_master_param_dtype,
                element_count=d_model * E_total + E_total * beta_bias,
                hbm_occupy=(d_model * E_total + E_total * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
            )
            layer_tensors.append(MOE_ROUTER_master_param_tensor)

            MOE_ROUTER_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"batch":GBS,"seq_len":S,"experts":E_total},
                dtype=precision_policy.storage_act_dtype,
                element_count=GBS * S * E_total,
                hbm_occupy=GBS * S * E_total * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(MOE_ROUTER_activation_tensor)

            MOE_ROUTER_optimizer_state_tensor = tensor_spec(
                role=globle_tensor_role.OPTIMIZER_STATE,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"router":(d_model, E_total),"bias_router":(E_total,),"state":int(gamma_adam)},
                dtype=precision_policy.storage_opt_state_dtype,
                element_count=(d_model * E_total + E_total * beta_bias) * gamma_adam,
                hbm_occupy=(d_model * E_total + E_total * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
            )
            layer_tensors.append(MOE_ROUTER_optimizer_state_tensor)

            MOE_ROUTER_param_gradient_tensor = tensor_spec(
                role=globle_tensor_role.PARAM_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"router":(d_model, E_total),"bias_router":(E_total,)},
                dtype=precision_policy.storage_grad_param_dtype,
                element_count=d_model * E_total + E_total * beta_bias,
                hbm_occupy=(d_model * E_total + E_total * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
            )
            layer_tensors.append(MOE_ROUTER_param_gradient_tensor)

            MOE_ROUTER_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_ROUTER,
                shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model},
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=GBS * S * d_model,
                hbm_occupy=GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(MOE_ROUTER_input_gradient_tensor)

            #MOE_EXPERT_FFN_UP
            MOE_EXPERT_FFN_UP_param_tensor = tensor_spec(
                role=globle_tensor_role.PARAM,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"up":(d_model, dff_moe),"gate":(d_model, dff_moe),
                       "bias_up":(dff_moe,),"bias_gate":(dff_moe,)},
                dtype=precision_policy.storage_param_dtype,
                element_count=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias),
                hbm_occupy=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_param_tensor)

            MOE_EXPERT_FFN_UP_master_param_tensor = tensor_spec(
                role=globle_tensor_role.MASTER_PARAM,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"up":(d_model, dff_moe),"gate":(d_model, dff_moe),
                       "bias_up":(dff_moe,),"bias_gate":(dff_moe,)},
                dtype=precision_policy.storage_master_param_dtype,
                element_count=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias),
                hbm_occupy=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_master_param_tensor)

            MOE_EXPERT_FFN_UP_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_moe,"experts":E_active},
                dtype=precision_policy.storage_act_dtype,
                element_count=E_active * GBS * S * dff_moe * 2,  # up+gate * 激活专家数
                hbm_occupy=E_active * GBS * S * dff_moe * 2 * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_activation_tensor)

            MOE_EXPERT_FFN_UP_optimizer_state_tensor = tensor_spec(
                role=globle_tensor_role.OPTIMIZER_STATE,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"up":(d_model, dff_moe),"gate":(d_model, dff_moe),
                       "bias_up":(dff_moe,),"bias_gate":(dff_moe,),"state":int(gamma_adam)},
                dtype=precision_policy.storage_opt_state_dtype,
                element_count=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias) * gamma_adam,
                hbm_occupy=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_optimizer_state_tensor)

            MOE_EXPERT_FFN_UP_param_gradient_tensor = tensor_spec(
                role=globle_tensor_role.PARAM_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"up":(d_model, dff_moe),"gate":(d_model, dff_moe),
                       "bias_up":(dff_moe,),"bias_gate":(dff_moe,)},
                dtype=precision_policy.storage_grad_param_dtype,
                element_count=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias),
                hbm_occupy=E_active * (2 * d_model * dff_moe + 2 * dff_moe * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_param_gradient_tensor)

            MOE_EXPERT_FFN_UP_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_UP,
                shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model,"experts":E_active},
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=E_active * GBS * S * d_model,
                hbm_occupy=E_active * GBS * S * d_model * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_UP_input_gradient_tensor)

            #MOE_EXPERT_FFN_NONLINEAR
            MOE_EXPERT_FFN_NONLINEAR_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_NONLINEAR,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_moe,"experts":E_active},
                dtype=precision_policy.storage_act_dtype,
                element_count=E_active * GBS * S * dff_moe,
                hbm_occupy=E_active * GBS * S * dff_moe * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_NONLINEAR_activation_tensor)

            MOE_EXPERT_FFN_NONLINEAR_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_NONLINEAR,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_moe * 2,"experts":E_active},  # up+gate * 激活专家数
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=E_active * GBS * S * dff_moe * 2,
                hbm_occupy=E_active * GBS * S * dff_moe * 2 * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_NONLINEAR_input_gradient_tensor)

            #MOE_EXPERT_FFN_DOWN
            MOE_EXPERT_FFN_DOWN_param_tensor = tensor_spec(
                role=globle_tensor_role.PARAM,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"down":(dff_moe, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_param_dtype,
                element_count=E_active * (dff_moe * d_model + d_model * beta_bias),
                hbm_occupy=E_active * (dff_moe * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_param_tensor)

            MOE_EXPERT_FFN_DOWN_master_param_tensor = tensor_spec(
                role=globle_tensor_role.MASTER_PARAM,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"down":(dff_moe, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_master_param_dtype,
                element_count=E_active * (dff_moe * d_model + d_model * beta_bias),
                hbm_occupy=E_active * (dff_moe * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_master_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_master_param_tensor)

            MOE_EXPERT_FFN_DOWN_activation_tensor = tensor_spec(
                role=globle_tensor_role.ACTIVATION,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"batch":GBS,"seq_len":S,"hidden_dim":d_model,"experts":E_active},
                dtype=precision_policy.storage_act_dtype,
                element_count=E_active * GBS * S * d_model,
                hbm_occupy=E_active * GBS * S * d_model * dtype_bytes[precision_policy.storage_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_activation_tensor)

            MOE_EXPERT_FFN_DOWN_optimizer_state_tensor = tensor_spec(
                role=globle_tensor_role.OPTIMIZER_STATE,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"down":(dff_moe, d_model),"bias_down":(d_model,),"state":int(gamma_adam)},
                dtype=precision_policy.storage_opt_state_dtype,
                element_count=E_active * (dff_moe * d_model + d_model * beta_bias) * gamma_adam,
                hbm_occupy=E_active * (dff_moe * d_model + d_model * beta_bias) * gamma_adam * dtype_bytes[precision_policy.storage_opt_state_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_optimizer_state_tensor)

            MOE_EXPERT_FFN_DOWN_param_gradient_tensor = tensor_spec(
                role=globle_tensor_role.PARAM_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"down":(dff_moe, d_model),"bias_down":(d_model,)},
                dtype=precision_policy.storage_grad_param_dtype,
                element_count=E_active * (dff_moe * d_model + d_model * beta_bias),
                hbm_occupy=E_active * (dff_moe * d_model + d_model * beta_bias) * dtype_bytes[precision_policy.storage_grad_param_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_param_gradient_tensor)

            MOE_EXPERT_FFN_DOWN_input_gradient_tensor = tensor_spec(
                role=globle_tensor_role.INPUT_GRADIENT,
                layer_id=l,
                component_type=model_component.MOE_EXPERT_FFN_DOWN,
                shape={"batch":GBS,"seq_len":S,"ffn_dim":dff_moe,"experts":E_active},
                dtype=precision_policy.storage_grad_act_dtype,
                element_count=E_active * GBS * S * dff_moe,
                hbm_occupy=E_active * GBS * S * dff_moe * dtype_bytes[precision_policy.storage_grad_act_dtype],
            )
            layer_tensors.append(MOE_EXPERT_FFN_DOWN_input_gradient_tensor)

        #添加当前层张量到模型张量列表
        model_tensors.append(model_layer_tensor_spec(
            layer_id=l,
            tensors=layer_tensors
        ))

    #返回完整的模型张量图
    return model_tensor_graph(layers=model_tensors)
