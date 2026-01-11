from typing import Dict, List, Union
from pydantic import BaseModel
from Precision_Policy import get_policy
from User_Config import ParallelConfig, ModelConfig, TrainConfig, PrecisionConfig, GPU_Resource      

dtype_bytes = {"FP8": 1, "BF16": 2, "FP16": 2, "FP32": 4}
class parallel_rank(BaseModel):
    rank_idx: Dict[str, int]

class runtime_rank_spec(BaseModel):
    rank_location: parallel_rank
    pipeline_stage: int
    model_layers_span: List[int]
    components_collectives_fwd: List[Dict[str, Dict[str, Union[str, int, float]]]]
    components_collectives_bwd: List[Dict[str, Dict[str, Union[str, int, float]]]]
    grad_update_collective: List[Dict[str, Dict[str, Union[str, int, float]]]]
    pp_stage_collective: List[Dict[str, Dict[str, Union[str, int, float]]]]

#-------------------根据用户输入构建parallel_rank_specs-------------------
def build_runtime_rank_graph(
    parallel_config: ParallelConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    precision_config: PrecisionConfig,
    gpu_config: GPU_Resource,
) -> List[runtime_rank_spec]:
    #根据用户配置输入变量
    GBS = train_config.global_batch_size
    S = train_config.seq_len
    dp_size = parallel_config.dp_size
    pp_size = parallel_config.pp_size
    ep_size = parallel_config.ep_size
    tp_size = parallel_config.tp_size
    d_model = model_config.hidden_size
    dff_dense = model_config.dense_intermediate_size
    dff_moe = model_config.moe_intermediate_size
    L = model_config.num_hidden_layers
    L_dense = model_config.num_dense_hidden_layers
    E_router = model_config.n_routed_experts
    E_shared = model_config.n_shared_experts
    E_active = model_config.num_experts_per_tok
    M = train_config.num_micro_batches
    precision = precision_config.policy_name
    layer_span = L // pp_size
    B_local = GBS / dp_size / M / ep_size   # 每个并行rank的local batch size
    N_token_local = B_local * S                # 每个并行rank的local token数
    num_gpus = gpu_config.num_gpus

    #验证并行配置合法性
    total_parallel_size = dp_size * pp_size * tp_size * ep_size
    if total_parallel_size != num_gpus:
        raise ValueError(f"总并行切分Size {total_parallel_size} 不匹配GPU数量 {num_gpus}.")

    # 实现逻辑：根据并行配置、模型配置、训练配置、精度配置、GPU配置，构建并行rank规格列表
    #V1版本核心工程假设：
    #1 同一EP RANK下的TP RANK计算时独立复制EP rank输入的local_batch,TP域不做batch的切片，避免引入过于复杂的TP域内通信
    #2 同一PP RANK下的EP RANK计算时分片PP rank输入的local_batch,避免重复计算
    #3 DP域间模型复制副本，GBS切分
    #4 组件与层的边界激活值统一维持（B_local,S,H)张量shape，减少不必要通信量

    #初始化ranks属性列表
    runtime_ranks: List[runtime_rank_spec] = []
    #遍历所有并行rank，逐个构建runtime_rank_spec
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            for ep_rank in range(ep_size):
                for tp_rank in range(tp_size):
                    
                    #初始化集合通信属性
                    components_collectives_fwd: List[Dict[str, Dict[str, str | int | float]]] = []
                    components_collectives_bwd: List[Dict[str, Dict[str, str | int | float]]] = []
                    grad_update_collective: List[Dict[str, Dict[str, str | int | float]]] = []
                    pp_stage_collective: List[Dict[str, Dict[str, str | int | float]]] = [] 

                    # 计算当前并行rank的模型层归属
                    if pp_rank == pp_size - 1:
                        start_layer = pp_rank * layer_span+ 1
                        end_layer = L
                    else:
                        start_layer = pp_rank * layer_span+ 1
                        end_layer = start_layer + layer_span-1
                    model_layers_span = list(range(start_layer, end_layer+1))

                    #计算PP-STAGE属性
                    pipeline_stage = pp_rank
                    #计算rank_location
                    rank_location = parallel_rank(rank_idx={
                        "dp_idx": dp_rank,
                        "pp_idx": pp_rank,
                        "ep_idx": ep_rank,
                        "tp_idx": tp_rank,
                    })

                    #计算pp_stage集合通信属性
                    N_PP_boundary_FWD= N_token_local * d_model
                    V_PP_boundary_FWD = 2 * N_PP_boundary_FWD
                    V_PP_boundary_BWD = V_PP_boundary_FWD
                    PP_boundary_DTYPE = get_policy(precision).comm_act_dtype
                    PP_boundary_BYTES = dtype_bytes[PP_boundary_DTYPE]
                    V_PP_boundary_FWD_BYTES = V_PP_boundary_FWD * PP_boundary_BYTES
                    V_PP_boundary_BWD_BYTES = V_PP_boundary_BWD * PP_boundary_BYTES
                    if pipeline_stage == 0:
                        pp_boundary_FWD_collective = {
                            "PP_boundary_FWD": {
                                "pp_rank": pp_rank,
                                "collective_type": "send",
                                "collective_group": "PP",
                                "collective_elements": N_PP_boundary_FWD,
                                "collective-throughput": V_PP_boundary_FWD_BYTES,
                                "collective-dtype": PP_boundary_DTYPE,
                            }}
                        pp_stage_collective.append(pp_boundary_FWD_collective)
                        pp_boundary_BWD_collective = {
                            "PP_boundary_BWD": {
                                "pp_rank": pp_rank,
                                "collective_type": "recv",
                                "collective_group": "PP",
                                "collective_elements": N_PP_boundary_FWD,
                                "collective-throughput": V_PP_boundary_BWD_BYTES,
                                "collective-dtype": PP_boundary_DTYPE,
                            }}
                        pp_stage_collective.append(pp_boundary_BWD_collective)
                    elif pipeline_stage < pp_size and pipeline_stage > 0:
                            pp_boundary_FWD_collective = {
                                "PP_boundary_FWD": {
                                "pp_rank": pp_rank,
                                "collective_type": "send&recv",
                                "collective_group": "PP",
                                "collective_elements": 2* (N_PP_boundary_FWD),
                                "collective-throughput": 2 * (V_PP_boundary_FWD_BYTES),
                                "collective-dtype": PP_boundary_DTYPE,
                                }}
                            pp_stage_collective.append(pp_boundary_FWD_collective)

                            pp_boundary_BWD_collective = {
                                "PP_boundary_BWD": {
                                "pp_rank": pp_rank,
                                "collective_type": "send&recv",
                                "collective_group": "PP",
                                "collective_elements": 2* (N_PP_boundary_FWD),
                                "collective-throughput": 2 * (V_PP_boundary_BWD_BYTES),
                                "collective-dtype": PP_boundary_DTYPE,
                                }}
                            pp_stage_collective.append(pp_boundary_BWD_collective)
                    else:
                        pp_boundary_FWD_collective = {
                            "PP_boundary_FWD": {
                            "pp_rank": pp_rank,
                            "collective_type": "recv",
                            "collective_group": "PP",
                            "collective_elements": N_PP_boundary_FWD,
                            "collective-throughput": V_PP_boundary_FWD_BYTES,
                            "collective-dtype": PP_boundary_DTYPE,
                            }}
                        pp_stage_collective.append(pp_boundary_FWD_collective)
                        pp_boundary_BWD_collective = {
                            "PP_boundary_BWD": {
                            "pp_rank": pp_rank,
                            "collective_type": "send",
                            "collective_group": "PP",
                            "collective_elements": N_PP_boundary_FWD,
                            "collective-throughput": V_PP_boundary_BWD_BYTES,
                            "collective-dtype": PP_boundary_DTYPE,
                            }}
                        pp_stage_collective.append(pp_boundary_BWD_collective)

                    # 计算当前并行rank的组件通信
                    for layer in model_layers_span:
                        #每一层的通用组件-前向过程
                        #MHA-out-Fwd
                        N_MHA_OUT_FWD = N_token_local * (d_model / tp_size)
                        V_MHA_OUT_FWD = 2 * N_MHA_OUT_FWD * (tp_size-1)/tp_size
                        MHA_OUT_dtype = get_policy(precision).comm_act_dtype
                        MHA_OUT_bytes = dtype_bytes[MHA_OUT_dtype]
                        V_MHA_OUT_FWD_BYTES = V_MHA_OUT_FWD * MHA_OUT_bytes
                        mha_out_fwd_collective = {
                            "mha_out_fwd": {
                            "layer": layer,
                            "collective_type": "allreduce",
                            "collective_group": "TP",
                            "collective_elements": N_MHA_OUT_FWD,
                            "collective-throughput": V_MHA_OUT_FWD_BYTES,
                            "collective-dtype": MHA_OUT_dtype,
                            }}
                        components_collectives_fwd.append(mha_out_fwd_collective)

                        #MHA-QKV-FWD
                        mha_qkv_fwd_collective = {
                            "mha_qkv_fwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_fwd.append(mha_qkv_fwd_collective)
                        
                        #MHA-CORE-FWD
                        mha_core_fwd_collective = {
                            "mha_core_fwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_fwd.append(mha_core_fwd_collective)

                        #Res1-FWD
                        res1_fwd_collective = {
                            "res1_fwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_fwd.append(res1_fwd_collective)

                        #RMSNorm2-FWD
                        rmsn2_fwd_collective = {
                            "rmsn2_fwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_fwd.append(rmsn2_fwd_collective)

                        #Res2-FWD
                        res2_fwd_collective = {
                            "res2_fwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_fwd.append(res2_fwd_collective)

                        #每一层的通用组件-反向过程
                        #MHA QKV-BWD
                        N_MHA_QKV_BWD = N_token_local * d_model
                        V_MHA_QKV_BWD = 2 * N_MHA_QKV_BWD * (tp_size-1)/tp_size
                        MHA_QKV_dtype = get_policy(precision).comm_act_dtype
                        MHA_QKV_bytes = dtype_bytes[MHA_QKV_dtype]
                        V_MHA_QKV_BWD_BYTES = V_MHA_QKV_BWD * MHA_QKV_bytes
                        mha_qkv_bwd_collective = {
                            "mha_qkv_bwd": {
                            "layer": layer,
                            "collective_type": "allreduce",
                            "collective_group": "TP",
                            "collective_elements": N_MHA_QKV_BWD,
                            "collective-throughput": V_MHA_QKV_BWD_BYTES,
                            "collective-dtype": MHA_QKV_dtype,
                            }}
                        components_collectives_bwd.append(mha_qkv_bwd_collective)

                        #MHA-CORE-BWD
                        mha_core_bwd_collective = {
                            "mha_core_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(mha_core_bwd_collective)

                        #MHA-OUT-BWD
                        mha_out_bwd_collective = {
                            "mha_out_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(mha_out_bwd_collective)

                        #Res2-BWD
                        res2_bwd_collective = {
                            "res2_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(res2_bwd_collective)

                        #Res1-BWD
                        res1_bwd_collective = {
                            "res1_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(res1_bwd_collective)

                        #RMSNorm2-BWD
                        rmsn2_bwd_collective = {
                            "rmsn2_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(rmsn2_bwd_collective)

                        #RMSNorm1-BWD
                        rmsn1_bwd_collective = {
                            "rmsn1_bwd": {
                            "layer": layer,
                            "collective_type": "none",
                            }}
                        components_collectives_bwd.append(rmsn1_bwd_collective)

                        #处理反向梯度同步
                        #MHA-QKV-GRAD-BWD
                        COPY_FIELD = dp_size
                        N_MHA_QKV_GRAD = d_model * 3 * d_model / tp_size
                        V_MHA_QKV_GRAD = 2 * N_MHA_QKV_GRAD * (COPY_FIELD-1) / COPY_FIELD
                        MHA_QKV_GRAD_dtype = get_policy(precision).comm_grad_param_dtype
                        MHA_QKV_GRAD_bytes = dtype_bytes[MHA_QKV_GRAD_dtype]
                        V_MHA_QKV_GRAD_BYTES = V_MHA_QKV_GRAD * MHA_QKV_GRAD_bytes 
                        MHA_QKV_GRAD = {
                            "MHA_QKV":{
                            "layer": layer,
                            "collective_type": "allreduce",
                            "collective_group": f"DP={COPY_FIELD}",
                            "collective_elements": N_MHA_QKV_GRAD,
                            "collective-throughput": V_MHA_QKV_GRAD_BYTES,
                            "collective-dtype": MHA_QKV_GRAD_dtype,
                            }}
                        grad_update_collective.append(MHA_QKV_GRAD)

                        #MHA-OUT-GRAD-BWD
                        COPY_FIELD = dp_size
                        N_MHA_OUT_GRAD = d_model * d_model / tp_size
                        V_MHA_OUT_GRAD = 2 * N_MHA_OUT_GRAD * (COPY_FIELD-1) / COPY_FIELD
                        MHA_OUT_GRAD_dtype = get_policy(precision).comm_grad_param_dtype
                        MHA_OUT_GRAD_bytes = dtype_bytes[MHA_OUT_GRAD_dtype]
                        V_MHA_OUT_GRAD_BYTES = V_MHA_OUT_GRAD * MHA_OUT_GRAD_bytes 
                        MHA_OUT_GRAD = {
                            "MHA_OUT":{
                            "layer": layer,
                            "collective_type": "allreduce",
                            "collective_group": f"DP={COPY_FIELD}",
                            "collective_elements": N_MHA_OUT_GRAD,
                            "collective-throughput": V_MHA_OUT_GRAD_BYTES,
                            "collective-dtype": MHA_OUT_GRAD_dtype,
                            }}
                        grad_update_collective.append(MHA_OUT_GRAD)

                        #RMSNORM-GRAD-BWD
                        COPY_FIELD = dp_size
                        N_RMSNORM_GRAD = 2 * d_model
                        V_RMSNORM_GRAD = 2 * N_RMSNORM_GRAD * (COPY_FIELD-1) / COPY_FIELD
                        RMSNORM_GRAD_dtype = get_policy(precision).comm_grad_param_dtype
                        RMSNORM_GRAD_bytes = dtype_bytes[RMSNORM_GRAD_dtype]
                        V_RMSNORM_GRAD_BYTES = V_RMSNORM_GRAD * RMSNORM_GRAD_bytes 
                        RMSNORM_GRAD = {
                            "RMSNORM_GRAD":{
                            "layer": layer,
                            "collective_type": "allreduce",
                            "collective_group": f"DP={COPY_FIELD}",
                            "collective_elements": N_RMSNORM_GRAD,
                            "collective-throughput": V_RMSNORM_GRAD_BYTES,
                            "collective-dtype": RMSNORM_GRAD_dtype,
                            }}
                        grad_update_collective.append(RMSNORM_GRAD)

                        #处理MoE层组件通信
                        if layer > L_dense and layer <= L:
                        #存在集合通信的组件
                            #MoE Dispatch-Fwd
                            N_MOE_DISPATCH_FWD = N_token_local * E_active * d_model 
                            V_MOE_DISPATCH_FWD = 2 * N_MOE_DISPATCH_FWD * (ep_size-1)/ep_size
                            MOE_DISPATCH_dtype = get_policy(precision).comm_act_dtype
                            MOE_DISPATCH_bytes = dtype_bytes[MOE_DISPATCH_dtype]
                            V_MOE_DISPATCH_FWD_BYTES = V_MOE_DISPATCH_FWD * MOE_DISPATCH_bytes
                            moe_dispatch_fwd_collective = {
                                        "moe_dispatch_fwd": {
                                        "layer": layer,
                                        "collective_type": "all2all",
                                        "collective_group": "EP",
                                        "collective_elements": N_MOE_DISPATCH_FWD,
                                        "collective-throughput": V_MOE_DISPATCH_FWD_BYTES,
                                        "collective-dtype": MOE_DISPATCH_dtype,
                                        }}
                            components_collectives_fwd.append(moe_dispatch_fwd_collective)

                            #MoE FFN Down-Fwd
                            N_FFN_W2_FWD_MOE = N_token_local * d_model / tp_size
                            V_FFN_W2_FWD_MOE = 2 * N_FFN_W2_FWD_MOE * (tp_size-1)/tp_size
                            FFN_W2_dtype = get_policy(precision).comm_act_dtype
                            FFN_W2_bytes = dtype_bytes[FFN_W2_dtype]
                            V_FFN_W2_FWD_MOE_BYTES = V_FFN_W2_FWD_MOE * FFN_W2_bytes
                            moe_ffn_down_fwd_collective = {
                                        "moe_ffn_DOWN_fwd": {
                                        "layer": layer,
                                        "collective_type": "allreduce",
                                        "collective_group": "TP",
                                        "collective_elements": N_FFN_W2_FWD_MOE,
                                        "collective-throughput": V_FFN_W2_FWD_MOE_BYTES,
                                        "collective-dtype": FFN_W2_dtype,
                                        }}
                            components_collectives_fwd.append(moe_ffn_down_fwd_collective)

                            #MoE Gather-FWD
                            N_MOE_GATHER_FWD = N_token_local * E_active * d_model 
                            V_MOE_GATHER_FWD = 2 * N_MOE_GATHER_FWD * (ep_size-1)/ep_size
                            MOE_GATHER_dtype = get_policy(precision).comm_act_dtype
                            MOE_GATHER_bytes = dtype_bytes[MOE_GATHER_dtype]
                            V_MOE_GATHER_FWD_BYTES = V_MOE_GATHER_FWD * MOE_GATHER_bytes
                            moe_gather_fwd_collective = {
                                        "moe_gather_fwd": {
                                        "layer": layer,
                                        "collective_type": "all2all",
                                        "collective_group": "EP",
                                        "collective_elements": N_MOE_GATHER_FWD,
                                        "collective-throughput": V_MOE_GATHER_FWD_BYTES,
                                        "collective-dtype": MOE_GATHER_dtype,
                                        }}
                            components_collectives_fwd.append(moe_gather_fwd_collective)

                            #反向传播的集合通信
                            #MoE Gather-BWD
                            N_MOE_GATHER_BWD = N_token_local * E_active * d_model 
                            V_MOE_GATHER_BWD = 2 * N_MOE_GATHER_BWD * (ep_size-1)/ep_size
                            MOE_GATHER_dtype = get_policy(precision).comm_act_dtype
                            MOE_GATHER_bytes = dtype_bytes[MOE_GATHER_dtype]
                            V_MOE_GATHER_BWD_BYTES = V_MOE_GATHER_BWD * MOE_GATHER_bytes
                            moe_gather_bwd_collective = {
                                            "moe_gather_bwd": {
                                            "layer": layer,
                                            "collective_type": "all2all",
                                            "collective_group": "EP",
                                            "collective_elements": N_MOE_GATHER_BWD,
                                            "collective-throughput": V_MOE_GATHER_BWD_BYTES,
                                            "collective-dtype": MOE_GATHER_dtype,
                                            }}
                            components_collectives_bwd.append(moe_gather_bwd_collective)

                            #MoE FFN Up-BWD
                            N_FFN_W2_BWD_MOE = N_token_local * d_model / tp_size
                            V_FFN_W2_BWD_MOE = 2 * N_FFN_W2_BWD_MOE * (tp_size-1)/tp_size
                            FFN_W2_dtype = get_policy(precision).comm_act_dtype
                            FFN_W2_bytes = dtype_bytes[FFN_W2_dtype]
                            V_FFN_W2_BWD_MOE_BYTES = V_FFN_W2_BWD_MOE * FFN_W2_bytes
                            moe_ffn_up_bwd_collective = {
                                            "moe_ffn_UP_bwd": {
                                            "layer": layer,
                                            "collective_type": "allreduce",
                                            "collective_group": "TP",
                                            "collective_elements": N_FFN_W2_BWD_MOE,
                                            "collective-throughput": V_FFN_W2_BWD_MOE_BYTES,
                                            "collective-dtype": FFN_W2_dtype,
                                            }}
                            components_collectives_bwd.append(moe_ffn_up_bwd_collective)
                            
                            #MoE Dispatch-BWD
                            N_MOE_DISPATCH_BWD = N_token_local * E_active * d_model 
                            V_MOE_DISPATCH_BWD = 2 * N_MOE_DISPATCH_BWD * (ep_size-1)/ep_size
                            MOE_DISPATCH_dtype = get_policy(precision).comm_act_dtype
                            MOE_DISPATCH_bytes = dtype_bytes[MOE_DISPATCH_dtype]
                            V_MOE_DISPATCH_BWD_BYTES = V_MOE_DISPATCH_BWD * MOE_DISPATCH_bytes
                            moe_dispatch_bwd_collective = {
                                            "moe_dispatch_bwd": {
                                            "layer": layer,
                                            "collective_type": "all2all",
                                            "collective_group": "EP",
                                            "collective_elements": N_MOE_DISPATCH_BWD,
                                            "collective-throughput": V_MOE_DISPATCH_BWD_BYTES,
                                            "collective-dtype": MOE_DISPATCH_dtype,
                                            }}
                            components_collectives_bwd.append(moe_dispatch_bwd_collective)

                            #MoE Router-FWD
                            moe_router_fwd_collective = {
                                "moe_router_fwd": {
                                "layer": layer,
                                "collective_type": "none",
                                }}
                            components_collectives_fwd.append(moe_router_fwd_collective)

                            #MoE Router-BWD
                            moe_router_bwd_collective = {
                                "moe_router_bwd": {
                                "layer": layer,
                                "collective_type": "none",
                                }}
                            components_collectives_bwd.append(moe_router_bwd_collective)

                            #MoE FFN Down-BWD
                            moe_ffn_down_bwd_collective = {
                                "moe_ffn_DOWN_bwd": {
                                "layer": layer,
                                "collective_type": "none",
                                }}
                            components_collectives_bwd.append(moe_ffn_down_bwd_collective)

                            #MoE FFN Up-FWD
                            moe_ffn_up_fwd_collective = {
                                "moe_ffn_UP_fwd": {
                                "layer": layer,
                                "collective_type": "none",
                                }}
                            components_collectives_fwd.append(moe_ffn_up_fwd_collective)

                            #MoE EXPERT FFN-UP GRAD
                            # 计算每个 rank 持有的 expert 数（不改 EP 切 token 逻辑）
                            experts_per_rank = max(1, (E_router + E_shared) // ep_size)
                            # 每个 expert 的参数量：d_model × dff_moe + dff_moe × d_model
                            # 注意：不乘 token，因为梯度张量形状 = 参数形状
                            N_param_per_expert = (d_model * dff_moe + dff_moe * d_model) / tp_size
                            # rank 上的总梯度元素数
                            N_grad_moe_expert = experts_per_rank * N_param_per_expert
                            COPY_FIELD = dp_size
                            V_FFN_GRAD_MOE = 2 * N_grad_moe_expert * (COPY_FIELD - 1) / COPY_FIELD
                            FFN_GRAD_MOE_dtype = get_policy(precision).comm_grad_param_dtype
                            FFN_GRAD_MOE_bytes = dtype_bytes[FFN_GRAD_MOE_dtype]
                            V_FFN_GRAD_MOE_BYTES = V_FFN_GRAD_MOE * FFN_GRAD_MOE_bytes
                            moe_ffn_grad_collective = {
                                            "moe_ffn_UP_grad": {
                                            "layer": layer,
                                            "collective_type": "allreduce",
                                            "collective_group": "DP",
                                            "collective_elements": N_grad_moe_expert,
                                            "collective-throughput": V_FFN_GRAD_MOE_BYTES,
                                            "collective-dtype": FFN_GRAD_MOE_dtype,
                                            }}
                            grad_update_collective.append(moe_ffn_grad_collective)

                            #MoE Router Grad通信量d*E_router，暂不考虑，避免过于复杂的整体调整

                            #处理最后一层的logits和loss组件
                            if layer == L:
                                #Logits
                                logits_collective = {
                                    "logits": {
                                    "layer": layer,
                                    "collective_type": "none",
                                    }}
                                components_collectives_fwd.append(logits_collective)
                                components_collectives_bwd.append(logits_collective)
                                #Loss
                                loss_collective = {
                                    "loss": {
                                    "layer": layer,
                                    "collective_type": "none",
                                    }}
                                components_collectives_fwd.append(loss_collective)
                                components_collectives_bwd.append(loss_collective)


                        #处理特殊Dense层组件通信
                        elif layer <= L_dense:
                            #Dense FFN Down-FWD
                            N_FFN_W2_FWD_DENSE = N_token_local * d_model
                            V_FFN_W2_FWD_DENSE = 2 * N_FFN_W2_FWD_DENSE * (tp_size-1)/tp_size
                            FFN_W2_dtype = get_policy(precision).comm_act_dtype
                            FFN_W2_bytes = dtype_bytes[FFN_W2_dtype]
                            V_FFN_W2_FWD_DENSE_BYTES = V_FFN_W2_FWD_DENSE * FFN_W2_bytes
                            dense_ffn_down_fwd_collective = {
                                "dense_ffn_DOWN_FWD": {
                                "layer": layer,
                                "collective_type": "allreduce",
                                "collective_group": "TP",
                                "collective_elements": N_FFN_W2_FWD_DENSE,
                                "collective-throughput": V_FFN_W2_FWD_DENSE_BYTES,
                                "collective-dtype": FFN_W2_dtype,
                                }}
                            components_collectives_fwd.append(dense_ffn_down_fwd_collective)
                            
                            #Dense FFN Up-BWD
                            N_FFN_W1_BWD_DENSE = N_token_local * d_model
                            V_FFN_W1_BWD_DENSE = 2 * N_FFN_W1_BWD_DENSE * (tp_size-1)/tp_size
                            FFN_W1_dtype = get_policy(precision).comm_act_dtype
                            FFN_W1_bytes = dtype_bytes[FFN_W1_dtype]
                            V_FFN_W1_BWD_DENSE_BYTES = V_FFN_W1_BWD_DENSE * FFN_W1_bytes
                            dense_ffn_up_bwd_collective = {
                                "dense_ffn_UP_BWD": {
                                "layer": layer,
                                "collective_type": "allreduce",
                                "collective_group": "TP",
                                "collective_elements": N_FFN_W1_BWD_DENSE,
                                "collective-throughput": V_FFN_W1_BWD_DENSE_BYTES,
                                "collective-dtype": FFN_W1_dtype,
                            }}
                            components_collectives_bwd.append(dense_ffn_up_bwd_collective)

                            #Dense FFN NONLINEAR
                            dense_ffn_nonlinear_collective = {
                                "dense_ffn_nonlinear": {
                                "layer": layer,
                                "collective_type": "none",
                                }}
                            components_collectives_fwd.append(dense_ffn_nonlinear_collective)
                            components_collectives_bwd.append(dense_ffn_nonlinear_collective)

                            #Dense FFN UP GRAD
                            COPY_FIELD = dp_size
                            N_FFN_UP_GRAD_DENSE = dff_dense * d_model / tp_size
                            V_FFN_UP_GRAD_DENSE = 2 * N_FFN_UP_GRAD_DENSE * (COPY_FIELD-1)/COPY_FIELD
                            FFN_UP_GRAD_DENSE_dtype = get_policy(precision).comm_grad_param_dtype
                            FFN_UP_GRAD_DENSE_bytes = dtype_bytes[FFN_UP_GRAD_DENSE_dtype]
                            V_FFN_UP_GRAD_DENSE_BYTES = V_FFN_UP_GRAD_DENSE * FFN_UP_GRAD_DENSE_bytes   
                            dense_ffn_up_grad_collective = {
                                "dense_ffn_UP_GRAD": {
                                "layer": layer,
                                "collective_type": "allreduce",
                                "collective_group": f"DP={COPY_FIELD}",
                                "collective_elements": N_FFN_UP_GRAD_DENSE,
                                "collective-throughput": V_FFN_UP_GRAD_DENSE_BYTES,
                                "collective-dtype": FFN_UP_GRAD_DENSE_dtype,
                                }}
                            grad_update_collective.append(dense_ffn_up_grad_collective)

                            #Dense FFN DOWN GRAD
                            N_FFN_DOWN_GRAD_DENSE = dff_dense * d_model / tp_size
                            V_FFN_DOWN_GRAD_DENSE = 2 * N_FFN_DOWN_GRAD_DENSE * (COPY_FIELD-1)/COPY_FIELD
                            FFN_DOWN_GRAD_DENSE_dtype = get_policy(precision).comm_grad_param_dtype
                            FFN_DOWN_GRAD_DENSE_bytes = dtype_bytes[FFN_DOWN_GRAD_DENSE_dtype]
                            V_FFN_DOWN_GRAD_DENSE_BYTES = V_FFN_DOWN_GRAD_DENSE * FFN_DOWN_GRAD_DENSE_bytes   
                            dense_ffn_down_grad_collective = {
                                "dense_ffn_DOWN_GRAD": {
                                "layer": layer,
                                "collective_type": "allreduce",
                                "collective_group": f"DP={COPY_FIELD}",
                                "collective_elements": N_FFN_DOWN_GRAD_DENSE,
                                "collective-throughput": V_FFN_DOWN_GRAD_DENSE_BYTES,
                                "collective-dtype": FFN_DOWN_GRAD_DENSE_dtype,
                                }}
                            grad_update_collective.append(dense_ffn_down_grad_collective)

                            #处理第一层的初始化嵌入组件
                            if layer == 1:
                            #Embedding
                                emb_collective = {
                                    "embedding": {
                                    "layer": layer,
                                    "collective_type": "none"
                                    }}
                                components_collectives_fwd.append(emb_collective)
                                components_collectives_bwd.append(emb_collective)
                                #Position Embedding
                                pos_emb_collective = {
                                    "position_embedding": {
                                    "layer": layer,
                                    "collective_type": "none"
                                    }}
                                components_collectives_fwd.append(pos_emb_collective)
                                components_collectives_bwd.append(pos_emb_collective)
                    
                    #创建该rank坐标下的runtime_rank_spec
                    current_rank_spec = runtime_rank_spec(
                        rank_location=rank_location,
                        pipeline_stage=pipeline_stage,
                        model_layers_span=model_layers_span,
                        components_collectives_fwd=components_collectives_fwd,
                        components_collectives_bwd=components_collectives_bwd,
                        grad_update_collective=grad_update_collective,
                        pp_stage_collective=pp_stage_collective,
                    )
                    #添加该rank到runtime_ranks
                    runtime_ranks.append(current_rank_spec)
    
    #返回最终的runtime_ranks
    return runtime_ranks


                            

                        
                                    

                                    
