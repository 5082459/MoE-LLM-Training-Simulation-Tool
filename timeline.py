"""
Resource timeline simulation for a single training step.

This module aims to be an *engineering-style* lower-bound simulator:
- It tracks multiple resource "lanes": TensorCore, CUDA core, HBM, ScaleUp fabric, ScaleOut fabric.
- It simulates **per-microbatch** compute (forward then backward) for a given rank's pipeline stage.
- It schedules **per-microbatch** collectives (TP/EP/PP, activation-style comm) with simple data-ready
  dependencies derived from layer end-times.
- It separately estimates **per-step** gradient synchronization collectives (parameter/grad allreduces),
  which typically happen once after gradient accumulation, not once per microbatch.

Key design choices (pragmatic + explicit):
1) Compute graph comes from `parse_model_compute_spec()` which is built at GLOBAL batch-size.
   We scale compute/HBM to per-rank per-microbatch using:
      batch_scale = 1 / (dp * ep * num_micro_batches)
   and a component-dependent TP sharding rule.
2) Collective payload sizes coming from `build_runtime_rank_graph()` are assumed to be already
   "per-rank" quantities; we DO NOT re-scale them by dp/ep/tp/M except for per-step grad sync,
   which we keep separate.
3) Fabric selection (ScaleUp vs ScaleOut) uses `profile.scaleup_boundary` and an inferred
   collective group size; this avoids treating all "DP" as ScaleOut when DP is intra-node.

The result is still an approximation, but it avoids the big systematic under/over-counting traps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math

from Compute_Graph_fork4 import model_component_compute_spec, model_compute_graph, parse_model_compute_spec
from GPU_Profiler_fork4 import GPUHwProfile, get_gpu_profile
from Parallel_fork4 import build_runtime_rank_graph, runtime_rank_spec
from Precision_Policy_fork4 import get_policy
from User_Config_fork4 import GPU_Resource, ParallelConfig, SimulationInput

# Dtype sets for bucket assignment
TENSOR_CORE_DTYPES = {"fp8", "bf16", "fp16", "tf32"}
CUDA_CORE_DTYPES = {"fp32", "fp64"}


@dataclass
class RankTimeline:
    # Per-microbatch (one FWD + one BWD) lane end times (seconds)
    compute_tensor_s: float
    compute_cuda_s: float
    hbm_s: float
    scaleup_s: float
    scaleout_s: float

    # Per-microbatch comm volumes (bytes)
    comm_scaleup_bytes: float
    comm_scaleout_bytes: float

    # Per-microbatch stage time (seconds)
    step_s: float

    # Per-step grad-sync (after grad accumulation) time/bytes
    grad_update_s: float = 0.0
    grad_update_scaleup_bytes: float = 0.0
    grad_update_scaleout_bytes: float = 0.0

    bottleneck: str = "unknown"


# --------------------------
# Heuristic performance model
# --------------------------
def _kernel_factor(comp_type: str) -> float:
    """Crude factor for kernel efficiency (launch/bubbles/etc.)."""
    upper = comp_type.upper()
    if "MHA" in upper:
        return 0.7
    if "FFN" in upper:
        return 0.7
    if "NORM" in upper or "SOFTMAX" in upper or "RESIDUAL" in upper:
        return 0.5
    if "EMBED" in upper or "LOGITS" in upper:
        return 0.6
    return 0.8


def _component_efficiency(
    comp: model_component_compute_spec,
    peak_flops: float,
    hbm_bw: float,
    batch_local: float,
    eff_min: float = 0.25,
    batch_ref: float = 256.0,
) -> float:
    """
    Effective utilization model: roofline-like + small batch penalty + kernel-type penalty.
    """
    flops = float(sum(comp.fwd_flops.values()) + sum(comp.bwd_flops.values()))
    hbm_bytes = float(sum(comp.fwd_hbm_throughput.values()) + sum(comp.bwd_hbm_throughput.values()))
    if peak_flops <= 0:
        return eff_min

    if hbm_bytes > 0 and hbm_bw > 0:
        eff_roof = min(1.0, (flops / hbm_bytes) * (hbm_bw / peak_flops))
    else:
        eff_roof = 1.0

    factor_batch = min(1.0, batch_local / batch_ref) if batch_ref > 0 else 1.0
    factor_kernel = _kernel_factor(comp.component_type)
    eff = eff_roof * factor_batch * factor_kernel
    return max(eff, eff_min)


def _main_dtype_from_flops(flops_map: Dict[str, float]) -> str:
    if flops_map:
        dtype = max(flops_map.items(), key=lambda kv: kv[1])[0]
        return str(dtype).lower()
    return "fp32"


def _tp_scale_for_component(component_type: str, tp_size: int) -> float:
    """
    Component-dependent TP scaling.
    You can refine this mapping as your TP definition becomes more precise.
    """
    tp = max(tp_size, 1)
    ct = component_type.lower()

    # Default assumption in this project: hidden/head dims are TP-sharded,
    # so most compute/HBM scale ~1/tp.
    # If later you model certain ops as replicated, move them to the return 1.0 bucket.
    replicated = {
        # examples to toggle in the future:
        # "tokenizer",
    }
    if ct in replicated:
        return 1.0
    return 1.0 / tp


def _infer_collective_group_size(group: str, parallel: ParallelConfig, gpu: GPU_Resource) -> int:
    """
    Parse/estimate collective group size from collective_group string.
    Examples seen in this codebase:
      - "TP", "EP", "DP", "PP"
      - "DP*EP=64" or "EP*DP=64"
    """
    g = (group or "").upper()
    # Pattern: "...=NUMBER"
    if "=" in g:
        try:
            return int(g.split("=")[-1].strip())
        except Exception:
            pass

    if "TP" == g:
        return int(parallel.tp_size)
    if "EP" == g:
        return int(parallel.ep_size)
    if "DP" == g:
        return int(parallel.dp_size)
    if "PP" == g:
        # PP is point-to-point between stages; whether it crosses nodes depends on placement.
        # Use total GPU count as a conservative proxy for "might cross node" in large clusters.
        return int(gpu.num_gpus)

    # Mixed groups fallback
    if "DP" in g and "EP" in g:
        return int(parallel.dp_size * parallel.ep_size)
    if "DP" in g:
        return int(parallel.dp_size)
    if "EP" in g:
        return int(parallel.ep_size)
    if "TP" in g:
        return int(parallel.tp_size)

    return int(gpu.num_gpus)


def _select_fabric_lane(group: str, parallel: ParallelConfig, gpu: GPU_Resource, profile: GPUHwProfile) -> str:
    """
    Decide whether the collective likely uses ScaleUp (intra-node) or ScaleOut (inter-node)
    based on inferred group size and the GPU profile's scaleup boundary.
    """
    size = _infer_collective_group_size(group, parallel, gpu)
    return "scaleup" if size <= int(profile.scaleup_boundary) else "scaleout"

def _collective_efficiency(ctype: str) -> float:
    """Heuristic efficiency factor (0..1) for different collective types.

    Real systems rarely achieve the raw link-rate, especially for all-to-all patterns.
    This factor is applied to the fabric bandwidth to obtain an *effective* bandwidth.
    """
    c = (ctype or "").lower()
    if c in {"send", "recv"}:
        return 0.95
    if c == "allreduce":
        return 0.85
    if c == "reducescatter":
        return 0.80
    if c == "allgather":
        return 0.80
    if c == "all2all":
        return 0.65
    return 0.80


def _split_fabric_bytes(
    *,
    bytes_total: float,
    ctype: str,
    group: str,
    parallel: ParallelConfig,
    gpu: GPU_Resource,
    profile: GPUHwProfile,
    moe_max_nodes_per_token: int | None = None,
) -> tuple[float, float]:
    """Split a collective's payload into ScaleUp (intra-node) vs ScaleOut (inter-node) bytes.

    Why:
      The previous v1 logic used a hard threshold: if group_size > scaleup_boundary, send *all* bytes on ScaleOut.
      This systematically overestimates ScaleOut time for large groups, because a fraction of peers are still
      on the same node, and some workloads (e.g., DeepSeek MoE routing) intentionally constrain cross-node fanout.

    Model:
      - Let B = scaleup_boundary (GPUs per node).
      - Let G = group size.
      - If G <= B: (su, so) = (bytes_total, 0)
      - Else:
          intra_frac ~= (B-1)/(G-1)  (fraction of peer exchanges that stay within the node)
          inter_frac = 1 - intra_frac
          su = bytes_total * intra_frac
          so = bytes_total * inter_frac
        For MoE all2all on EP groups, optionally apply a locality reduction to inter-node traffic:
          nodes_total = ceil(G / B)
          nodes_used  = min(nodes_total, moe_max_nodes_per_token)
          locality_frac ~= (nodes_used-1)/(nodes_total-1)
          so *= locality_frac
        This reduces *effective* cross-node bytes while keeping intra-node bytes unchanged.

    Returns:
      (scaleup_bytes, scaleout_bytes)
    """
    b = float(bytes_total or 0.0)
    if b <= 0:
        return 0.0, 0.0

    G = _infer_collective_group_size(group, parallel, gpu)
    B = int(getattr(profile, "scaleup_boundary", 0) or 0)
    if B <= 1 or G <= 1:
        # Degenerate or unknown: treat as scale-out if group spans >1, else scale-up.
        return (b, 0.0) if G <= 1 else (0.0, b)

    if G <= B:
        return b, 0.0

    intra_frac = (B - 1) / (G - 1)
    intra_frac = max(0.0, min(1.0, intra_frac))
    inter_frac = 1.0 - intra_frac

    su = b * intra_frac
    so = b * inter_frac

    c = (ctype or "").lower()
    # Optional: DeepSeek-style MoE locality (max nodes per token) for EP all2all.
    if c == "all2all" and "EP" in (group or "") and moe_max_nodes_per_token:
        nodes_total = int(math.ceil(G / B))
        if nodes_total > 1:
            nodes_used = int(max(1, min(nodes_total, moe_max_nodes_per_token)))
            locality_frac = (nodes_used - 1) / (nodes_total - 1) if nodes_total > 1 else 1.0
            locality_frac = max(0.0, min(1.0, locality_frac))
            so *= locality_frac

    return su, so


def _schedule_bytes(
    tail: float,
    ready_time: float,
    bytes_total: float,
    bw_bytes_per_s: float,
    latency_s: float = 0.0,
) -> float:
    """
    Schedule a comm event on a fabric lane with a readiness dependency.
    """
    if bytes_total <= 0 or bw_bytes_per_s <= 0:
        return max(tail, ready_time)
    start = max(tail, ready_time)
    return start + latency_s + (bytes_total / bw_bytes_per_s)


# --------------------------
# Core simulation
# --------------------------
def _run_component(
    *,
    comp: model_component_compute_spec,
    flops: float,
    hbm_bytes: float,
    dtype: str,
    peak_tensor: float,
    peak_cuda: float,
    hbm_bw: float,
    batch_local: float,
    tensor_tail: float,
    cuda_tail: float,
    hbm_tail: float,
) -> Tuple[float, float, float]:
    """
    Schedule one component (either TensorCore lane or CUDA lane) with HBM lane coupling.
    Returns updated (tensor_tail, cuda_tail, hbm_tail).
    """
    if dtype in TENSOR_CORE_DTYPES:
        eff = _component_efficiency(comp, peak_tensor, hbm_bw, batch_local=batch_local)
        t_compute = flops / (peak_tensor * eff) if peak_tensor > 0 else 0.0
        t_hbm = hbm_bytes / hbm_bw if hbm_bw > 0 else 0.0
        end = max(tensor_tail, hbm_tail) + max(t_compute, t_hbm)
        return end, cuda_tail, end

    eff = _component_efficiency(comp, peak_cuda, hbm_bw, batch_local=batch_local)
    t_compute = flops / (peak_cuda * eff) if peak_cuda > 0 else 0.0
    t_hbm = hbm_bytes / hbm_bw if hbm_bw > 0 else 0.0
    end = max(cuda_tail, hbm_tail) + max(t_compute, t_hbm)
    return tensor_tail, end, end


def simulate_rank(
    rank: runtime_rank_spec,
    compute_graph: model_compute_graph,
    profile: GPUHwProfile,
    sim: SimulationInput,
    parallel: ParallelConfig,
    gpu: GPU_Resource | None = None,
) -> RankTimeline:
    """
    Returns:
      - step_s: per-microbatch stage time (FWD+BWD) for this rank's stage
      - grad_update_s: per-step gradient sync time (not multiplied by microbatches)
    """
    policy = get_policy(sim.precision.policy_name)
    peak_flops_map = profile.peak_flops
    peak_tensor = float(peak_flops_map.get(policy.compute_w_dtype.lower(), peak_flops_map.get("fp16", 0.0)))
    peak_cuda = float(peak_flops_map.get("fp32", 0.0))
    hbm_bw = float(profile.hbm_bandwidth)

    # Per-microbatch compute scaling: global -> per-rank micro
    batch_scale = 1.0 / max(parallel.dp_size * parallel.ep_size * sim.train.num_micro_batches, 1)
    tp = max(parallel.tp_size, 1)
    batch_local = float(sim.train.global_batch_size) * batch_scale

    tensor_tail = 0.0
    cuda_tail = 0.0
    hbm_tail = 0.0

    span = set(rank.model_layers_span)
    if gpu is None:
        gpu = GPU_Resource(num_gpus=parallel.dp_size * parallel.pp_size * parallel.tp_size * parallel.ep_size, gpu_sku=profile.name)

    if not span:
        return RankTimeline(
            compute_tensor_s=0.0,
            compute_cuda_s=0.0,
            hbm_s=0.0,
            scaleup_s=0.0,
            scaleout_s=0.0,
            comm_scaleup_bytes=0.0,
            comm_scaleout_bytes=0.0,
            step_s=0.0,
            grad_update_s=0.0,
            bottleneck="empty",
        )

    first_layer = min(span)
    last_layer = max(span)
    L_total = int(sim.model.num_hidden_layers)

    # Record layer-end times for readiness dependencies
    fwd_layer_end: Dict[int, float] = {}
    bwd_layer_end: Dict[int, float] = {}

    # ---- Forward (per micro) ----
    if first_layer == 1:
        # embedding -> pos_embedding (if stage owns layer-1)
        for comp in (compute_graph.embedding, compute_graph.pos_embedding):
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.fwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.fwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.fwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )

    for layer in compute_graph.layers:
        if layer.lidx not in span:
            continue
        for comp in layer.components:
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.fwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.fwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.fwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )
        fwd_layer_end[int(layer.lidx)] = max(tensor_tail, cuda_tail, hbm_tail)

    if last_layer == L_total:
        # logits -> loss on the last stage
        for comp in (compute_graph.logits, compute_graph.entropyloss):
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.fwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.fwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.fwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )
        # Use a virtual layer id for "post last" readiness if needed
        fwd_layer_end[L_total + 1] = max(tensor_tail, cuda_tail, hbm_tail)

    # ---- Backward (per micro) ----
    if last_layer == L_total:
        # backprop starts from loss (loss -> logits)
        for comp in (compute_graph.entropyloss, compute_graph.logits):
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.bwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.bwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.bwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )

    for layer in reversed(compute_graph.layers):
        if layer.lidx not in span:
            continue
        for comp in reversed(layer.components):
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.bwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.bwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.bwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )
        bwd_layer_end[int(layer.lidx)] = max(tensor_tail, cuda_tail, hbm_tail)

    if first_layer == 1:
        # embedding grads come at the end on the first stage (pos -> embedding)
        for comp in (compute_graph.pos_embedding, compute_graph.embedding):
            tp_scale = _tp_scale_for_component(comp.component_type, tp)
            flops = float(sum(comp.bwd_flops.values())) * batch_scale * tp_scale
            hbm_bytes = float(sum(comp.bwd_hbm_throughput.values())) * batch_scale * tp_scale
            dtype = _main_dtype_from_flops(comp.bwd_flops)
            tensor_tail, cuda_tail, hbm_tail = _run_component(
                comp=comp,
                flops=flops,
                hbm_bytes=hbm_bytes,
                dtype=dtype,
                peak_tensor=peak_tensor,
                peak_cuda=peak_cuda,
                hbm_bw=hbm_bw,
                batch_local=batch_local,
                tensor_tail=tensor_tail,
                cuda_tail=cuda_tail,
                hbm_tail=hbm_tail,
            )

    compute_end = max(tensor_tail, cuda_tail, hbm_tail)

    # ---- Per-micro collectives (schedule with readiness) ----
    su_tail = 0.0
    so_tail = 0.0
    su_bytes = 0.0
    so_bytes = 0.0
    su_bw = float(profile.scaleup_bandwidth)
    so_bw = float(profile.scaleout_bandwidth)

    # MoE routing locality hint: DeepSeek-style tokens are constrained to a limited number of nodes.
    # This is only used to split ScaleUp/ScaleOut bytes for EP all2all. If you want explicit control,
    # add `moe_max_nodes_per_token` to ModelConfig (optional) and set it from the UI.
    moe_max_nodes = getattr(getattr(sim, "model", None), "moe_max_nodes_per_token", None)
    if moe_max_nodes is None:
        # Heuristic default for DeepSeek-like MoE configs (TopK=8, MoE present, many routed experts).
        if (
            int(getattr(sim.model, "num_experts_per_tok", 0) or 0) == 8
            and int(getattr(sim.model, "num_moe_hidden_layers", 0) or 0) > 0
            and int(getattr(sim.model, "n_routed_experts", 0) or 0) >= 64
        ):
            moe_max_nodes = 4

    def _schedule_collective(name: str, payload: Dict[str, Any], phase: str) -> None:
        nonlocal su_tail, so_tail, su_bytes, so_bytes
        ctype = str(payload.get("collective_type", "none")).lower()
        if ctype == "none":
            return
        bytes_total = float(payload.get("collective-throughput", 0.0) or 0.0)
        if bytes_total <= 0:
            return

        layer_id = int(payload.get("layer", 0) or 0)

        # readiness time
        if phase == "fwd":
            ready = float(fwd_layer_end.get(layer_id, 0.0))
        elif phase == "bwd":
            ready = float(bwd_layer_end.get(layer_id, compute_end))
        elif phase == "pp_fwd":
            ready = float(fwd_layer_end.get(layer_id, compute_end))
        elif phase == "pp_bwd":
            ready = float(bwd_layer_end.get(layer_id, compute_end))
        else:
            ready = compute_end

        group = str(payload.get("collective_group", "") or "")
        # Split bytes across ScaleUp/ScaleOut lanes (hybrid collectives + optional MoE locality)
        su_part, so_part = _split_fabric_bytes(
            bytes_total=bytes_total,
            ctype=ctype,
            group=group,
            parallel=parallel,
            gpu=gpu,
            profile=profile,
            moe_max_nodes_per_token=moe_max_nodes,
        )

        eff = _collective_efficiency(ctype)
        su_eff_bw = su_bw * eff
        so_eff_bw = so_bw * eff

        if su_part > 0:
            su_tail = _schedule_bytes(su_tail, ready, su_part, su_eff_bw)
            su_bytes += su_part
        if so_part > 0:
            so_tail = _schedule_bytes(so_tail, ready, so_part, so_eff_bw)
            so_bytes += so_part

    # FWD collectives
    for coll in rank.components_collectives_fwd:
        name, payload = list(coll.items())[0]
        _schedule_collective(name, payload, "fwd")

    # BWD collectives
    for coll in rank.components_collectives_bwd:
        name, payload = list(coll.items())[0]
        _schedule_collective(name, payload, "bwd")

    # PP boundaries (explicit keys include *_fwd / *_bwd)
    for coll in rank.pp_stage_collective:
        name, payload = list(coll.items())[0]
        lname = name.lower()
        if "fwd" in lname:
            _schedule_collective(name, payload, "pp_fwd")
        elif "bwd" in lname:
            _schedule_collective(name, payload, "pp_bwd")
        else:
            _schedule_collective(name, payload, "pp_fwd")

    micro_step_time = max(compute_end, su_tail, so_tail)

    # ---- Per-step grad update collectives (NOT per micro) ----
    # We conservatively treat grad-sync as happening after the last micro finishes on this stage.
    grad_su_tail = 0.0
    grad_so_tail = 0.0
    grad_su_bytes = 0.0
    grad_so_bytes = 0.0

    for coll in rank.grad_update_collective:
        name, payload = list(coll.items())[0]
        ctype = str(payload.get("collective_type", "none")).lower()
        if ctype == "none":
            continue
        bytes_total = float(payload.get("collective-throughput", 0.0) or 0.0)
        if bytes_total <= 0:
            continue

        group = str(payload.get("collective_group", "") or "")
        # ready at end of micro compute (we add it as a post-step tail in simulate_distributed)
        ready = micro_step_time

        su_part, so_part = _split_fabric_bytes(
            bytes_total=bytes_total,
            ctype=ctype,
            group=group,
            parallel=parallel,
            gpu=gpu,
            profile=profile,
            moe_max_nodes_per_token=moe_max_nodes,
        )

        eff = _collective_efficiency(ctype)
        su_eff_bw = su_bw * eff
        so_eff_bw = so_bw * eff

        if su_part > 0:
            grad_su_tail = _schedule_bytes(grad_su_tail, ready, su_part, su_eff_bw)
            grad_su_bytes += su_part
        if so_part > 0:
            grad_so_tail = _schedule_bytes(grad_so_tail, ready, so_part, so_eff_bw)
            grad_so_bytes += so_part

    grad_update_time = max(grad_su_tail, grad_so_tail, micro_step_time) - micro_step_time

    # Bottleneck classification (micro stage)
    bottleneck = max(
        [
            ("tensor", tensor_tail),
            ("cuda", cuda_tail),
            ("hbm", hbm_tail),
            ("scaleup", su_tail),
            ("scaleout", so_tail),
        ],
        key=lambda kv: kv[1],
    )[0]

    return RankTimeline(
        compute_tensor_s=tensor_tail,
        compute_cuda_s=cuda_tail,
        hbm_s=hbm_tail,
        scaleup_s=su_tail,
        scaleout_s=so_tail,
        comm_scaleup_bytes=su_bytes,
        comm_scaleout_bytes=so_bytes,
        step_s=micro_step_time,
        grad_update_s=grad_update_time,
        grad_update_scaleup_bytes=grad_su_bytes,
        grad_update_scaleout_bytes=grad_so_bytes,
        bottleneck=bottleneck,
    )


def _scale_time_unit(seconds: float) -> Tuple[float, str]:
    if seconds >= 1:
        return seconds, "s"
    if seconds >= 1e-3:
        return seconds * 1e3, "ms"
    return seconds * 1e6, "us"


def simulate_distributed(sim: SimulationInput, parallel: ParallelConfig, gpu: GPU_Resource) -> Dict[str, Any]:
    """
    Returns a dict that is easy to JSONify.

    step_time is an *optimizer-step wall-clock estimate*:
      step_time ≈ (M + P - 1) * max_stage_micro_time  +  max_stage_grad_update_time
    """
    profile = get_gpu_profile(gpu.gpu_sku)
    compute_graph = parse_model_compute_spec(sim.model, sim.train, sim.precision)
    ranks = build_runtime_rank_graph(parallel, sim.model, sim.train, sim.precision, gpu)

    rank_results: List[RankTimeline] = []
    stage_micro: Dict[int, float] = {}
    stage_grad: Dict[int, float] = {}

    for r in ranks:
        result = simulate_rank(r, compute_graph, profile, sim, parallel, gpu)
        rank_results.append(result)

        stage = int(r.pipeline_stage)
        stage_micro[stage] = max(stage_micro.get(stage, 0.0), float(result.step_s))
        stage_grad[stage] = max(stage_grad.get(stage, 0.0), float(result.grad_update_s))

    max_stage_micro = max(stage_micro.values()) if stage_micro else 0.0
    max_stage_grad = max(stage_grad.values()) if stage_grad else 0.0

    P = int(parallel.pp_size)
    M = int(sim.train.num_micro_batches)

    bubbles = (M + P - 1) if (M > 0 and P > 0) else 0
    step_seconds = bubbles * max_stage_micro + max_stage_grad

    step_scaled, unit = _scale_time_unit(step_seconds)
    per_stage_micro = {k: _scale_time_unit(v)[0] for k, v in stage_micro.items()}
    per_stage_grad = {k: _scale_time_unit(v)[0] for k, v in stage_grad.items()}

    return {
        "per_rank": rank_results,
        "per_stage_micro_time": per_stage_micro,
        "per_stage_grad_update_time": per_stage_grad,
        "time_unit": unit,
        "step_time": step_scaled,
        "max_stage_micro_time": _scale_time_unit(max_stage_micro)[0],
        "max_stage_grad_update_time": _scale_time_unit(max_stage_grad)[0],
    }