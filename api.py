from __future__ import annotations
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import time
from math import ceil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydeck.widget.widget import Int

from Compute_Graph import parse_model_compute_spec, parse_model_tensor_spec, tensor_spec
from GPU_Profiler import GPU_PROFILES, get_gpu_profile
from Precision_Policy import get_policy
from User_Config import GPU_Resource, ParallelConfig, SimulationInput
from Parallel import build_runtime_rank_graph

# timeline.py 你本地项目里有（你说是最新版本）
from Frontend_app import total_tokens
from Timeline import simulate_distributed as simulate_distributed_core


# -----------------------------
# utilities
# -----------------------------
def _bytes_to_gb(x: float) -> float:
    return float(x) / (1024**3)


def _bytes_to_tb(x: float) -> float:
    return float(x) / (1024**4)


def _flops_to_eflops(x: float) -> float:
    return float(x) / 1e18


def _normalize_total_tokens(total_tokens: float) -> float:
    """
    兼容 UI：Total Tokens (Billion)
    - 如果传 14800 这种（<1e8），按 Billion => *1e9
    - 如果传 1.48e13，保持
    """
    x = float(total_tokens)
    return x * 1e9 if x < 1e8 else x


GPU_SKU_ALIASES: Dict[str, str] = {
    "A100 80G": "A100_80G_SXM",
    "H100 80G": "H100_80G_SXM",
    "H800 80G": "H800_80G_SXM",
    "H20 96G": "H20_96G",
    "Ascend 910C": "Ascend_910C2",
}


def _normalize_gpu_sku(sku: str) -> str:
    s = (sku or "").strip()
    if s in GPU_PROFILES:
        return s
    return GPU_SKU_ALIASES.get(s, s)


def _is_moe_expert(component_type: str) -> bool:
    u = (component_type or "").upper()
    return "MOE_EXPERT" in u


def _layer_to_stage(layer_id: int, num_layers: int, pp: int) -> int:
    if pp <= 1:
        return 0
    span = max(num_layers // pp, 1)
    boundary_last = (pp - 1) * span
    if layer_id > boundary_last:
        return pp - 1
    return max(0, (layer_id - 1) // span)


def _ckpt_keep_factor(use_ckpt: bool, ckpt_ratio: float) -> float:
    if not use_ckpt:
        return 1.0
    r = max(0.0, min(1.0, float(ckpt_ratio)))
    # 保守且单调：最多压到 25%
    return max(0.25, 1.0 - 0.7 * r)


# -----------------------------
# Param count (total vs active)
# -----------------------------
def _params_total_vs_active(sim: SimulationInput) -> Dict[str, float]:
    """
    你当前 tensor graph 的 MoE 参数大概率是按 “active experts” 生成（E_active）。
    为了让 UI 既能显示 paper 的 “总参数”，又能显示 “激活参数”，这里同时返回两种口径。

    total_params_b：把 MoE expert param 从 E_active 线性还原到 E_total
    active_params_b：保持 tensor graph 的现状
    """
    tg = parse_model_tensor_spec(sim.model, sim.train, sim.precision)

    E_active = int(sim.model.num_experts_per_tok)
    E_total = int(sim.model.n_routed_experts + sim.model.n_shared_experts)
    scale = (E_total / E_active) if E_active > 0 else 1.0

    active = 0.0
    total = 0.0
    for layer in tg.layers:
        for t in layer.tensors:
            if t.role != "param":
                continue
            active += float(t.element_count)
            if _is_moe_expert(t.component_type):
                total += float(t.element_count) * scale
            else:
                total += float(t.element_count)

    return {
        "active_params_b": active / 1e9,
        "total_params_b": total / 1e9,
        "E_active": float(E_active),
        "E_total": float(E_total),
    }


# -----------------------------
# HBM capacity estimator (per GPU)
# -----------------------------
def estimate_hbm_per_gpu_bytes(
    sim: SimulationInput,
    *,
    dp: int,
    tp: int,
    pp: int,
    ep: int,
    zero_stage: int,
    system_overhead_ratio: float,
) -> Dict[str, Any]:
    """
    返回 “worst stage” 的 per-GPU HBM 需求（bytes），以及按 role 拆分（用于判断 OOM 根因）。
    默认贴近 Parallel 的 batch 口径：
      B_local = GBS / dp / M / ep
    TP：切 hidden/ffn 维度 => activation / param 都 /tp（粗略）
    EP：MoE expert param /ep；同时 B_local /ep（已在上式体现）
    PP：按 layer 分 stage，取 worst stage
    ZeRO：stage>=1 shard optimizer_state；stage>=2 shard param_gradient；stage>=3 shard param/master_param
    Checkpoint：只压 activation（保守）
    """
    tg = parse_model_tensor_spec(sim.model, sim.train, sim.precision)

    dp = max(int(dp), 1)
    tp = max(int(tp), 1)
    pp = max(int(pp), 1)
    ep = max(int(ep), 1)
    zero_stage = int(max(0, min(3, zero_stage)))

    use_zero = bool(getattr(sim.train, "use_zero", True))
    use_ckpt = bool(getattr(sim.train, "use_activation_checkpoint", True))
    ckpt_ratio = float(getattr(sim.train, "ckpt_ratio", 0.5))
    keep = _ckpt_keep_factor(use_ckpt, ckpt_ratio)

    M = max(int(sim.train.num_micro_batches), 1)
    L = int(sim.model.num_hidden_layers)

    per_stage_total: Dict[int, float] = {i: 0.0 for i in range(pp)}
    per_stage_role: Dict[int, Dict[str, float]] = {i: {} for i in range(pp)}

    def add(stage: int, role: str, b: float) -> None:
        per_stage_total[stage] += b
        per_stage_role[stage][role] = per_stage_role[stage].get(role, 0.0) + b

    for layer in tg.layers:
        for t in layer.tensors:
            role = str(t.role)
            b = float(t.hbm_occupy)
            stage = _layer_to_stage(int(t.layer_id), num_layers=L, pp=pp)

            # ---- param-like roles ----
            if role in {"param", "master_param", "optimizer_state", "param_gradient"}:
                # TP shard
                b /= tp
                # EP shard only for MoE expert tensors
                if _is_moe_expert(t.component_type):
                    b /= ep

                if use_zero:
                    if role == "optimizer_state" and zero_stage >= 1:
                        b /= dp
                    if role == "param_gradient" and zero_stage >= 2:
                        b /= dp
                    if role in {"param", "master_param"} and zero_stage >= 3:
                        b /= dp

                add(stage, role, b)
                continue

            # ---- activation-like roles ----
            if role in {"activation", "input_gradient"}:
                # local batch scaling: /dp /M /ep
                b /= (dp * M * ep)
                # TP shard activation (hidden split)
                b /= tp
                # ckpt applies to activation only
                if role == "activation":
                    b *= keep
                add(stage, role, b)
                continue

            add(stage, role, b)

    worst_stage = max(per_stage_total.items(), key=lambda kv: kv[1])[0] if per_stage_total else 0
    base = float(per_stage_total.get(worst_stage, 0.0))
    with_overhead = base * (1.0 + float(system_overhead_ratio))

    return {
        "worst_stage": worst_stage,
        "per_gpu_bytes_base": base,
        "per_gpu_bytes_with_overhead": with_overhead,
        "role_breakdown_bytes": per_stage_role.get(worst_stage, {}),
        "assumptions": {
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "ep": ep,
            "use_zero": use_zero,
            "zero_stage": zero_stage,
            "use_activation_checkpoint": use_ckpt,
            "ckpt_ratio": ckpt_ratio,
            "ckpt_keep_factor": keep,
            "system_overhead_ratio": float(system_overhead_ratio),
        },
    }

def _default_pod_sizes(max_total_gpus: int) -> List[int]:
    # Typical pods; always keep within max_total_gpus
    base = [128, 256, 512, 1024, 2048, 4096, 8192]
    return [x for x in base if x <= max_total_gpus]


def _find_min_dp_pow2_for_scheme(
    sim,
    gpu_sku: str,
    *,
    tp: int,
    pp: int,
    ep: int,
    zero_stage: int,
    system_overhead_ratio: float,
    max_total_gpus: int,
    _hbm_cache: dict | None = None,   # ✅ 新增：可选 cache
):
    """
    固定 (tp,pp,ep,zero_stage)，搜索 dp=2^k 的最小可行解（per_gpu_bytes_with_overhead <= cap）。
    兼容 fast recommender：支持 _hbm_cache 进行 estimate 结果缓存。
    """
    sku = _normalize_gpu_sku(gpu_sku)
    prof = GPU_PROFILES[sku]
    cap_bytes = float(getattr(prof, "hbm_capacity", 0.0))
    if cap_bytes <= 0:
        return None

    tp = max(int(tp), 1)
    pp = max(int(pp), 1)
    ep = max(int(ep), 1)
    denom = tp * pp * ep
    if denom <= 0:
        return None

    dp_max = max(int(max_total_gpus // denom), 1)

    dp = 1
    while dp <= dp_max:
        key = (sku, dp, tp, pp, ep, int(zero_stage), float(system_overhead_ratio))

        est = None
        if _hbm_cache is not None:
            est = _hbm_cache.get(key)

        if est is None:
            est = estimate_hbm_per_gpu_bytes(
                sim,
                dp=dp,
                tp=tp,
                pp=pp,
                ep=ep,
                zero_stage=zero_stage,
                system_overhead_ratio=system_overhead_ratio,
            )
            if _hbm_cache is not None:
                _hbm_cache[key] = est

        per_gpu = float(est.get("per_gpu_bytes_with_overhead", 0.0))
        if per_gpu > 0 and per_gpu <= cap_bytes:
            return {"dp": dp, "est": est}

        dp *= 2

    return None

def _select_diverse_to_sim(items: list[dict], k: int, heuristic_key):
    """
    从一个 bucket 的候选 items 里选出 k 个用于 timeline 仿真：
    - 先选 heuristic 最优的 1 个（保证基本正确性）
    - 再用“最远点优先”（farthest-first）补齐剩余：保证结构多样性
    这样最优方案就算落在启发式中间，也更可能被覆盖到。
    """
    k = max(int(k), 0)
    if k <= 0 or not items:
        return []

    # 1) 先按 heuristic 排序，拿第 1 个当 seed
    items_sorted = sorted(items, key=heuristic_key)
    chosen = [items_sorted[0]]
    if k == 1:
        return chosen

    # 2) 定义“结构特征向量”（只用离散并行因子 + util）
    def feat(s: dict):
        p = s["parallel"]
        # log2 更符合尺度感
        dp = float(p.get("dp", 1))
        tp = float(p.get("tp", 1))
        pp = float(p.get("pp", 1))
        ep = float(p.get("ep", 1))
        util = float(s.get("hbm_util", 0.0))

        def lg(x):  # safe log2
            x = max(float(x), 1.0)
            return math.log2(x)

        return (lg(dp), lg(tp), lg(pp), lg(ep), util)

    def dist2(a, b):
        # 简单欧氏距离（不指定颜色那种朴素好用）
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

    feats = {id(s): feat(s) for s in items_sorted}

    # 3) 极值点也优先考虑（覆盖边界）
    def pick_extreme(key_fn):
        best = None
        best_v = None
        for s in items_sorted:
            if s in chosen:
                continue
            v = key_fn(s)
            if best is None or v > best_v:
                best, best_v = s, v
        if best is not None:
            chosen.append(best)

    # 覆盖一些“方向”：dp 大、pp 小、ep 大、tp 小、util 更贴 target
    pick_extreme(lambda s: float(s["parallel"].get("dp", 1)))           # max dp
    if len(chosen) < k:
        pick_extreme(lambda s: -float(s["parallel"].get("pp", 1)))      # min pp
    if len(chosen) < k:
        pick_extreme(lambda s: float(s["parallel"].get("ep", 1)))       # max ep
    if len(chosen) < k:
        pick_extreme(lambda s: -float(s["parallel"].get("tp", 1)))      # min tp

    # 4) 剩余用 farthest-first：每次挑离当前 chosen 最远的点
    while len(chosen) < k:
        best = None
        best_d = -1.0
        for s in items_sorted:
            if s in chosen:
                continue
            fs = feats[id(s)]
            # 到已选集合的最小距离（maximin）
            d = min(dist2(fs, feats[id(c)]) for c in chosen)
            if d > best_d:
                best, best_d = s, d
        if best is None:
            break
        chosen.append(best)

    # 保持 chosen 内也按 heuristic 排（仿真顺序稳定）
    chosen = sorted(chosen, key=heuristic_key)
    return chosen[:k]


def _attach_train_time_metrics_bounded(
    sim,
    gpu_sku: str,
    schemes: list[dict],
    *,
    system_overhead_ratio: float,
    per_bucket_k: int,
    time_budget_s: float,
    heuristic_key =None,  #新增排序用
):
    """
    Timeline simulation with strict bounds:
    - Only simulate up to `per_bucket_k` per total_gpus bucket
    - Stop when time_budget_s exceeded
    - Memoize simulate results by (sku,total_gpus,dp,tp,pp,ep,system_overhead_ratio)

    Requires in scope:
      - _normalize_gpu_sku
      - ParallelConfig, GPU_Resource
      - simulate_distributed_core
    """
    sku = _normalize_gpu_sku(gpu_sku)

    total_tokens = float(getattr(sim.train, "total_tokens", 0.0))
    gbs = int(getattr(sim.train, "global_batch_size", 1))
    seq = int(getattr(sim.train, "seq_len", 1))
    if total_tokens <= 0 or gbs <= 0 or seq <= 0:
        return schemes

    total_steps = int(ceil(total_tokens / float(gbs * seq)))
    if total_steps <= 0:
        return schemes

    # group by bucket
    buckets = {}
    for s in schemes:
        buckets.setdefault(int(s["total_gpus"]), []).append(s)

    start_t = time.time()
    sim_cache = {}

    out = []
    for total, items in buckets.items():
        # items 应该已经按 heuristic 排好序，这里只跑前 K 个
        to_sim = _select_diverse_to_sim(items, per_bucket_k,heuristic_key=heuristic_key)

        for s in to_sim:
            if time_budget_s > 0 and (time.time() - start_t) > time_budget_s:
                out.extend(items)  # 超时：当前桶剩余直接原样返回
                break

            p = s["parallel"]
            dp = int(p["dp"]); tp = int(p["tp"]); pp = int(p["pp"]); ep = int(p["ep"])
            key = (sku, int(total), dp, tp, pp, ep, float(system_overhead_ratio))

            if key in sim_cache:
                step_time_s, wall_h, gpu_h = sim_cache[key]
                s2 = dict(s)
                s2["step_time_s"] = step_time_s
                s2["train_wall_hours"] = wall_h
                s2["train_gpu_hours"] = gpu_h
                out.append(s2)
                continue

            try:
                parallel = ParallelConfig(dp_size=dp, tp_size=tp, pp_size=pp, ep_size=ep)
                gpu = GPU_Resource(num_gpus=int(total), gpu_sku=sku)

                core = simulate_distributed_core(sim, parallel, gpu)
                unit = str(core.get("time_unit", "s"))
                step_time = float(core.get("step_time", 0.0))
                scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}.get(unit, 1.0)
                step_time_s_raw = step_time * scale

                over = max(0.0, float(system_overhead_ratio))
                step_time_s = step_time_s_raw * (1.0 + over)

                wall_h = (step_time_s * float(total_steps)) / 3600.0
                gpu_h = wall_h * float(total)

                sim_cache[key] = (float(step_time_s), float(wall_h), float(gpu_h))

                s2 = dict(s)
                s2["step_time_s"] = float(step_time_s)
                s2["train_wall_hours"] = float(wall_h)
                s2["train_gpu_hours"] = float(gpu_h)
                out.append(s2)
            except Exception:
                out.append(s)

        else:
            # 当前桶未模拟的剩余方案，直接拼回
            out.extend(items[max(int(per_bucket_k), 0):])

        # 全局时间预算耗尽：直接返回已完成部分（足够用于 topk）
        if time_budget_s > 0 and (time.time() - start_t) > time_budget_s:
            return out

    return out

def suggest_feasible_schemes(
    sim: Any,
    gpu_sku: str,
    *,
    system_overhead_ratio: float,
    topk: int = 8,
    max_total_gpus: int = 4096,
) -> List[Dict[str, Any]]:
    """
    Fast recommendation.
    Key performance knobs (optional, read from sim.train):
      - tab1_pod_sizes: list[int]  (default common pod sizes up to max_total_gpus)
      - tab1_beam_per_pod: int     (default 20) keep only top-N candidates per pod during enumeration
      - tab1_max_candidates_total: int (default 200) global cap before simulation/rerank
      - tab1_enable_timeline_rerank: bool (default True)
      - tab1_sim_per_pod: int      (default 6) simulate up to K per pod
      - tab1_sim_time_budget_s: float (default 10) hard time budget for timeline calls
      - tab1_optimize_objective: "time" | "gpu_hours" | "balanced" (default "time")
      - tab1_same_cards_preference: "dp" | "pp_ep" | "balanced" (default "balanced")
      - tab1_min_hbm_util / tab1_target_hbm_util / tab1_max_hbm_util
      - tab1_max_tp_ep_product (default 128 here for speed)
      - use_zero (bool), zero_stage (int)

    Returns Top-K (across all pods) in final objective order, but always pod-scale (power-of-two total_gpus).
    """
    sku = _normalize_gpu_sku(gpu_sku)
    if sku not in GPU_PROFILES:
        return []

    prof = GPU_PROFILES[sku]
    cap_gb = _bytes_to_gb(float(getattr(prof, "hbm_capacity", 0.0)))

    # ---- knobs ----
    pod_sizes = getattr(sim.train, "tab1_pod_sizes", None)
    if isinstance(pod_sizes, list) and pod_sizes:
        pod_sizes = [int(x) for x in pod_sizes if int(x) > 0 and int(x) <= max_total_gpus]
        pod_sizes = sorted(set(pod_sizes))
    else:
        pod_sizes = _default_pod_sizes(max_total_gpus)

    beam_per_pod = int(getattr(sim.train, "tab1_beam_per_pod", 20))
    beam_per_pod = max(1, min(beam_per_pod, 24))

    max_candidates_total = int(getattr(sim.train, "tab1_max_candidates_total", 200))
    max_candidates_total = max(16, min(max_candidates_total, 400))

    enable_rerank = bool(getattr(sim.train, "tab1_enable_timeline_rerank", True))
    sim_per_pod = int(getattr(sim.train, "tab1_sim_per_pod", 6))
    sim_per_pod = max(0, min(sim_per_pod, 8))

    sim_time_budget_s = float(getattr(sim.train, "tab1_sim_time_budget_s", 10.0))
    sim_time_budget_s = max(0.0, min(sim_time_budget_s, 20.0))

    objective = str(getattr(sim.train, "tab1_optimize_objective", "time")).strip().lower()
    if objective not in ("time", "gpu_hours", "balanced"):
        objective = "time"

    pref = str(getattr(sim.train, "tab1_same_cards_preference", "balanced")).strip().lower()
    if pref not in ("dp", "pp_ep", "balanced"):
        pref = "balanced"

    min_util = float(getattr(sim.train, "tab1_min_hbm_util", 0.70))
    target_util = float(getattr(sim.train, "tab1_target_hbm_util", 0.88))
    max_util = float(getattr(sim.train, "tab1_max_hbm_util", 0.93))

    # speed: prune more aggressively by default
    max_tp_ep = int(getattr(sim.train, "tab1_max_tp_ep_product", 128))
    max_tp_ep = max(8, min(max_tp_ep, 512))

    # ---- grids (keep small for speed) ----
    tp_cands = [1, 2, 4, 8]
    pp_cands = [1, 2, 4, 8]
    ep_cands = [1, 2, 4, 8, 16, 32, 64]

    tp_cands = [x for x in tp_cands if x <= max(1, int(getattr(sim.model, "num_attention_heads", 1)))]
    pp_cands = [x for x in pp_cands if x <= max(1, int(getattr(sim.model, "num_hidden_layers", 1)))]
    E_total = int(getattr(sim.model, "n_routed_experts", 0) + getattr(sim.model, "n_shared_experts", 0))
    ep_cands = [x for x in ep_cands if x <= max(1, E_total)]

    # ---- ZeRO policy ----
    use_zero = bool(getattr(sim.train, "use_zero", True))
    user_zero = int(getattr(sim.train, "zero_stage", 2))
    primary_zero = user_zero if user_zero in (0, 1, 2) else (2 if use_zero else 0)
    if primary_zero == 3:
        primary_zero = 2
    zero_primary_list = [primary_zero]
    zero_fallback_list = [3] if use_zero else []

    def _note(zero_stage: int, dp: int, tp: int, pp: int, ep: int, util: float) -> str:
        parts = [f"ZeRO-{zero_stage}", f"DP×{dp}"]
        if tp > 1: parts.append(f"TP×{tp}")
        if ep > 1: parts.append(f"EP×{ep}")
        if pp > 1: parts.append(f"PP×{pp}")
        parts.append(f"HBM~{util*100:.1f}%")
        return " + ".join(parts)

    def _heuristic_score(s: Dict[str, Any]):
        total = int(s["total_gpus"])
        util = float(s.get("hbm_util", 0.0))
        p = s["parallel"]
        dp = int(p.get("dp", 1))
        pp = int(p.get("pp", 1))
        ep = int(p.get("ep", 1))
        tp = int(p.get("tp", 1))
        zero = int(s.get("zero_stage", 2))

        # risk/complexity
        pp_pen = pp
        ep_pen = math.log2(ep) if ep > 0 else 0.0
        tp_pen = math.log2(tp) if tp > 0 else 0.0

        if pref == "dp":
            tie = (-dp, pp_pen + 0.5 * ep_pen + 0.25 * tp_pen)
        elif pref == "pp_ep":
            tie = (-(pp * ep), pp_pen + 0.25 * tp_pen)
        else:
            tie = (-dp, pp_pen)

        return (
            # total is power-of-two; smaller is better
            int(math.log2(total)) if total > 0 else 999,
            0 if zero != 3 else 10,
            abs(util - target_util),
            *tie,
        )

    # Candidate storage: bucketed by pod_size
    buckets: Dict[int, List[Dict[str, Any]]] = {int(p): [] for p in pod_sizes}

    def _push(bucket_total: int, item: Dict[str, Any]):
        lst = buckets.get(bucket_total)
        if lst is None:
            return
        lst.append(item)
        lst.sort(key=_heuristic_score)
        # beam keep
        if len(lst) > beam_per_pod:
            del lst[beam_per_pod:]

    # caches
    _hbm_cache: Dict[Tuple, Dict[str, Any]] = {}

    def _enumerate_for_zero(zero_stage: int):
        for tp in tp_cands:
            for pp in pp_cands:
                for ep in ep_cands:
                    if tp * ep > max_tp_ep:
                        continue

                    denom = tp * pp * ep
                    if denom <= 0:
                        continue

                    # 对每个 pod 桶反推 dp，保证能生成 dp=8,pp=4,ep=64,tp=1 这类“非最小dp但更快”的方案
                    for pod_total in list(buckets.keys()):
                        if pod_total % denom != 0:
                            continue
                        dp = pod_total // denom
                        if dp < 1:
                            continue
                        # dp 必须是 2 的幂
                        if (dp & (dp - 1)) != 0:
                            continue

                        # 复用缓存的 HBM 可行性检查
                        key = (sku, int(dp), int(tp), int(pp), int(ep), int(zero_stage), float(system_overhead_ratio))
                        est = _hbm_cache.get(key)
                        if est is None:
                            est = estimate_hbm_per_gpu_bytes(
                                sim,
                                dp=int(dp),
                                tp=int(tp),
                                pp=int(pp),
                                ep=int(ep),
                                zero_stage=int(zero_stage),
                                system_overhead_ratio=float(system_overhead_ratio),
                            )
                            _hbm_cache[key] = est

                        per_gpu_bytes = float(est.get("per_gpu_bytes_with_overhead", 0.0))
                        cap_bytes = float(getattr(GPU_PROFILES[sku], "hbm_capacity", 0.0))
                        if per_gpu_bytes <= 0 or per_gpu_bytes > cap_bytes:
                            continue

                        per_gpu_gb = _bytes_to_gb(per_gpu_bytes)
                        util = (per_gpu_gb / cap_gb) if cap_gb > 0 else 0.0
                        headroom_gb = cap_gb - per_gpu_gb
                        
                        if int(pod_total) == 2048 and int(tp) == 1 and int(pp) == 4 and int(ep) == 64:
                                print("DBG_HIT_SHAPE",
                                    f"pod={int(pod_total)},,dp={dp}",
                                    f"zero={zero_stage},util={util*100:.1f}%",
                                    flush=True
                                    )


                        _push(
                            int(pod_total),
                            {
                                "total_gpus": int(pod_total),
                                "parallel": {"dp": int(dp), "tp": int(tp), "pp": int(pp), "ep": int(ep)},
                                "zero_stage": int(zero_stage),
                                "per_gpu_hbm_gb": float(per_gpu_gb),
                                "per_gpu_hbm_cap_gb": float(cap_gb),
                                "hbm_util": float(util),
                                "hbm_headroom_gb": float(headroom_gb),
                                "worst_stage": int(est.get("worst_stage", 0)),
                                "role_breakdown_gb": {k: _bytes_to_gb(v) for k, v in est.get("role_breakdown_bytes", {}).items()},
                                "note": _note(int(zero_stage), int(dp), int(tp), int(pp), int(ep), float(util)),
                            },
                        )

    # Primary ZeRO
    for z in zero_primary_list:
        _enumerate_for_zero(z)

    # If absolutely none, fallback to ZeRO-3
    any_primary = any(buckets[p] for p in buckets)
    if not any_primary and zero_fallback_list:
        for z in zero_fallback_list:
            _enumerate_for_zero(z)

    # Flatten candidates with util filtering (keep best per bucket)
    flat: List[Dict[str, Any]] = []
    for total in pod_sizes:
        lst = buckets.get(int(total), [])
        if not lst:
            continue
        # filter util to avoid low-util dominating
        good = [s for s in lst if (min_util <= float(s.get("hbm_util", 0.0)) <= max_util)]
        if not good:
            good = [s for s in lst if float(s.get("hbm_util", 0.0)) >= min_util] or lst
        flat.extend(good)

    # Global cap before any simulation
    flat.sort(key=_heuristic_score)
    flat = flat[:max_candidates_total]

    # Optional timeline-based metrics (bounded)
    if enable_rerank and sim_per_pod > 0 and sim_time_budget_s >= 0:
        # ensure within each bucket, heuristic-sorted before sim
        flat.sort(key=lambda s: (s["total_gpus"], _heuristic_score(s)))
        flat = _attach_train_time_metrics_bounded(
            sim,
            sku,
            flat,
            system_overhead_ratio=system_overhead_ratio,
            per_bucket_k=sim_per_pod,
            time_budget_s=sim_time_budget_s,
            heuristic_key=_heuristic_score,
        )

    # Final objective sort
    def _final_score(s: Dict[str, Any]):
        wall_h = float(s.get("train_wall_hours", 1e30))
        gpu_h = float(s.get("train_gpu_hours", 1e30))
        zero = int(s.get("zero_stage", 2))
        util = float(s.get("hbm_util", 0.0))
        if objective == "time":
            return (wall_h, gpu_h, 0 if zero != 3 else 10, abs(util - target_util), _heuristic_score(s))
        if objective == "gpu_hours":
            return (gpu_h, wall_h, 0 if zero != 3 else 10, abs(util - target_util), _heuristic_score(s))
        return (wall_h, gpu_h / 1e3, 0 if zero != 3 else 10, abs(util - target_util), _heuristic_score(s))

    flat.sort(key=_final_score)

    # Top-K globally (across pods)
    return flat[:max(int(topk), 1)]


# -----------------------------
# Tab1 schemas
# -----------------------------
class Tab1Request(BaseModel):
    sim: SimulationInput
    system_overhead_ratio: float = Field(0.10, ge=0.0, le=0.5)
    gpu_skus: Optional[List[str]] = None  # 可传 UI 展示名或 profile key


class Tab1GpuRow(BaseModel):
    gpu_sku: str
    hbm_gb: float

    # ✅ 始终返回 Top-K 可行切分方案（可直接填 Tab2）
    feasible_schemes: List[Dict[str, Any]] = []

class Tab1Response(BaseModel):
    # 参数与显存需求
    total_params_b: float
    active_params_b: float

    # HBM（容量需求，按 dp=1 worst-stage per-gpu 口径展示；用于 UI 卡片）
    hbm_total_need_tb: float
    hbm_system_overhead_gb: float

    # Compute & throughput（per optimizer-step）
    step_flops_eflops: float
    step_hbm_throughput_tb: float

    total_steps: int
    gpu_rows: List[Tab1GpuRow]
    notes: List[str]


# -----------------------------
# Tab2 schemas（按 UI 卡片结构输出）
# -----------------------------
class Tab2Request(BaseModel):
    sim: SimulationInput
    gpu: GPU_Resource
    parallel: ParallelConfig
    system_overhead_ratio: float = Field(0.10, ge=0.0, le=0.5)


class Tab2Response(BaseModel):
    # 通信需求
    comm_bytes_per_rank_tb: float
    comm_by_collective_tb: Dict[str, float]

    # 通信耗时（用 timeline 的 per-rank lane 时间为主）
    comm_time_scaleup_s: float
    comm_time_scaleout_s: float

    # step wall time
    step_time: float
    time_unit: str
    step_time_s: float
    step_gpu_hours: float
    Total_gpu_hours: float
    Total_days: float



    # 占比（用 timeline 的 bottleneck 分桶近似）
    ratio_compute: float
    ratio_hbm: float
    ratio_comm: float

    notes: List[str]


def _collect_comm_by_collective_from_rank0(rank0, num_micro_batches: int) -> Dict[str, float]:
    """
    返回 per-optimizer-step 的通信量拆分（bytes）。
    - components_collectives_fwd/bwd/pp_stage: 每个 micro-batch 都发生 => 乘 num_micro_batches
    - grad_update_collective: 每个 optimizer step 发生一次 => 不乘
    """
    buckets = {
        "allreduce": 0.0,
        "all2all": 0.0,
        "allgather": 0.0,
        "reducescatter": 0.0,
        "send": 0.0,
        "recv": 0.0,
        "other": 0.0,
    }

    M = max(int(num_micro_batches), 1)

    micro_lists = rank0.components_collectives_fwd + rank0.components_collectives_bwd + rank0.pp_stage_collective
    grad_lists = rank0.grad_update_collective

    def add_lists(lists, scale: float) -> None:
        for coll in lists:
            for _, payload in coll.items():
                ctype = str(payload.get("collective_type", "other")).lower()
                b = float(payload.get("collective-throughput", 0.0))
                if b <= 0:
                    continue
                b *= scale
                buckets[ctype if ctype in buckets else "other"] += b

    add_lists(micro_lists, scale=float(M))
    add_lists(grad_lists, scale=1.0)

    return buckets


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="LLM Training Resource Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/tab1/analyze", response_model=Tab1Response)
def tab1_analyze(req: Tab1Request) -> Tab1Response:
    sim = req.sim
    sim.train.total_tokens = _normalize_total_tokens(sim.train.total_tokens)

    overhead = float(req.system_overhead_ratio)

    # params
    params = _params_total_vs_active(sim)

    # flops / hbm throughput per step
    cg = parse_model_compute_spec(sim.model, sim.train, sim.precision)
    flops_step = 0.0
    hbm_step = 0.0
    for layer in cg.layers:
        for comp in layer.components:
            flops_step += float(sum(comp.fwd_flops.values()) + sum(comp.bwd_flops.values()))
            hbm_step += float(sum(comp.fwd_hbm_throughput.values()) + sum(comp.bwd_hbm_throughput.values()))

    # steps
    total_steps = int(ceil(sim.train.total_tokens / (sim.train.global_batch_size * sim.train.seq_len)))

    # HBM capacity card：用 dp=1 的 per-gpu worst-stage（这跟 UI “HBM 总容量需求”最贴近）
    zero_stage = int(getattr(sim.train, "zero_stage", 2))
    est_dp1 = estimate_hbm_per_gpu_bytes(
        sim,
        dp=1,
        tp=1,
        pp=1,
        ep=1,
        zero_stage=zero_stage,
        system_overhead_ratio=overhead,
    )
    hbm_need_tb = _bytes_to_tb(float(est_dp1["per_gpu_bytes_with_overhead"]))
    hbm_overhead_gb = _bytes_to_gb(float(est_dp1["per_gpu_bytes_base"]) * overhead)

    # GPUs list
    default_skus = ["A100 80G", "H100 80G", "H800 80G", "H20 96G", "Ascend 910C"]
    skus = req.gpu_skus or default_skus
    skus = [_normalize_gpu_sku(s) for s in skus]
    skus = [s for s in skus if s in GPU_PROFILES]

    rows: List[Tab1GpuRow] = []
    for sku in skus:
        prof = get_gpu_profile(sku)
        cap = float(prof.hbm_capacity)

        schemes = suggest_feasible_schemes(
            sim,
            sku,
            system_overhead_ratio
            =overhead,
            topk=5,
            max_total_gpus=4096,
        )

        rows.append(
            Tab1GpuRow(
                gpu_sku=sku,
                hbm_gb=_bytes_to_gb(cap),
                feasible_schemes=schemes,
            )
        )

    notes = [
        "Tab1 不再输出 dp-only 最小卡数；统一输出每种 GPU 的 Top-K 可行切分方案（可直接填 Tab2）。",
        "方案按总卡数最少优先排序，并且 **总卡数/DP 均按 2 的幂次（pod 规模）** 推荐；同时包含默认 ZeRO-2 与 ZeRO-3（救命选项）。",
        "HBM 吞吐是 Compute_Graph 的理论估算；若 attention 仍按 S×S 落地，会明显偏大（可改为 FlashAttention 风格估算）。",
    ]

    return Tab1Response(
        total_params_b=float(params["total_params_b"]),
        active_params_b=float(params["active_params_b"]),
        hbm_total_need_tb=float(hbm_need_tb),
        hbm_system_overhead_gb=float(hbm_overhead_gb),
        step_flops_eflops=float(_flops_to_eflops(flops_step)),
        step_hbm_throughput_tb=float(_bytes_to_tb(hbm_step)),
        total_steps=total_steps,
        gpu_rows=rows,
        notes=notes,
    )


@app.post("/tab2/analyze", response_model=Tab2Response)
def tab2_analyze(req: Tab2Request) -> Tab2Response:
    sim = req.sim
    sim.train.total_tokens = _normalize_total_tokens(sim.train.total_tokens)

    gpu = GPU_Resource(num_gpus=req.gpu.num_gpus, gpu_sku=_normalize_gpu_sku(req.gpu.gpu_sku))
    parallel = req.parallel
    overhead = float(req.system_overhead_ratio)

    # 1) comm requirement (rank0 展示；按 optimizer-step 口径)
    rank_specs = build_runtime_rank_graph(parallel, sim.model, sim.train, sim.precision, gpu)
    rank0 = rank_specs[0]
    span = list(getattr(rank0, "model_layers_span", []))
      
    M = max(int(sim.train.num_micro_batches), 1)
    comm_by_collective = _collect_comm_by_collective_from_rank0(rank0, M)
    comm_bytes_rank0 = float(sum(comm_by_collective.values()))

    # 2) timeline simulate
    core = simulate_distributed_core(sim, parallel, gpu)
    unit = str(core.get("time_unit", "s"))
    step_time = float(core.get("step_time", 0.0))

    # convert to seconds
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}.get(unit, 1.0)
    step_time_s = step_time * scale

    # ---- system overhead (wall-time only) ----
    # 建议你把这个字段放到 train config 里：sim.train.system_overhead_ratio
    system_overhead_ratio = float(getattr(sim.train, "system_overhead_ratio", 0.0))
    system_overhead_ratio = max(0.0, system_overhead_ratio)  # 不允许负数

    step_time_s_eff = step_time_s * (1.0 + system_overhead_ratio)
    step_gpu_hours = step_time_s_eff * float(gpu.num_gpus) / 3600.0
    Total_gpu_hours = step_gpu_hours * total_tokens / sim.train.global_batch_size / sim.train.seq_len
    Total_days = Total_gpu_hours / 24.0 / float(gpu.num_gpus)



    per_rank = core.get("per_rank", [])
    if per_rank:
        # worst-rank as bottleneck proxy
        worst = max(per_rank, key=lambda r: float(getattr(r, "step_s", 0.0)))

        # micro-step lane end times (seconds). 注意：这些是“lane 尾时间”，不是互斥耗时。
        compute_end = max(
            float(getattr(worst, "compute_tensor_s", 0.0)),
            float(getattr(worst, "compute_cuda_s", 0.0)),
            float(getattr(worst, "hbm_s", 0.0)),
        )
        comm_end = max(float(getattr(worst, "scaleup_s", 0.0)), float(getattr(worst, "scaleout_s", 0.0)))
        micro_time = float(getattr(worst, "step_s", 0.0))  # == max(compute_end, comm_end) in your timeline

        # optimizer-step wall time from timeline (already includes pipeline bubble + grad_update)
        denom = max(float(step_time_s_eff), 1e-12)

        PP = max(int(parallel.pp_size), 1)
        micro_factor = M + PP - 1  # pipeline bubble model used by timeline

        # ===== DBG: implied effective bandwidth used by timeline =====
        micro_su_bytes = float(getattr(worst, "comm_scaleup_bytes", 0.0))
        micro_so_bytes = float(getattr(worst, "comm_scaleout_bytes", 0.0))
        micro_su_s = float(getattr(worst, "scaleup_s", 0.0))
        micro_so_s = float(getattr(worst, "scaleout_s", 0.0))

        implied_su_bw = (micro_su_bytes / micro_su_s) if micro_su_s > 0 else 0.0  # bytes/s
        implied_so_bw = (micro_so_bytes / micro_so_s) if micro_so_s > 0 else 0.0  # bytes/s

        gpu_prof = GPU_PROFILES[_normalize_gpu_sku(gpu.gpu_sku)]
        prof_su_bw = float(getattr(gpu_prof, "scaleup_bandwidth", 0.0))
        prof_so_bw = float(getattr(gpu_prof, "scaleout_bandwidth", 0.0))

        notes = []
        notes.append(
            "DBG_BW_USED: "
            f"implied_su_bw={implied_su_bw/1e9:.2f}GB/s prof_su_bw={prof_su_bw/1e9:.2f}GB/s, "
            f"implied_so_bw={implied_so_bw/1e9:.2f}GB/s prof_so_bw={prof_so_bw/1e9:.2f}GB/s"
            f"comm_end={comm_end:.2f}s compute_end={compute_end:.2f}s"

        )
        # ===== END DBG =====

        # micro-step critical-path attribution
        if comm_end > compute_end:
            micro_comm_critical = micro_time
            micro_compute_critical = 0.0
            micro_scaleup_critical = micro_time if float(getattr(worst, "scaleup_s", 0.0)) >= float(getattr(worst, "scaleout_s", 0.0)) else 0.0
            micro_scaleout_critical = micro_time if float(getattr(worst, "scaleout_s", 0.0)) > float(getattr(worst, "scaleup_s", 0.0)) else 0.0
        else:
            micro_comm_critical = 0.0
            micro_compute_critical = micro_time
            micro_scaleup_critical = 0.0
            micro_scaleout_critical = 0.0

        # grad update (per-step once)
        grad_update_s = float(getattr(worst, "grad_update_s", 0.0))
        grad_su_bytes = float(getattr(worst, "grad_update_scaleup_bytes", 0.0))
        grad_so_bytes = float(getattr(worst, "grad_update_scaleout_bytes", 0.0))
        grad_total = grad_su_bytes + grad_so_bytes
        if grad_total > 0:
            grad_su_time = grad_update_s * (grad_su_bytes / grad_total)
            grad_so_time = grad_update_s * (grad_so_bytes / grad_total)
        else:
            grad_su_time = grad_so_time = 0.0

        # lane comm time (optimizer-step attribution estimate)
        comm_scaleup_s = micro_factor * micro_scaleup_critical + grad_su_time
        comm_scaleout_s = micro_factor * micro_scaleout_critical + grad_so_time

        comm_total_s = micro_factor * micro_comm_critical + grad_update_s
        compute_total_s = micro_factor * micro_compute_critical

        ratio_comm = min(1.0, comm_total_s / denom)
        ratio_compute = min(1.0, compute_total_s / denom)

        # HBM：更像“memory-bound 指标”，用 hbm_end/compute_end 表示（0..1）
        ratio_hbm = float(getattr(worst, "hbm_s", 0.0)) / max(compute_end, 1e-12)
    else:
        ratio_compute = ratio_hbm = ratio_comm = 0.0
        comm_scaleup_s = comm_scaleout_s = 0.0


    return Tab2Response(
        comm_bytes_per_rank_tb=float(_bytes_to_tb(comm_bytes_rank0)),
        comm_by_collective_tb={k: float(_bytes_to_tb(v)) for k, v in comm_by_collective.items()},
        comm_time_scaleup_s=float(comm_scaleup_s),
        comm_time_scaleout_s=float(comm_scaleout_s),
        step_time=float(step_time),
        time_unit=unit,
        step_time_s=float(step_time_s_eff),
        step_gpu_hours=int(step_gpu_hours),
        Total_gpu_hours=float(Total_gpu_hours),
        ratio_compute=float(ratio_compute),
        ratio_hbm=float(ratio_hbm),
        ratio_comm=float(ratio_comm),
        notes=notes,
        Total_days=float(Total_days),
    )
    
