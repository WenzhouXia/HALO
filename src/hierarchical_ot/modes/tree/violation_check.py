# hierarchical_ot/strategies/violation_check.py
"""
Constraint Violation Check (HALO 迁移)
检测当前 dual solution 是否有违反约束的边被遗漏。
"""

import numpy as np
import logging
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from typing import Tuple, Optional, Literal, Dict, Any

from .logger import tree_log

logger = logging.getLogger(__name__)


def _normalize_cost_type(cost_type: str) -> str:
    """
    CN: 统一不同写法的 cost type 到 tree 模式内部使用的标准名称。
    EN: Normalize variant cost-type names to the canonical names used by tree mode.
    """
    c = str(cost_type).strip().upper()
    if c in {"L2", "L2^2", "L2SQ", "SQEUCLIDEAN", "SQ_EUCLIDEAN", "EUCLIDEAN"}:
        return "L2"
    if c in {"L1"}:
        return "L1"
    if c in {"LINF", "L_INF"}:
        return "LINF"
    return c


def check_constraint_violations(
    x_dual: np.ndarray,
    level_s,
    level_t,
    cost_type: str = "L2",
    p: float = 2.0,
    eps: float = 0.0,
    max_keep: Optional[int] = None,
    method: Literal['cpu', 'gpu', 'gpu_approx', 'auto', 'cupy'] = 'auto',
    sampled_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 按指定方法检查当前 tree dual 解中的约束违反并返回违反边及其 slack。
    EN: Check constraint violations in the current tree dual solution and return violating edges with their slack.
    """
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    
    # 拆分 u, v
    u = x_dual[:n_s].astype(np.float32)
    v = x_dual[n_s:].astype(np.float32)
    
    method_norm = str(method).lower()
    if method_norm in {"cupy", "gpu_approx", "approx"}:
        method_norm = "gpu_approx"
    elif method_norm in {"gpu", "gpu_full", "full_gpu"}:
        method_norm = "gpu"

    # 选择方法
    if method_norm == 'auto':
        if HAS_CUPY and n_s * n_t > 1_000_000:
            method_norm = 'gpu_approx'
        else:
            method_norm = 'cpu'

    if method_norm == 'gpu' and HAS_CUPY:
        return _check_violations_cupy_full(u, v, level_s, level_t, cost_type, p, eps, max_keep)
    if method_norm == 'gpu_approx' and HAS_CUPY:
        return _check_violations_cupy_sampled(
            u=u,
            v=v,
            level_s=level_s,
            level_t=level_t,
            cost_type=cost_type,
            p=p,
            eps=eps,
            max_keep=max_keep,
            sampled_config=sampled_config,
        )
    return _check_violations_cpu(u, v, level_s, level_t, cost_type, p, eps, max_keep)


def _check_violations_cpu(
    u: np.ndarray,
    v: np.ndarray,
    level_s,
    level_t,
    cost_type: str,
    p: float,
    eps: float,
    max_keep: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 在 CPU 上精确计算违反边，并按 slack 从大到小返回 top-K。
    EN: Compute violating edges exactly on CPU and return top-K entries sorted by slack.
    """
    X = np.asarray(level_s.points, dtype=np.float32)
    Y = np.asarray(level_t.points, dtype=np.float32)
    n_t = Y.shape[0]

    diff = X[:, None, :] - Y[None, :, :]
    cost_type_upper = _normalize_cost_type(cost_type)
    if cost_type_upper in ("L2", "SQEUCLIDEAN"):
        C = np.sum(diff * diff, axis=2)
    elif cost_type_upper == "L1":
        C = np.sum(np.abs(diff), axis=2)
    elif cost_type_upper == "LINF":
        C = np.max(np.abs(diff), axis=2)
    else:
        raise NotImplementedError(f"Unsupported cost_type={cost_type}")

    slack = u[:, None] + v[None, :] - C
    mask = slack > eps
    if not np.any(mask):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    idx_i, idx_j = np.where(mask)
    slack_pos = slack[idx_i, idx_j].astype(np.float32, copy=False)
    keep_1d = idx_i.astype(np.int64) * np.int64(n_t) + idx_j.astype(np.int64)

    if max_keep is not None and slack_pos.size > max_keep:
        top_idx = np.argpartition(-slack_pos, max_keep - 1)[:max_keep]
        keep_1d = keep_1d[top_idx]
        slack_pos = slack_pos[top_idx]

    order = np.argsort(-slack_pos)
    return keep_1d[order], slack_pos[order]


def _check_violations_cupy_full(
    u: np.ndarray,
    v: np.ndarray,
    level_s,
    level_t,
    cost_type: str,
    p: float,
    eps: float,
    max_keep: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 在 GPU 上全量精确检查 tree 约束违反。
    EN: Check tree constraint violations exactly on GPU using the full dense computation.
    """
    if not HAS_CUPY:
        return _check_violations_cpu(u, v, level_s, level_t, cost_type, p, eps, max_keep)
    
    n_s = len(u)
    n_t = len(v)
    
    # 移动到 GPU
    u_gpu = cp.asarray(u)
    v_gpu = cp.asarray(v)
    X_gpu = cp.asarray(level_s.points)
    Y_gpu = cp.asarray(level_t.points)
    
    # 计算成本矩阵 (n_s, n_t)
    # 使用广播: (n_s, 1, dim) - (1, n_t, dim) = (n_s, n_t, dim)
    diff = X_gpu[:, None, :] - Y_gpu[None, :, :]
    
    cost_type_upper = _normalize_cost_type(cost_type)
    if cost_type_upper in ('L2', 'SQEUCLIDEAN'):
        C_gpu = cp.sum(diff ** 2, axis=2)
    elif cost_type_upper == 'L1':
        C_gpu = cp.sum(cp.abs(diff), axis=2)
    elif cost_type_upper == 'LINF':
        C_gpu = cp.max(cp.abs(diff), axis=2)
    else:
        C_gpu = cp.sum(diff ** 2, axis=2)
    
    # 计算约束: u + v^T - C > eps
    lhs = u_gpu[:, None] + v_gpu[None, :]
    rhs = C_gpu + eps
    
    # 找出违反
    violate_mask = lhs > rhs
    
    if not cp.any(violate_mask):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    
    # 获取违反的索引
    violate_idx = cp.where(violate_mask)
    i_idx = violate_idx[0].astype(cp.int64)
    j_idx = violate_idx[1].astype(cp.int64)
    
    # 计算 1D 索引
    keep_1d = i_idx * n_t + j_idx
    
    # 计算 slack
    slack = lhs[violate_mask] - C_gpu[violate_mask]
    
    # 限制数量
    if max_keep is not None and len(keep_1d) > max_keep:
        # 取 slack 最大的 max_keep 个
        top_k_idx = cp.argpartition(-slack, max_keep - 1)[:max_keep]
        keep_1d = keep_1d[top_k_idx]
        slack = slack[top_k_idx]

    # HALO 对齐：按 slack 从大到小排序
    order = cp.argsort(-slack)
    keep_1d = keep_1d[order]
    slack = slack[order]

    # 转回 CPU
    return cp.asnumpy(keep_1d), cp.asnumpy(slack)


def check_constraint_violations_sampled_adaptive_submatrix(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    max_keep: int,
    eps: float = 0.0,
    cost_type: str = "L2",
    p: int = 2,
    n_X_batch: int = 2048,
    n_Y_batch: int = 2048,
    max_batches: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 通过自适应子矩阵采样近似寻找违反约束的边。
    EN: Approximate violating edges with adaptive submatrix sampling.
    """
    if not HAS_CUPY:
        return np.array([], dtype=np.int64), np.empty(0, dtype=np.float32)

    N_X, _ = nodes_X.shape
    N_Y, _ = nodes_Y.shape

    cost_type_norm = _normalize_cost_type(cost_type)
    use_L2_fast = (cost_type_norm in {"L2", "SQEUCLIDEAN"}) or (cost_type_norm == "LP" and p == 2)
    if not use_L2_fast:
        raise NotImplementedError(
            f"[CheckVio-Sampled-Adaptive] 仅支持 L2/Lp(p=2)，收到 cost_type={cost_type}, p={p}"
        )

    rng = np.random.default_rng(seed=seed)
    n_X_batch = int(min(max(1, n_X_batch), N_X))
    n_Y_batch = int(min(max(1, n_Y_batch), N_Y))

    v_full = x_solution[N_X:]
    v_gpu_full = cp.asarray(v_full, dtype=cp.float32)
    Y_gpu_full = cp.asarray(nodes_Y, dtype=cp.float32)
    Y_norm_sq_T_full = cp.sum(Y_gpu_full ** 2, axis=1, keepdims=True).T

    all_vio_slacks_list = []
    all_vio_indices_list = []
    global_min_kept_slack = float(eps)
    total_batches = 0

    for _ in range(max_batches):
        total_batches += 1

        rows_idx = rng.choice(N_X, size=n_X_batch, replace=False)
        cols_idx = rng.choice(N_Y, size=n_Y_batch, replace=False)

        Y_sub = Y_gpu_full[cols_idx]
        v_sub = v_gpu_full[cols_idx]
        Y_norm_sq_T_sub = Y_norm_sq_T_full[:, cols_idx]

        u_batch_gpu = cp.asarray(x_solution[rows_idx], dtype=cp.float32)
        X_batch_gpu = cp.asarray(nodes_X[rows_idx], dtype=cp.float32)

        X_norm_sq = cp.sum(X_batch_gpu ** 2, axis=1, keepdims=True)
        C_batch_gpu = X_norm_sq + Y_norm_sq_T_sub - 2.0 * (X_batch_gpu @ Y_sub.T)
        C_batch_gpu = cp.maximum(C_batch_gpu, 0)

        S_batch_gpu = u_batch_gpu[:, None] + v_sub[None, :] - C_batch_gpu
        del C_batch_gpu, u_batch_gpu, X_batch_gpu

        vio_mask = S_batch_gpu > global_min_kept_slack
        vio_slacks = S_batch_gpu[vio_mask]
        if vio_slacks.size == 0:
            del S_batch_gpu, vio_mask, vio_slacks
            continue

        r_idx, c_idx = cp.where(vio_mask)
        rows_idx_gpu = cp.asarray(rows_idx, dtype=cp.int64)
        cols_idx_gpu = cp.asarray(cols_idx, dtype=cp.int64)
        global_i_indices = rows_idx_gpu[r_idx]
        global_j_indices = cols_idx_gpu[c_idx]
        vio_indices_flat = global_i_indices * N_Y + global_j_indices

        all_vio_slacks_list.append(vio_slacks)
        all_vio_indices_list.append(vio_indices_flat)

        del S_batch_gpu, vio_mask, vio_slacks, vio_indices_flat, r_idx, c_idx, rows_idx_gpu, cols_idx_gpu

        g_slk = cp.concatenate(all_vio_slacks_list)
        g_idx = cp.concatenate(all_vio_indices_list)

        if g_slk.size > max_keep:
            kth_slack_val = cp.partition(-g_slk, max_keep - 1)[max_keep - 1]
            global_min_kept_slack = float(max(eps, -kth_slack_val))

            mask_topk = g_slk >= global_min_kept_slack
            g_slk = g_slk[mask_topk]
            g_idx = g_idx[mask_topk]

        all_vio_slacks_list = [g_slk]
        all_vio_indices_list = [g_idx]

        if g_slk.size >= max_keep:
            break

    if not all_vio_indices_list:
        logger.debug(
            "[CheckVio-Sampled-Adaptive] N_X=%s, N_Y=%s, batches=%s, violations=0",
            N_X,
            N_Y,
            total_batches,
        )
        return np.array([], dtype=np.int64), np.float32(0.0)

    g_slk = all_vio_slacks_list[0]
    g_idx = all_vio_indices_list[0]
    idx_sort = cp.argsort(-g_slk)
    if idx_sort.size > max_keep:
        idx_sort = idx_sort[:max_keep]
    g_slk = g_slk[idx_sort]
    g_idx = g_idx[idx_sort]

    slack_arr = cp.asnumpy(g_slk).astype(np.float32)
    keep_add = cp.asnumpy(g_idx).astype(np.int64)

    logger.debug(
        "[CheckVio-Sampled-Adaptive] N_X=%s, N_Y=%s, batch=(%sx%s), batches_used=%s, violations=%s, max_slack=%.2e",
        N_X,
        N_Y,
        n_X_batch,
        n_Y_batch,
        total_batches,
        len(keep_add),
        float(slack_arr[0]) if len(slack_arr) > 0 else float("nan"),
    )
    return keep_add, slack_arr


def _check_violations_cupy_sampled(
    u: np.ndarray,
    v: np.ndarray,
    level_s,
    level_t,
    cost_type: str,
    p: float,
    eps: float,
    max_keep: Optional[int],
    sampled_config: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 封装基于 CuPy 采样近似的 violation 检查流程并处理资源释放。
    EN: Wrap the CuPy sampled-approximate violation check and handle resource cleanup.
    """
    if not HAS_CUPY:
        return _check_violations_cpu(u, v, level_s, level_t, cost_type, p, eps, max_keep)

    n_s = len(u)
    n_t = len(v)
    max_keep_local = max_keep
    if max_keep_local is None:
        max_keep_local = int(min(n_s * n_t, 100_000))

    cfg = sampled_config or {}
    n_X_batch = int(cfg.get("n_X_batch", cfg.get("n_x_batch", 2048)))
    n_Y_batch = int(cfg.get("n_Y_batch", cfg.get("n_y_batch", 2048)))
    max_batches = int(cfg.get("max_batches", 100))
    seed = int(cfg.get("seed", 42))

    try:
        keep_add, slack_arr = check_constraint_violations_sampled_adaptive_submatrix(
            x_solution=np.concatenate([u, v]),
            nodes_X=np.asarray(level_s.points, dtype=np.float32),
            nodes_Y=np.asarray(level_t.points, dtype=np.float32),
            max_keep=max_keep_local,
            eps=eps,
            cost_type=cost_type,
            p=int(p),
            n_X_batch=n_X_batch,
            n_Y_batch=n_Y_batch,
            max_batches=max_batches,
            seed=seed,
        )
        return keep_add, slack_arr
    except Exception as exc:
        raise RuntimeError(
            "Sampled GPU violation check failed and no fallback path is allowed in strict mode."
        ) from exc
    finally:
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()


def expand_active_set_with_violations(
    keep_current: np.ndarray,
    y_current: np.ndarray,
    violation_keep: np.ndarray,
    violation_slack: np.ndarray,
    max_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 将 violation 边并入当前 active set，并在需要时按大小限制裁剪。
    EN: Merge violation edges into the current active set and trim by size when needed.
    """
    if len(violation_keep) == 0:
        return keep_current.copy(), y_current.copy()
    
    # 合并并去重
    combined_keep = np.concatenate([keep_current, violation_keep])
    combined_y = np.concatenate([y_current, np.zeros(len(violation_keep), dtype=np.float32)])
    
    # 去重
    unique_keep, unique_idx = np.unique(combined_keep, return_index=True)
    unique_y = combined_y[unique_idx]
    
    # 限制大小
    if max_size is not None and len(unique_keep) > max_size:
        # 保留原活跃集 + slack 最大的部分 violation
        old_mask = np.isin(unique_keep, keep_current)
        n_old = np.sum(old_mask)
        n_new = min(max_size - n_old, len(violation_keep))
        
        if n_new > 0:
            # 获取 slack
            slack_dict = dict(zip(violation_keep, violation_slack))
            slack_arr = np.array([slack_dict.get(k, 0.0) for k in unique_keep])
            
            # 保留旧集 + top slack 的新边
            top_idx = np.argpartition(-slack_arr, max_size - 1)[:max_size]
            unique_keep = unique_keep[top_idx]
            unique_y = unique_y[top_idx]
        else:
            # 只保留旧集
            unique_keep = unique_keep[old_mask]
            unique_y = unique_y[old_mask]
    
    return unique_keep, unique_y


def apply_violation_check(solver, **kwargs):
    """
    CN: 执行 tree 模式的 violation check，并把新违反边并入 keep。
    EN: Run the tree-mode violation check and merge new violating edges into keep.
    """
    x_dual = kwargs["x_dual"]
    level_s = kwargs["level_s"]
    level_t = kwargs["level_t"]
    keep = np.asarray(kwargs["keep"], dtype=np.int64)
    cost_type = kwargs["cost_type"]
    ifcheck = kwargs["ifcheck"]
    vd_thr = kwargs["vd_thr"]
    check_method = kwargs["check_method"]
    sampled_config = kwargs.get("sampled_config")
    if not ifcheck:
        return keep

    n_s = len(level_s.points)
    n_t = len(level_t.points)
    max_keep = int(vd_thr * (n_s + n_t)) if vd_thr is not None else None
    violation_keep, _ = check_constraint_violations(
        x_dual,
        level_s,
        level_t,
        cost_type=cost_type,
        eps=0.0,
        max_keep=max_keep,
        method=check_method,
        sampled_config=sampled_config,
    )
    if violation_keep.size == 0:
        tree_log(solver, "    [Check] found: 0")
        return keep

    if keep.size > 0:
        new_mask = ~np.isin(violation_keep, keep)
        violation_new = violation_keep[new_mask]
    else:
        violation_new = violation_keep

    if violation_new.size == 0:
        tree_log(solver, f"    [Check] found: {len(violation_keep)} (all already in active set)")
        return keep

    keep_new = np.unique(np.concatenate([keep, violation_new.astype(np.int64)]))
    tree_log(solver, f"    [Check] found: {len(violation_keep)} ({len(violation_new)} new)")
    tree_log(solver, f"    [Check] expanding active set: {len(keep)} -> {len(keep_new)}")
    return keep_new
