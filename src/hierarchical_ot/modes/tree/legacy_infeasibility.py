import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None
import math
import time


def _cupy_available() -> bool:
    return cp is not None and bool(cp.is_available())


def dualOT_primal_infeas_pointcloud_cupy(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2,
    batch_size: int = 2048,
    use_cupy: bool = True,
) -> float:
    """
    计算基于点云的全局原始可行性:
        infeas = || max(0, u_i + v_j - C_ij) ||_2  / (1 + ||C||_2)

    支持 L1, L2, Linf, Lp (1<p<∞)。
    其中 Lp 且 p=2 走 L2 快速路径。
    """

    # --- 1. backend ---
    xp = cp if (use_cupy and _cupy_available()) else np
    dtype = np.float32

    # --- 2. move data ---
    x_solution_dev = xp.asarray(x_solution, dtype=dtype)
    X = xp.asarray(nodes_X, dtype=dtype)
    Y = xp.asarray(nodes_Y, dtype=dtype)

    N_X = X.shape[0]
    N_Y = Y.shape[0]

    u = x_solution_dev[:N_X]
    v = x_solution_dev[N_X:]

    # === 3. 判断 cost 类型 ===
    # L2 快速路径（平方距离）
    use_fast_l2 = (cost_type == "L2") or (cost_type == "Lp" and p == 2)

    if use_fast_l2:
        # 预计算 ||x||² 与 ||y||²
        sum_X_sq = xp.sum(X ** 2, axis=1, keepdims=True)        # (N_X,1)
        sum_Y_sq = xp.sum(Y ** 2, axis=1, keepdims=True).T      # (1,N_Y)

    # === 4. 分块 ===
    total_infeas_sum_sq = 0.0
    total_c_norm_sq = 0.0

    n_batches_X = math.ceil(N_X / batch_size)
    n_batches_Y = math.ceil(N_Y / batch_size)

    for i in range(n_batches_X):
        i0 = i * batch_size
        i1 = min(i0 + batch_size, N_X)
        Xb = X[i0:i1]
        ub = u[i0:i1]

        if use_fast_l2:
            sum_Xb = sum_X_sq[i0:i1]

        for j in range(n_batches_Y):
            j0 = j * batch_size
            j1 = min(j0 + batch_size, N_Y)
            Yb = Y[j0:j1]
            vb = v[j0:j1]

            # -------------------------------------
            #  C_ij 计算（分类型）
            # -------------------------------------
            if use_fast_l2:
                # L2 squared: ||x||² + ||y||² - 2 x^T y
                dot_tile = Xb @ Yb.T
                sum_Yb = sum_Y_sq[:, j0:j1]         # (1,B)
                C = sum_Xb + sum_Yb - 2 * dot_tile
                xp.maximum(C, 0, out=C)             # 非负

            else:
                # 通用 diff （B,B,D）
                diff = Xb[:, None, :] - Yb[None, :, :]
                abs_diff = xp.abs(diff)

                # ---- L1 ----
                if cost_type == "L1":
                    C = xp.sum(abs_diff, axis=2)

                # ---- L∞ ----
                elif cost_type == "Linf":
                    C = xp.max(abs_diff, axis=2)

                # ---- Lp (1<p<∞) ----
                elif cost_type == "Lp":
                    if p == 1:
                        C = xp.sum(abs_diff, axis=2)
                    elif p == xp.inf or p == float("inf"):
                        C = xp.max(abs_diff, axis=2)
                    else:
                        C = xp.sum(abs_diff ** p, axis=2)
                else:
                    raise NotImplementedError(f"Unknown cost_type = {cost_type}")

                xp.maximum(C, 0, out=C)

                del diff, abs_diff

            # -------------------------------------
            #   u+v - C
            # -------------------------------------
            uv = ub[:, None] + vb[None, :]
            viol = uv - C
            xp.maximum(viol, 0, out=viol)

            # infeas 部分
            viol_sq = viol ** 2
            total_infeas_sum_sq += xp.sum(viol_sq)

            # cost 范数部分
            C_sq = C ** 2
            total_c_norm_sq += xp.sum(C_sq)

            del viol, viol_sq, C, C_sq

    # === 5. 回到 CPU ===
    if xp is cp:
        total_infeas_sum_sq = float(total_infeas_sum_sq.get())
        total_c_norm_sq = float(total_c_norm_sq.get())
    else:
        total_infeas_sum_sq = float(total_infeas_sum_sq)
        total_c_norm_sq = float(total_c_norm_sq)

    infeas_norm = math.sqrt(total_infeas_sum_sq)
    C_norm = math.sqrt(total_c_norm_sq)

    return infeas_norm / (1 + C_norm)

def _compute_row_contribs_for_rows(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    idx_rows: np.ndarray,
    batch_size_rows: int = 1024,
    batch_size_Y: int = 4096,
    use_cupy: bool = True,
):
    """
    给定若干行 idx_rows，计算每行的:
      S_{v,i} = sum_j viol_{ij}^2
      S_{c,i} = sum_j C_{ij}^2

    返回:
      row_v_sq: (len(idx_rows),)
      row_c_sq: (len(idx_rows),)
    """
    N_X = nodes_X.shape[0]
    N_Y = nodes_Y.shape[0]

    u_full = x_solution[:N_X]
    v_full = x_solution[N_X:]

    X_sub = nodes_X[idx_rows]      # (R, D)
    u_sub = u_full[idx_rows]       # (R,)
    Y_full = nodes_Y               # (N_Y, D)
    v_full = v_full                # (N_Y,)

    xp = cp if (use_cupy and _cupy_available()) else np
    dtype = np.float32

    X = xp.asarray(X_sub, dtype=dtype)
    Y = xp.asarray(Y_full, dtype=dtype)
    u = xp.asarray(u_sub, dtype=dtype)
    v = xp.asarray(v_full, dtype=dtype)

    n_rows = X.shape[0]
    n_Ys   = Y.shape[0]

    sum_X_sq = xp.sum(X ** 2, axis=1, keepdims=True)    # (R, 1)
    sum_Y_sq = xp.sum(Y ** 2, axis=1, keepdims=True).T  # (1, N_Y)

    n_batches_rows = math.ceil(n_rows / batch_size_rows)
    n_batches_Y    = math.ceil(n_Ys   / batch_size_Y)

    row_v_sq = xp.zeros((n_rows,), dtype=dtype)
    row_c_sq = xp.zeros((n_rows,), dtype=dtype)

    for bx in range(n_batches_rows):
        i0 = bx * batch_size_rows
        i1 = min(i0 + batch_size_rows, n_rows)
        Bx = i1 - i0

        Xb = X[i0:i1]
        ub = u[i0:i1]
        sum_Xb = sum_X_sq[i0:i1]

        v_acc = xp.zeros((Bx,), dtype=dtype)
        c_acc = xp.zeros((Bx,), dtype=dtype)

        for by in range(n_batches_Y):
            j0 = by * batch_size_Y
            j1 = min(j0 + batch_size_Y, n_Ys)

            Yb = Y[j0:j1]
            vb = v[j0:j1]
            sum_Yb = sum_Y_sq[:, j0:j1]

            dot = Xb @ Yb.T
            C = sum_Xb + sum_Yb - 2.0 * dot
            xp.maximum(C, 0, out=C)

            c_acc += xp.sum(C ** 2, axis=1)

            uv = ub[:, None] + vb[None, :]
            viol = uv - C
            xp.maximum(viol, 0, out=viol)
            v_acc += xp.sum(viol ** 2, axis=1)

        row_v_sq[i0:i1] = v_acc
        row_c_sq[i0:i1] = c_acc

    if xp is cp:
        row_v_sq = row_v_sq.get()
        row_c_sq = row_c_sq.get()

    return row_v_sq.astype(np.float64), row_c_sq.astype(np.float64)

def dualOT_primal_infeas_L2_cupy_rows_adaptive(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    target_rel_err: float = 0.05,   # 目标相对误差，比如 5%
    min_rows: int = 512,
    max_rows: int = 16384,
    batch_rows: int = 512,
    batch_size_rows: int = 1024,
    batch_size_Y: int = 4096,
    alpha: float = 2.0,             # importance 权重 |u|^alpha
    eps: float = 1e-6,
    use_cupy: bool = True,
    rng: np.random.Generator = None,
):
    """
    自适应 importance Row Sampling 版 L2 infeas 估计：
      - 不再手调 n_X_sub
      - 根据 target_rel_err 自动决定需要多少行

    返回:
      infeas_est: float   估计的 infeas
      n_used    : int     实际使用行数
      rel_se_v  : float   S_v 的估计相对标准误
      rel_se_c  : float   S_c 的估计相对标准误
    """
    N_X = nodes_X.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    max_rows = min(max_rows, N_X)

    # importance 权重: w_i ∝ (|u_i| + eps)^alpha
    u_full = x_solution[:N_X]
    w = (np.abs(u_full) + eps) ** alpha
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        # 极端情况：所有 u 为 0，就退回均匀
        p = np.full(N_X, 1.0 / N_X)
    else:
        p = w / w_sum

    # 在线估计 Z_v = S_v_i / p_i, Z_c = S_c_i / p_i 的均值和方差
    n = 0
    mean_v = 0.0
    M2_v = 0.0
    mean_c = 0.0
    M2_c = 0.0

    rel_se_v = float("inf")
    rel_se_c = float("inf")

    while True:
        # 本批要新增多少行
        m = min(batch_rows, max_rows - n)
        if m <= 0:
            break

        # 按概率 p 抽样行（允许重复）
        idx_rows = rng.choice(N_X, size=m, replace=True, p=p)
        p_rows = p[idx_rows]  # 每个样本行的抽样概率

        # 计算每行的 S_v_i, S_c_i
        row_v_sq, row_c_sq = _compute_row_contribs_for_rows(
            x_solution,
            nodes_X,
            nodes_Y,
            idx_rows,
            batch_size_rows=batch_size_rows,
            batch_size_Y=batch_size_Y,
            use_cupy=use_cupy,
        )

        Z_v_batch = row_v_sq / p_rows
        Z_c_batch = row_c_sq / p_rows

        # Welford 在线更新
        for zv, zc in zip(Z_v_batch, Z_c_batch):
            n += 1
            # v
            delta_v = zv - mean_v
            mean_v += delta_v / n
            M2_v += delta_v * (zv - mean_v)
            # c
            delta_c = zc - mean_c
            mean_c += delta_c / n
            M2_c += delta_c * (zc - mean_c)

        if n >= min_rows and n > 1:
            var_v = M2_v / (n - 1)
            var_c = M2_c / (n - 1)
            se_v = math.sqrt(var_v / n)
            se_c = math.sqrt(var_c / n)

            rel_se_v = se_v / max(abs(mean_v), 1e-30)
            rel_se_c = se_c / max(abs(mean_c), 1e-30)

            if max(rel_se_v, rel_se_c) <= target_rel_err:
                break

        if n >= max_rows:
            break

    S_v_hat = max(mean_v, 0.0)
    S_c_hat = max(mean_c, 0.0)

    viol_norm = math.sqrt(S_v_hat)
    C_norm    = math.sqrt(S_c_hat)

    infeas_est = viol_norm / (1.0 + C_norm)
    return infeas_est, n, rel_se_v, rel_se_c

def dualOT_primal_infeas_pointcloud_cupy_auto(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2,
    batch_size: int = 2048,
    use_cupy: bool = True,
    # 调试开关：是否同时跑 full 并打印相对误差 + 加速比
    debug_compare_full: bool = False,
) -> float:
    """
    自适应版本:
      - 非 L2: 直接走原版
      - 小规模问题: 直接走原版
      - 大规模 L2: 使用自适应 importance-row
        - 自动决定 min_rows, max_rows, batch_rows
        - 如采样质量太差, 自动 fallback 到 full
    """
    N_X = nodes_X.shape[0]
    N_Y = nodes_Y.shape[0]
    total_pairs = N_X * N_Y

    # 1) 非 L2 / Lp(p!=2): 不搞花活，老老实实原版
    is_L2 = (cost_type == "L2") or (cost_type == "Lp" and p == 2)
    if not is_L2:
        return dualOT_primal_infeas_pointcloud_cupy(
            x_solution, nodes_X, nodes_Y,
            cost_type=cost_type, p=p,
            batch_size=batch_size, use_cupy=use_cupy,
        )

    # 2) 小规模问题: 直接走原版
    SMALL_N = 16384         # 任意一个维度 <= 8k 就当小问题
    SMALL_PAIRS = 2e8       # N_X * N_Y <= 5e7 也不一定值得采样
    if (N_X <= SMALL_N) or (N_Y <= SMALL_N) or (total_pairs <= SMALL_PAIRS):
        return dualOT_primal_infeas_pointcloud_cupy(
            x_solution, nodes_X, nodes_Y,
            cost_type=cost_type, p=p,
            batch_size=batch_size, use_cupy=use_cupy,
        )

    # ---- 走到这里: 大规模 L2 问题，启用自适应 importance-row ----
    # 3) 自动设置采样参数 (针对大问题)
    #    比例大概取 1.5% N_X，限制在 [2048, 65536]
    frac = 0.015
    max_rows = int(min(N_X, max(2048, frac * N_X)))
    # 保证 min_rows <= max_rows
    min_rows = min(1024, max_rows // 2 if max_rows >= 4 else max_rows)
    # 每批新增行数
    batch_rows = min(512, max_rows // 4 if max_rows >= 8 else max_rows)

    # 4) 目标相对误差: 大问题用 5%
    target_rel_err = 0.05

    # 5) 先可选地算 full (debug), 再算 adaptive
    infeas_full = None
    t_full = None
    if debug_compare_full:
        t0 = time.perf_counter()
        infeas_full = dualOT_primal_infeas_pointcloud_cupy(
            x_solution, nodes_X, nodes_Y,
            cost_type=cost_type, p=p,
            batch_size=batch_size, use_cupy=use_cupy,
        )
        t1 = time.perf_counter()
        t_full = t1 - t0

    # 6) 调用自适应 importance-row（计时）
    t2 = time.perf_counter()
    infeas_adp, n_used, rel_se_v, rel_se_c = dualOT_primal_infeas_L2_cupy_rows_adaptive(
        x_solution,
        nodes_X,
        nodes_Y,
        target_rel_err=target_rel_err,
        min_rows=min_rows,
        max_rows=max_rows,
        batch_rows=batch_rows,
        batch_size_rows=1024,
        batch_size_Y=4096,
        use_cupy=use_cupy,
        rng=np.random.default_rng(42),
    )
    t3 = time.perf_counter()
    t_adp = t3 - t2

    # 7) 如果开了 debug，对比 full 并打印相对误差 + 加速比
    if debug_compare_full and infeas_full is not None and t_full is not None:
        rel_err = abs(infeas_adp - infeas_full) / max(abs(infeas_full), 1e-12)
        speedup = t_full / max(t_adp, 1e-12)
        print(
            f"[Infeas-Auto-L2] N_X={N_X}, N_Y={N_Y}, "
            f"full={infeas_full:.6e}, auto={infeas_adp:.6e}, "
            f"rel_err={rel_err:.3e}, n_used={n_used}, "
            f"est_rel_se_v={rel_se_v:.3e}, est_rel_se_c={rel_se_c:.3e}, "
            f"t_full={t_full:.3f}s, t_auto={t_adp:.3f}s, speedup={speedup:.2f}x"
        )

    # 8) 质量控制: 如果采样质量太差就 fallback
    #   - rel_se_v 特别大 (比如 > 0.5)
    #   - 或者自适应已经用满 max_rows 仍然 rel_se 很大
    if (max(rel_se_v, rel_se_c) > 0.5) and (total_pairs <= 2e8):
        # 问题规模也不是特别大, 那就老实走 full
        if infeas_full is not None:
            return infeas_full
        else:
            return dualOT_primal_infeas_pointcloud_cupy(
                x_solution, nodes_X, nodes_Y,
                cost_type=cost_type, p=p,
                batch_size=batch_size, use_cupy=use_cupy,
            )

    # 否则接受自适应估计
    return infeas_adp
