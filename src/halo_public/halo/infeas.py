import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None
import math
import time

def _cupy_available() -> bool:
    return cp is not None and bool(cp.is_available())
def dualOT_primal_infeas_pointcloud_cpu(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: int = 2
) -> float:
    
    x_solution = np.array(x_solution, dtype=np.float64)
    nodes_X = np.array(nodes_X, dtype=np.float64)
    nodes_Y = np.array(nodes_Y, dtype=np.float64)
    N_X = len(nodes_X)
    N_Y = len(nodes_Y)
    
    u = x_solution[:N_X]
    v = x_solution[N_X:]
    
    
    
    
    
    
    
    
    
    diff = nodes_X[:, np.newaxis, :] - nodes_Y[np.newaxis, :, :] 
    
    if cost_type == "L2":
        
        cost_matrix = np.sum(diff * diff, axis=2) 
        
    elif cost_type == "L1":
        
        cost_matrix = np.sum(np.abs(diff), axis=2)
        
    elif cost_type == "Linf":
        
        cost_matrix = np.max(np.abs(diff), axis=2)
        
    elif cost_type == "Lp":
        
        cost_matrix = np.sum(np.abs(diff)**p, axis=2)
        
    else:
        raise NotImplementedError(f"Cost type '{cost_type}' not supported in CPU infeas check.")

    
    
    u_plus_v = u[:, np.newaxis] + v[np.newaxis, :]

    
    violation = u_plus_v - cost_matrix
    
    
    violation[violation < 0] = 0.0
    
    
    
    infeas_sum = np.sum(violation * violation)
    
    
    
    
    
    
    
    
    c_norm = np.sqrt(np.sum(cost_matrix * cost_matrix))

    
    return math.sqrt(infeas_sum) / (1.0 + c_norm)
    

def dualOT_primal_infeas_pointcloud_cupy(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2,
    batch_size: int = 2048,
    use_cupy: bool = True,
) -> float:
    

    
    xp = cp if (use_cupy and _cupy_available()) else np
    dtype = np.float32

    
    x_solution_dev = xp.asarray(x_solution, dtype=dtype)
    X = xp.asarray(nodes_X, dtype=dtype)
    Y = xp.asarray(nodes_Y, dtype=dtype)

    N_X = X.shape[0]
    N_Y = Y.shape[0]

    u = x_solution_dev[:N_X]
    v = x_solution_dev[N_X:]

    
    
    use_fast_l2 = (cost_type == "L2") or (cost_type == "Lp" and p == 2)

    if use_fast_l2:
        
        sum_X_sq = xp.sum(X ** 2, axis=1, keepdims=True)        
        sum_Y_sq = xp.sum(Y ** 2, axis=1, keepdims=True).T      

    
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

            
            
            
            if use_fast_l2:
                
                dot_tile = Xb @ Yb.T
                sum_Yb = sum_Y_sq[:, j0:j1]         
                C = sum_Xb + sum_Yb - 2 * dot_tile
                xp.maximum(C, 0, out=C)             

            else:
                
                diff = Xb[:, None, :] - Yb[None, :, :]
                abs_diff = xp.abs(diff)

                
                if cost_type == "L1":
                    C = xp.sum(abs_diff, axis=2)

                
                elif cost_type == "Linf":
                    C = xp.max(abs_diff, axis=2)

                
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

            
            
            
            uv = ub[:, None] + vb[None, :]
            viol = uv - C
            xp.maximum(viol, 0, out=viol)

            
            viol_sq = viol ** 2
            total_infeas_sum_sq += xp.sum(viol_sq)

            
            C_sq = C ** 2
            total_c_norm_sq += xp.sum(C_sq)

            del viol, viol_sq, C, C_sq

    
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
    
    N_X = nodes_X.shape[0]
    N_Y = nodes_Y.shape[0]

    u_full = x_solution[:N_X]
    v_full = x_solution[N_X:]

    X_sub = nodes_X[idx_rows]      
    u_sub = u_full[idx_rows]       
    Y_full = nodes_Y               
    v_full = v_full                

    xp = cp if (use_cupy and _cupy_available()) else np
    dtype = np.float32

    X = xp.asarray(X_sub, dtype=dtype)
    Y = xp.asarray(Y_full, dtype=dtype)
    u = xp.asarray(u_sub, dtype=dtype)
    v = xp.asarray(v_full, dtype=dtype)

    n_rows = X.shape[0]
    n_Ys   = Y.shape[0]

    sum_X_sq = xp.sum(X ** 2, axis=1, keepdims=True)    
    sum_Y_sq = xp.sum(Y ** 2, axis=1, keepdims=True).T  

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
    target_rel_err: float = 0.05,   
    min_rows: int = 512,
    max_rows: int = 16384,
    batch_rows: int = 512,
    batch_size_rows: int = 1024,
    batch_size_Y: int = 4096,
    alpha: float = 2.0,             
    eps: float = 1e-6,
    use_cupy: bool = True,
    rng: np.random.Generator = None,
):
    
    N_X = nodes_X.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    max_rows = min(max_rows, N_X)

    
    u_full = x_solution[:N_X]
    w = (np.abs(u_full) + eps) ** alpha
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        
        p = np.full(N_X, 1.0 / N_X)
    else:
        p = w / w_sum

    
    n = 0
    mean_v = 0.0
    M2_v = 0.0
    mean_c = 0.0
    M2_c = 0.0

    rel_se_v = float("inf")
    rel_se_c = float("inf")

    while True:
        
        m = min(batch_rows, max_rows - n)
        if m <= 0:
            break

        
        idx_rows = rng.choice(N_X, size=m, replace=True, p=p)
        p_rows = p[idx_rows]  

        
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

        
        for zv, zc in zip(Z_v_batch, Z_c_batch):
            n += 1
            
            delta_v = zv - mean_v
            mean_v += delta_v / n
            M2_v += delta_v * (zv - mean_v)
            
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

def dualOT_primal_infeas_pointcloud_cupy_with_adaptive_debug(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2,
    batch_size: int = 1024,
    use_cupy: bool = True,
    use_adaptive_L2: bool = True,
    target_rel_err: float = 0.05,
) -> float:
    
    
    infeas_full = dualOT_primal_infeas_pointcloud_cupy(
        x_solution,
        nodes_X,
        nodes_Y,
        cost_type=cost_type,
        p=p,
        batch_size=batch_size,
        use_cupy=use_cupy,
    )

    
    if use_adaptive_L2 and (cost_type == "L2" or (cost_type == "Lp" and p == 2)):
        infeas_adp, n_used, rel_se_v, rel_se_c = dualOT_primal_infeas_L2_cupy_rows_adaptive(
            x_solution,
            nodes_X,
            nodes_Y,
            target_rel_err=target_rel_err,
            
            min_rows=512,
            max_rows=min(16384, nodes_X.shape[0]),
            batch_rows=512,
            batch_size_rows=1024,
            batch_size_Y=4096,
            use_cupy=use_cupy,
            rng=np.random.default_rng(42),
        )

        rel_err = abs(infeas_adp - infeas_full) / max(abs(infeas_full), 1e-12)
        print(
            f"[Infeas-Adaptive-L2] full={infeas_full:.6e}, "
            f"adaptive={infeas_adp:.6e}, rel_err={rel_err:.3e}, "
            f"n_used={n_used}, est_rel_se_v={rel_se_v:.3e}, est_rel_se_c={rel_se_c:.3e}"
        )

    
    return infeas_full

def dualOT_primal_infeas_pointcloud_cupy_auto(
    x_solution: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2,
    batch_size: int = 2048,
    use_cupy: bool = True,
    
    debug_compare_full: bool = False,
) -> float:
    
    N_X = nodes_X.shape[0]
    N_Y = nodes_Y.shape[0]
    total_pairs = N_X * N_Y

    
    is_L2 = (cost_type == "L2") or (cost_type == "Lp" and p == 2)
    if not is_L2:
        return dualOT_primal_infeas_pointcloud_cupy(
            x_solution, nodes_X, nodes_Y,
            cost_type=cost_type, p=p,
            batch_size=batch_size, use_cupy=use_cupy,
        )

    
    SMALL_N = 16384         
    SMALL_PAIRS = 2e8       
    if (N_X <= SMALL_N) or (N_Y <= SMALL_N) or (total_pairs <= SMALL_PAIRS):
        return dualOT_primal_infeas_pointcloud_cupy(
            x_solution, nodes_X, nodes_Y,
            cost_type=cost_type, p=p,
            batch_size=batch_size, use_cupy=use_cupy,
        )

    
    
    
    frac = 0.015
    max_rows = int(min(N_X, max(2048, frac * N_X)))
    
    min_rows = min(1024, max_rows // 2 if max_rows >= 4 else max_rows)
    
    batch_rows = min(512, max_rows // 4 if max_rows >= 8 else max_rows)

    
    target_rel_err = 0.05

    
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

    
    
    
    if (max(rel_se_v, rel_se_c) > 0.5) and (total_pairs <= 2e8):
        
        if infeas_full is not None:
            return infeas_full
        else:
            return dualOT_primal_infeas_pointcloud_cupy(
                x_solution, nodes_X, nodes_Y,
                cost_type=cost_type, p=p,
                batch_size=batch_size, use_cupy=use_cupy,
            )

    
    return infeas_adp
