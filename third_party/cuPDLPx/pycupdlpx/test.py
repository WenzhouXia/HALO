# # test_transport_compare.py
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

# import numpy as np
# import pycupdlpx as px
# import cvxpy as cp

# supply = np.array([40.0, 40.0, 5.0], dtype=np.float64)  # 3
# demand = np.array([30.0, 25.0, 20.0, 10.0], dtype=np.float64)  # 4
# ni, nj = len(supply), len(demand)
# m = ni + nj
# n = ni * nj

# C = np.array([
#     [4.0,  8.0,  8.0,  6.0],
#     [6.0,  4.0,  5.0,  7.0],
#     [9.0, 12.0, 13.0,  9.0],
# ], dtype=np.float64)

# indptr  = np.zeros(n + 1, dtype=np.int32)
# indices = np.zeros(2 * n, dtype=np.int32)
# data    = np.ones(2 * n, dtype=np.float64)
# col = 0
# for i in range(ni):
#     for j in range(nj):
#         indices[2*col + 0] = i
#         indices[2*col + 1] = ni + j
#         indptr[col + 1] = 2 * (col + 1)
#         col += 1

# rhs = np.concatenate([supply, demand], axis=0)
# lb  = np.zeros(n, dtype=np.float64)        # x >= 0
# nEqs = m


# solver = px.cupdlpx()
# solver.loadData_csc(indptr, indices, data, m, n, c, rhs, lb, ub, nEqs)

# params = px.make_default_params()
# params["verbose"] = True
# params["eps_optimal_relative"]  = 1e-6
# params["eps_feasible_relative"] = 1e-6
# params["l_inf_ruiz_iterations"] = 0
# params["bound_objective_rescaling"] = False

# res_px = solver.solve(params)
# x_px = np.asarray(res_px["x"], dtype=np.float64)
# pobj_px = float(res_px["primal_objective"])
# status_px = res_px["termination_reason"]

# print("\n[cuPDLPx] status:", status_px)
# print("[cuPDLPx] pobj  :", pobj_px)

# x = cp.Variable(n, nonneg=True)
# cons = []
# for i in range(ni):
#     idxs = [i*nj + j for j in range(nj)]
#     cons.append(cp.sum(x[idxs]) == supply[i])
# for j in range(nj):
#     idxs = [i*nj + j for i in range(ni)]
#     cons.append(cp.sum(x[idxs]) == demand[j])

# prob = cp.Problem(cp.Minimize(c @ x), cons)

# x_cvx = x.value.astype(np.float64)
# pobj_cvx = float(prob.value)

# print("\n[CVXPY] status:", prob.status)
# print("[CVXPY] pobj  :", pobj_cvx)

# obj_diff = abs(pobj_px - pobj_cvx)
# x_diff = np.max(np.abs(x_px - x_cvx))
# print("\n[Compare] |pobj_px - pobj_cvx| =", obj_diff)
# print("[Compare] max|x_px - x_cvx|    =", x_diff)

# def check_residuals(xv):
#     supp_res = []
#     for i in range(ni):
#         idxs = [i*nj + j for j in range(nj)]
#         supp_res.append(abs(xv[idxs].sum() - supply[i]))
#     dem_res = []
#     for j in range(nj):
#         idxs = [i*nj + j for i in range(ni)]
#         dem_res.append(abs(xv[idxs].sum() - demand[j]))
#     return max(supp_res + dem_res)

# max_res_px = check_residuals(x_px)
# max_res_cvx = check_residuals(x_cvx)

# print("\n[Residual] max equality residual (cuPDLPx):", max_res_px)
# print("[Residual] max equality residual (CVXPY)  :", max_res_cvx)

# def pretty_matrix(xv): return xv.reshape((ni, nj))
# print("\nOptimal flow matrix (cuPDLPx):\n", pretty_matrix(x_px))
# print("\nOptimal flow matrix (CVXPY):\n", pretty_matrix(x_cvx))

# print("sum(x_px) =", float(np.sum(x_px)))
# print("sum(supply) =", float(np.sum(supply)))
# print("sum(demand) =", float(np.sum(demand)))

# def max_residual_csc(indptr, indices, data, x, rhs):
#     m = rhs.shape[0]
#     y = np.zeros(m, dtype=np.float64)
#     for j in range(x.shape[0]):
#         s = indptr[j]; e = indptr[j+1]
#         if s == e: continue
#         y[indices[s:e]] += data[s:e] * x[j]
#     return float(np.max(np.abs(y - rhs)))
# print("max_res after unscale:", max_residual_csc(indptr, indices, data, res_px["x"], rhs))


# px.reset_cuda_device()
# print("\nCleanup done.")

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

import numpy as np
import scipy.sparse as sp
import pycupdlpx as px

#                        A=[0 1], b=[2], x>=0
#                          [1 1]   [3]
A = sp.csc_matrix(np.array([[1.,0.],
                            [0.,1.],
                            [1.,1.]], dtype=np.float64))
m, n = A.shape
c = np.array([1., 2.], dtype=np.float64)         # minimize c^T x
rhs = np.array([1., 2., 3.], dtype=np.float64)   # Ax = rhs
lb  = np.zeros(n, dtype=np.float64)
ub  = np.full(n, np.inf, dtype=np.float64)
nEqs = m

px.set_free_mode(1)
model = px.cupdlpx()
model.loadData(A, c, rhs, lb, ub, nEqs)

params = px.make_default_params()
params["verbose"] = True
res = model.solve(params)

print("\nstatus:", res["termination_reason"])
print("obj   :", res["primal_objective"])
print("x     :", np.array(res["x"]))
