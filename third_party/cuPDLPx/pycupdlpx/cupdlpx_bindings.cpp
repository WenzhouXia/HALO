// pycupdlpx / cupdlpx_bindings.cpp  (CSC direct, safe device->host copy, unscale, atexit cleanup)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>

#include "../cupdlpx/struct.h"
#include "../cupdlpx/cupdlpx.h"

namespace py = pybind11;
extern "C"
{
  pdhg_solver_state_t *optimize(pdhg_parameters_t *params, lp_problem_t *problem);
  void pdhg_solver_state_free(pdhg_solver_state_t *state);
  void print_solver_summary(const pdhg_solver_state_t *solver_state);
  size_t get_last_solver_peak_mem();
}

static void copy_to_host(double *h_dst, const double *src, size_t n)
{
  if (!h_dst || !src || n == 0)
    return;

  cudaPointerAttributes attr;
  memset(&attr, 0, sizeof(attr));
  cudaError_t e = cudaPointerGetAttributes(&attr, (const void *)src);

#if CUDART_VERSION >= 10000
  auto is_dev = (e == cudaSuccess) && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
  auto is_dev = (e == cudaSuccess) && (attr.memoryType == cudaMemoryTypeDevice);
#endif

  if (is_dev)
  {
    cudaError_t ce = cudaMemcpy(h_dst, src, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess)
    {
      throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(ce));
    }
  }
  else
  {
    std::memcpy(h_dst, src, n * sizeof(double));
  }
}

// ====== CSC -> CSR ======
static void csc_to_csr(
    int m, int n,
    const int *csc_col_ptr, // n+1
    const int *csc_row_idx, // nnz
    const double *csc_val,  // nnz
    std::vector<int> &csr_row_ptr,
    std::vector<int> &csr_col_idx,
    std::vector<double> &csr_val)
{
  const int nnz = csc_col_ptr[n];
  csr_row_ptr.assign(m + 1, 0);
  csr_col_idx.resize(nnz);
  csr_val.resize(nnz);

  for (int k = 0; k < nnz; ++k)
  {
    int r = csc_row_idx[k];
    if (r < 0 || r >= m)
      throw std::runtime_error("CSC row index out of range");
    csr_row_ptr[r + 1]++;
  }
  for (int i = 0; i < m; ++i)
    csr_row_ptr[i + 1] += csr_row_ptr[i];

  std::vector<int> next = csr_row_ptr;
  for (int j = 0; j < n; ++j)
  {
    for (int p = csc_col_ptr[j]; p < csc_col_ptr[j + 1]; ++p)
    {
      int r = csc_row_idx[p];
      int dst = next[r]++;
      csr_col_idx[dst] = j;
      csr_val[dst] = csc_val[p];
    }
  }
}

static int g_free_mode = 1;

class CupdlpxHolder
{
public:
  CupdlpxHolder() : problem_(nullptr), m_(0), n_(0), nnz_(0) {}
  ~CupdlpxHolder()
  {
    if (problem_)
    {
      delete problem_;
      problem_ = nullptr;
    }
  }

  void loadData(py::object A,
                py::array_t<double, py::array::c_style | py::array::forcecast> c,
                py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
                py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                int nEqs)
  {
    py::module_ sp = py::module_::import("scipy.sparse");
    bool is_csc = sp.attr("isspmatrix_csc")(A).cast<bool>();
    if (!is_csc)
    {
      throw std::runtime_error("A must be a scipy.sparse.csc_matrix");
    }

    py::tuple shape = A.attr("shape").cast<py::tuple>();
    if (shape.size() != 2)
    {
      throw std::runtime_error("A.shape must be a 2-tuple");
    }
    int m = shape[0].cast<int>();
    int n = shape[1].cast<int>();

    auto indptr = A.attr("indptr").cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    auto indices = A.attr("indices").cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    auto data = A.attr("data").cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

    this->loadData_csc(indptr, indices, data, m, n, c, rhs, lb, ub, nEqs);
  }

  void loadData_csc(py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                    py::array_t<double, py::array::c_style | py::array::forcecast> data,
                    int m, int n,
                    py::array_t<double, py::array::c_style | py::array::forcecast> c,
                    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                    int nEqs)
  {
    if (m <= 0 || n <= 0)
      throw std::runtime_error("m,n must be positive.");
    if (indptr.ndim() != 1 || (int)indptr.size() != n + 1)
      throw std::runtime_error("indptr must have length n+1.");
    if (c.ndim() != 1 || (int)c.size() != n)
      throw std::runtime_error("c must have length n.");
    if (rhs.ndim() != 1 || (int)rhs.size() != m)
      throw std::runtime_error("rhs must have length m.");
    if (lb.ndim() != 1 || (int)lb.size() != n)
      throw std::runtime_error("lb must have length n.");
    if (ub.ndim() != 1 || (int)ub.size() != n)
      throw std::runtime_error("ub must have length n.");
    const int nnz = indptr.at(n);
    if (indices.ndim() != 1 || (int)indices.size() != nnz)
      throw std::runtime_error("indices length mismatch.");
    if (data.ndim() != 1 || (int)data.size() != nnz)
      throw std::runtime_error("data length mismatch.");
    if (nEqs < 0 || nEqs > m)
      throw std::runtime_error("nEqs out of range.");

    {
      auto ip = indptr.unchecked<1>();
      for (int j = 0; j < n; ++j)
        if (ip(j) > ip(j + 1))
          throw std::runtime_error("indptr must be non-decreasing.");
      auto ridx = indices.unchecked<1>();
      for (int k = 0; k < nnz; ++k)
        if (ridx(k) < 0 || ridx(k) >= m)
          throw std::runtime_error("row index out of range.");
    }

    m_ = m;
    n_ = n;
    nnz_ = nnz;

    var_lb_.assign((double *)lb.data(), (double *)lb.data() + n_);
    var_ub_.assign((double *)ub.data(), (double *)ub.data() + n_);
    obj_.assign((double *)c.data(), (double *)c.data() + n_);
    rhs_.assign((double *)rhs.data(), (double *)rhs.data() + m_);

    const double INF = std::numeric_limits<double>::infinity();
    con_lb_.assign(m_, -INF);
    con_ub_.assign(m_, INF);
    for (int i = 0; i < m_; ++i)
    {
      if (i < nEqs)
      {
        con_lb_[i] = rhs_[i];
        con_ub_[i] = rhs_[i];
      }
      else
      {
        con_ub_[i] = rhs_[i];
      }
    }

    // CSC -> CSR
    {
      auto ip = indptr.unchecked<1>();
      tmp_col_ptr_.resize(n_ + 1);
      for (int j = 0; j <= n_; ++j)
        tmp_col_ptr_[j] = ip(j);

      tmp_row_idx_.assign((int *)indices.data(), (int *)indices.data() + nnz_);
      tmp_val_.assign((double *)data.data(), (double *)data.data() + nnz_);

      csc_to_csr(m_, n_, tmp_col_ptr_.data(), tmp_row_idx_.data(), tmp_val_.data(),
                 csr_row_ptr_, csr_col_idx_, csr_val_);
    }

    if (!problem_)
      problem_ = new lp_problem_t();
    problem_->num_variables = n_;
    problem_->num_constraints = m_;
    problem_->variable_lower_bound = const_cast<double *>(var_lb_.data());
    problem_->variable_upper_bound = const_cast<double *>(var_ub_.data());
    problem_->objective_vector = const_cast<double *>(obj_.data());
    problem_->objective_constant = 0.0;

    problem_->constraint_matrix_row_pointers = const_cast<int *>(csr_row_ptr_.data());
    problem_->constraint_matrix_col_indices = const_cast<int *>(csr_col_idx_.data());
    problem_->constraint_matrix_values = const_cast<double *>(csr_val_.data());
    problem_->constraint_matrix_num_nonzeros = nnz_;

    problem_->constraint_lower_bound = const_cast<double *>(con_lb_.data());
    problem_->constraint_upper_bound = const_cast<double *>(con_ub_.data());
  }

  // py::dict solve(py::dict user_params)
  // {
  //   if (!problem_)
  //     throw std::runtime_error("No problem loaded. Call loadData_csc() first.");

  //   pdhg_parameters_t params;
  //   set_default_parameters(&params);
  //   auto pull = [&](const char *k, auto &dst)
  //   {
  //     if (user_params.contains(k))
  //       dst = user_params[k].cast<std::remove_reference_t<decltype(dst)>>();
  //   };
  //   pull("verbose", params.verbose);
  //   pull("time_sec_limit", params.termination_criteria.time_sec_limit);
  //   pull("iteration_limit", params.termination_criteria.iteration_limit);
  //   pull("eps_optimal_relative", params.termination_criteria.eps_optimal_relative);
  //   pull("eps_feasible_relative", params.termination_criteria.eps_feasible_relative);
  //   pull("eps_feasible_relative_primal", params.termination_criteria.eps_feasible_relative_primal);
  //   pull("eps_feasible_relative_dual", params.termination_criteria.eps_feasible_relative_dual);
  //   pull("eps_infeasible", params.termination_criteria.eps_infeasible);
  //   pull("l_inf_ruiz_iterations", params.l_inf_ruiz_iterations);
  //   pull("bound_objective_rescaling", params.bound_objective_rescaling);
  //   pull("has_pock_chambolle_alpha", params.has_pock_chambolle_alpha);
  //   pull("pock_chambolle_alpha", params.pock_chambolle_alpha);
  //   pull("termination_evaluation_frequency", params.termination_evaluation_frequency);
  //   pull("reflection_coefficient", params.reflection_coefficient);
  //   pull("use_dual_nnz_gate", params.termination_criteria.use_dual_nnz_gate);
  //   pull("dual_nnz_factor", params.termination_criteria.dual_nnz_factor);
  //   pull("dual_nnz_abs_tol", params.termination_criteria.dual_nnz_abs_tol);

  //   if (params.termination_criteria.use_dual_nnz_gate)
  //   {
  //     printf("Use checkSparsity, tol = %f x nCols, eps = %f\n", params.termination_criteria.dual_nnz_factor, params.termination_criteria.dual_nnz_abs_tol);
  //   }
  //   else
  //   {
  //     printf("Do Not Use checkSparsity.\n");
  //   }

  //   if (has_init_iterate_)
  //   {
  //     params.has_initial_iterate = true;
  //     if (x0_buf_.ndim() > 0 && x0_buf_.size() == n_)
  //     {
  //       params.initial_primal_unscaled = static_cast<const double *>(x0_buf_.data());
  //     }
  //     if (y0_buf_.ndim() > 0 && y0_buf_.size() == m_)
  //     {
  //       params.initial_dual_unscaled = static_cast<const double *>(y0_buf_.data());
  //     }
  //   }

  //   pdhg_solver_state_t *state = optimize(&params, problem_);
  //   if (!state)
  //     throw std::runtime_error("cuPDLPx optimize() failed.");
  //   print_solver_summary(state);
  //   std::vector<double> hx(n_), hy(m_);
  //   copy_to_host(hx.data(), state->pdhg_primal_solution, (size_t)n_);
  //   copy_to_host(hy.data(), state->pdhg_dual_solution, (size_t)m_);

  //   std::vector<double> svar(n_), scon(m_);
  //   svar.assign(n_, 1.0);
  //   scon.assign(m_, 1.0);
  //   if (state->variable_rescaling)
  //     copy_to_host(svar.data(), state->variable_rescaling, (size_t)n_);
  //   if (state->constraint_rescaling)
  //     copy_to_host(scon.data(), state->constraint_rescaling, (size_t)m_);

  //   // py::array_t<double> x(n_), y(m_);
  //   // {
  //   //   auto xb = x.mutable_unchecked<1>();
  //   //   auto yb = y.mutable_unchecked<1>();
  //   //   for (int i = 0; i < n_; ++i)
  //   //     xb(i) = hx[i] / (svar[i] == 0.0 ? 1.0 : svar[i]);
  //   //   for (int j = 0; j < m_; ++j)
  //   //     yb(j) = hy[j] / (scon[j] == 0.0 ? 1.0 : scon[j]);
  //   // }
  //   const double alpha_x = (state->constraint_bound_rescaling != 0.0)
  //                              ? state->constraint_bound_rescaling
  //                              : 1.0;
  //   const double alpha_y = (state->objective_vector_rescaling != 0.0)
  //                              ? state->objective_vector_rescaling
  //                              : 1.0;

  //   py::array_t<double> x(n_), y(m_);
  //   {
  //     auto xb = x.mutable_unchecked<1>();
  //     auto yb = y.mutable_unchecked<1>();
  //     for (int i = 0; i < n_; ++i)
  //     {
  //       const double sv = (i < (int)svar.size() && svar[i] != 0.0) ? svar[i] : 1.0;
  //       xb(i) = hx[i] / (sv * alpha_x);
  //     }
  //     for (int j = 0; j < m_; ++j)
  //     {
  //       const double sc = (j < (int)scon.size() && scon[j] != 0.0) ? scon[j] : 1.0;
  //       yb(j) = hy[j] / (sc * alpha_y);
  //     }
  //     x_cache_.assign(n_, 0.0);
  //     y_cache_.assign(m_, 0.0);
  //     for (int i = 0; i < n_; ++i)
  //       x_cache_[i] = hx[i] / ((svar[i] != 0.0 ? svar[i] : 1.0) * (alpha_x != 0.0 ? alpha_x : 1.0));
  //     for (int j = 0; j < m_; ++j)
  //       y_cache_[j] = hy[j] / ((scon[j] != 0.0 ? scon[j] : 1.0) * (alpha_y != 0.0 ? alpha_y : 1.0));
  //   }

  //   py::dict out;
  //   out["termination_reason"] = py::str(termination_reason_tToString(state->termination_reason));
  //   out["runtime_sec"] = state->cumulative_time_sec;
  //   out["iterations"] = state->total_count;
  //   out["primal_objective"] = state->primal_objective_value;
  //   out["dual_objective"] = state->dual_objective_value;
  //   out["abs_primal_res"] = state->absolute_primal_residual;
  //   out["rel_primal_res"] = state->relative_primal_residual;
  //   out["abs_dual_res"] = state->absolute_dual_residual;
  //   out["rel_dual_res"] = state->relative_dual_residual;
  //   out["abs_obj_gap"] = state->objective_gap;
  //   out["rel_obj_gap"] = state->relative_objective_gap;
  //   out["x"] = x;
  //   out["y"] = y;

  //   iters_cache_ = state->total_count;
  //   solve_time_cache_ = state->cumulative_time_sec;
  //   primal_obj_cache_ = state->primal_objective_value;
  //   dual_obj_cache_ = state->dual_objective_value;
  //   primal_feas_abs_cache_ = state->absolute_primal_residual;
  //   dual_feas_abs_cache_ = state->absolute_dual_residual;
  //   gap_abs_cache_ = fabs(state->primal_objective_value - state->dual_objective_value);
  //   primal_feas_rel_cache_ = state->relative_primal_residual;
  //   dual_feas_rel_cache_ = state->relative_dual_residual;
  //   gap_rel_cache_ = state->relative_objective_gap;
  //   // beta_cache_ = params.reflection_coefficient;

  //   switch (state->termination_reason)
  //   {
  //   case TERMINATION_REASON_OPTIMAL:
  //     saveinfo_cache_ = 1;
  //     break;
  //   case TERMINATION_REASON_TIME_LIMIT:
  //     saveinfo_cache_ = 3;
  //     break;
  //   case TERMINATION_REASON_ITERATION_LIMIT:
  //     saveinfo_cache_ = 3;
  //     break;
  //   case TERMINATION_REASON_PRIMAL_INFEASIBLE:
  //     saveinfo_cache_ = 4;
  //     break;
  //   case TERMINATION_REASON_DUAL_INFEASIBLE:
  //     saveinfo_cache_ = 4;
  //     break;
  //   default:
  //     saveinfo_cache_ = 0;
  //     break;
  //   }

  //   primal_step_cache_ = 0.0;
  //   dual_step_cache_ = 0.0;
  //   last_restart_iter_cache_ = -1;

  //   if (g_free_mode == 0 || g_free_mode == 1)
  //   {
  //     if (state)
  //       pdhg_solver_state_free(state);
  //     state = nullptr;
  //   }
  //   return out;
  // }
  py::dict solve(py::dict user_params)
  {
    if (!problem_)
      throw std::runtime_error("No problem loaded. Call loadData_csc() first.");

    using clk = std::chrono::high_resolution_clock;
    auto t_all0 = clk::now();

    bool print_summary = false;
    bool return_x = true;       // NEW
    bool return_y = true;       // NEW
    bool do_unscale = true;     // NEW

    if (user_params.contains("print_summary"))
      print_summary = user_params["print_summary"].cast<bool>();
    if (user_params.contains("return_x"))
      return_x = user_params["return_x"].cast<bool>();
    if (user_params.contains("return_y"))
      return_y = user_params["return_y"].cast<bool>();
    if (user_params.contains("unscale"))
      do_unscale = user_params["unscale"].cast<bool>();

    pdhg_parameters_t params;
    set_default_parameters(&params);

    auto pull = [&](const char *k, auto &dst)
    {
      if (user_params.contains(k))
        dst = user_params[k].cast<std::remove_reference_t<decltype(dst)>>();
    };
    pull("verbose", params.verbose);
    pull("verbose_time", params.verbose_time);
    pull("time_sec_limit", params.termination_criteria.time_sec_limit);
    pull("iteration_limit", params.termination_criteria.iteration_limit);
    pull("eps_optimal_relative", params.termination_criteria.eps_optimal_relative);
    pull("eps_feasible_relative", params.termination_criteria.eps_feasible_relative);
    pull("eps_feasible_relative_primal", params.termination_criteria.eps_feasible_relative_primal);
    pull("eps_feasible_relative_dual", params.termination_criteria.eps_feasible_relative_dual);
    pull("eps_infeasible", params.termination_criteria.eps_infeasible);
    pull("l_inf_ruiz_iterations", params.l_inf_ruiz_iterations);
    pull("bound_objective_rescaling", params.bound_objective_rescaling);
    pull("has_pock_chambolle_alpha", params.has_pock_chambolle_alpha);
    pull("pock_chambolle_alpha", params.pock_chambolle_alpha);
    pull("termination_evaluation_frequency", params.termination_evaluation_frequency);
    pull("reflection_coefficient", params.reflection_coefficient);
    pull("use_dual_nnz_gate", params.termination_criteria.use_dual_nnz_gate);
    pull("dual_nnz_factor", params.termination_criteria.dual_nnz_factor);
    pull("dual_nnz_abs_tol", params.termination_criteria.dual_nnz_abs_tol);
    // //
    pull("step_size_method", params.step_size_method);
    pull("step_size_safety", params.step_size_safety);
    pull("power_max_iterations", params.power_max_iterations);
    pull("power_tolerance", params.power_tolerance);
    pull("hybrid_refine_iterations", params.hybrid_refine_iterations);
    pull("stepsize_power_reference", params.stepsize_power_reference);
    pull("stepsize_reference_max_iterations", params.stepsize_reference_max_iterations);
    pull("stepsize_reference_tolerance", params.stepsize_reference_tolerance);
    if (params.verbose)
    {
      if (params.termination_criteria.use_dual_nnz_gate)
      {
        printf("Use checkSparsity, tol = %f x nCols, eps = %f\n",
               params.termination_criteria.dual_nnz_factor,
               params.termination_criteria.dual_nnz_abs_tol);
      }
      else
      {
        printf("Do Not Use checkSparsity.\n");
      }
    }

    if (has_init_iterate_)
    {
      params.has_initial_iterate = true;
      if (x0_buf_.ndim() > 0 && (int)x0_buf_.size() == n_)
      {
        params.initial_primal_unscaled = static_cast<const double *>(x0_buf_.data());
      }
      if (y0_buf_.ndim() > 0 && (int)y0_buf_.size() == m_)
      {
        params.initial_dual_unscaled = static_cast<const double *>(y0_buf_.data());
      }
    }

    auto t_opt0 = clk::now();
    pdhg_solver_state_t *state = optimize(&params, problem_);
    auto t_opt1 = clk::now();
    if (!state)
      throw std::runtime_error("cuPDLPx optimize() failed.");

    auto t_prn0 = clk::now();
    if (print_summary)
    {
      print_solver_summary(state);
    }
    auto t_prn1 = clk::now();

    auto t_d2h0 = clk::now();
    x_cache_.assign((size_t)n_, 0.0);
    y_cache_.assign((size_t)m_, 0.0);

    if (return_x)
      copy_to_host(x_cache_.data(), state->pdhg_primal_solution, (size_t)n_);
    if (return_y)
      copy_to_host(y_cache_.data(), state->pdhg_dual_solution, (size_t)m_);

    std::vector<double> svar, scon;
    double alpha_x = 1.0, alpha_y = 1.0;
    if (do_unscale)
    {
      if (return_x)
      {
        svar.assign((size_t)n_, 1.0);
        if (state->variable_rescaling)
          copy_to_host(svar.data(), state->variable_rescaling, (size_t)n_);
        if (state->constraint_bound_rescaling != 0.0)
          alpha_x = state->constraint_bound_rescaling;
      }
      if (return_y)
      {
        scon.assign((size_t)m_, 1.0);
        if (state->constraint_rescaling)
          copy_to_host(scon.data(), state->constraint_rescaling, (size_t)m_);
        if (state->objective_vector_rescaling != 0.0)
          alpha_y = state->objective_vector_rescaling;
      }
    }
    auto t_d2h1 = clk::now();

    auto t_uns0 = clk::now();
    if (do_unscale)
    {
      if (return_x)
      {
        const double ax = (alpha_x != 0.0 ? alpha_x : 1.0);
        for (int i = 0; i < n_; ++i)
        {
          const double sv = (i < (int)svar.size() && svar[i] != 0.0) ? svar[i] : 1.0;
          x_cache_[(size_t)i] /= (sv * ax);
        }
      }
      if (return_y)
      {
        const double ay = (alpha_y != 0.0 ? alpha_y : 1.0);
        for (int j = 0; j < m_; ++j)
        {
          const double sc = (j < (int)scon.size() && scon[j] != 0.0) ? scon[j] : 1.0;
          y_cache_[(size_t)j] /= (sc * ay);
        }
      }
    }
    auto t_uns1 = clk::now();

    py::dict out;
    out["termination_reason"] = py::str(termination_reason_tToString(state->termination_reason));
    out["runtime_sec"] = state->cumulative_time_sec;
    out["iterations"] = state->total_count;
    out["primal_objective"] = state->primal_objective_value;
    out["dual_objective"] = state->dual_objective_value;
    out["abs_primal_res"] = state->absolute_primal_residual;
    out["rel_primal_res"] = state->relative_primal_residual;
    out["abs_dual_res"] = state->absolute_dual_residual;
    out["rel_dual_res"] = state->relative_dual_residual;
    out["abs_obj_gap"] = state->objective_gap;
    out["rel_obj_gap"] = state->relative_objective_gap;
    size_t peak_bytes = get_last_solver_peak_mem();
    double peak_mib = (double)peak_bytes / (1024.0 * 1024.0);
    out["peak_gpu_mem_mib"] = peak_mib;
    // =========================================================

    auto t_wrap0 = clk::now();
    if (return_x)
    {
      // shape=(n_,), strides=(sizeof(double),)
      py::array_t<double> x({(py::ssize_t)n_}, {(py::ssize_t)sizeof(double)}, x_cache_.data(), py::none());
      out["x"] = std::move(x);
    }
    else
    {
      out["x"] = py::none();
      x_cache_.clear();
      x_cache_.shrink_to_fit();
    }
    if (return_y)
    {
      py::array_t<double> y({(py::ssize_t)m_}, {(py::ssize_t)sizeof(double)}, y_cache_.data(), py::none());
      out["y"] = std::move(y);
    }
    else
    {
      out["y"] = py::none();
      y_cache_.clear();
      y_cache_.shrink_to_fit();
    }
    auto t_wrap1 = clk::now();

    iters_cache_ = state->total_count;
    solve_time_cache_ = state->cumulative_time_sec;
    primal_obj_cache_ = state->primal_objective_value;
    dual_obj_cache_ = state->dual_objective_value;
    primal_feas_abs_cache_ = state->absolute_primal_residual;
    dual_feas_abs_cache_ = state->absolute_dual_residual;
    gap_abs_cache_ = std::fabs(state->primal_objective_value - state->dual_objective_value);
    primal_feas_rel_cache_ = state->relative_primal_residual;
    dual_feas_rel_cache_ = state->relative_dual_residual;
    gap_rel_cache_ = state->relative_objective_gap;
    switch (state->termination_reason)
    {
    case TERMINATION_REASON_OPTIMAL:
      saveinfo_cache_ = 1;
      break;
    case TERMINATION_REASON_TIME_LIMIT:
    case TERMINATION_REASON_ITERATION_LIMIT:
      saveinfo_cache_ = 3;
      break;
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
    case TERMINATION_REASON_DUAL_INFEASIBLE:
      saveinfo_cache_ = 4;
      break;
    default:
      saveinfo_cache_ = 0;
      break;
    }
    primal_step_cache_ = 0.0;
    dual_step_cache_ = 0.0;
    last_restart_iter_cache_ = -1;

    if (g_free_mode == 0 || g_free_mode == 1)
    {
      if (state)
        pdhg_solver_state_free(state);
      state = nullptr;
    }

    auto t_all1 = clk::now();

    auto ms = [](auto a, auto b)
    { return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
    py::dict timing;
    timing["optimize_ms"] = ms(t_opt0, t_opt1);
    timing["print_ms"] = ms(t_prn0, t_prn1);
    timing["d2h_ms"] = ms(t_d2h0, t_d2h1);
    timing["unscale_ms"] = ms(t_uns0, t_uns1);
    timing["wrap_numpy_ms"] = ms(t_wrap0, t_wrap1);
    timing["total_ms"] = ms(t_all0, t_all1);
    out["timing_ms"] = std::move(timing);
    // if (params.verbose_time)
    // {
    //   printf("=== timing (ms) ===\n");
    //   printf("total_ms: %lld\n", timing["total_ms"].cast<long long>());
    //   printf("  optimize_ms: %lld\n", timing["optimize_ms"].cast<long long>());
    //   printf("  print_ms: %lld\n", timing["print_ms"].cast<long long>());
    //   printf("  d2h_ms: %lld\n", timing["d2h_ms"].cast<long long>());
    //   printf("  unscale_ms: %lld\n", timing["unscale_ms"].cast<long long>());
    //   printf("  wrap_numpy_ms: %lld\n", timing["wrap_numpy_ms"].cast<long long>());
    // }
    return out;
  }

  py::dict getSolution() const
  {
    py::dict d;
    d["iters"] = iters_cache_;
    d["solve_time"] = solve_time_cache_;

    d["PrimalObj"] = primal_obj_cache_;
    d["DualObj"] = dual_obj_cache_;
    d["PrimalFeas"] = primal_feas_abs_cache_;
    d["DualFeas"] = dual_feas_abs_cache_;
    d["DualityGap"] = gap_abs_cache_;
    d["PrimalFeasRel"] = primal_feas_rel_cache_;
    d["DualFeasRel"] = dual_feas_rel_cache_;
    d["RelObjGap"] = gap_rel_cache_;

    d["PrimalFeasAvg"] = py::none();
    d["DualFeasAvg"] = py::none();
    d["DualityGapAvg"] = py::none();
    d["PrimalFeasAvgRel"] = py::none();
    d["DualFeasAvgRel"] = py::none();
    d["RelObjGapAverage"] = py::none();

    d["SaveInfo"] = saveinfo_cache_;
    d["LastRestartIter"] = last_restart_iter_cache_;
    d["PrimalStep"] = primal_step_cache_;
    d["DualStep"] = dual_step_cache_;
    d["Beta"] = beta_cache_;

    // ---------- x: always return a 1D numpy array ----------
    {
      const py::ssize_t nx = static_cast<py::ssize_t>(x_cache_.size());
      py::array_t<double> x_arr(nx);
      if (nx > 0)
      {
        auto xb = x_arr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < nx; ++i)
          xb(i) = x_cache_[static_cast<size_t>(i)];
      }
      d["x"] = std::move(x_arr); // shape = (nx,)
    }

    // ---------- y: always return a 1D numpy array ----------
    {
      const py::ssize_t ny = static_cast<py::ssize_t>(y_cache_.size());
      py::array_t<double> y_arr(ny);
      if (ny > 0)
      {
        auto yb = y_arr.mutable_unchecked<1>();
        for (py::ssize_t j = 0; j < ny; ++j)
          yb(j) = y_cache_[static_cast<size_t>(j)];
      }
      d["y"] = std::move(y_arr); // shape = (ny,)
    }

    return d;
  }

  // Set initial iterate in UN-SCALED space.
  void setInitSol(py::object x0_obj, py::object y0_obj)
  {
    if (n_ <= 0 || m_ <= 0)
    {
      throw std::runtime_error("setInitSol must be called after loadData()");
    }

    x0_buf_ = py::array_t<double>();
    y0_buf_ = py::array_t<double>();
    has_init_iterate_ = false;

    if (!x0_obj.is_none())
    {
      auto x0 = x0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      if ((int)x0.size() != n_)
      {
        throw std::runtime_error("x0 length mismatch with num_variables");
      }
      x0_buf_ = x0;
      has_init_iterate_ = true;
    }
    if (!y0_obj.is_none())
    {
      auto y0 = y0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      if ((int)y0.size() != m_)
      {
        throw std::runtime_error("y0 length mismatch with num_constraints");
      }
      y0_buf_ = y0;
      has_init_iterate_ = true;
    }
  }

  int num_variables() const { return n_; }
  int num_constraints() const { return m_; }
  int nnz() const { return nnz_; }

private:
  lp_problem_t *problem_;
  int m_, n_, nnz_;
  std::vector<double> var_lb_, var_ub_, obj_, rhs_;
  std::vector<double> con_lb_, con_ub_;
  std::vector<int> tmp_col_ptr_, tmp_row_idx_;
  std::vector<double> tmp_val_;
  std::vector<int> csr_row_ptr_, csr_col_idx_;
  std::vector<double> csr_val_;
  std::vector<double> x_cache_;
  std::vector<double> y_cache_;

  int iters_cache_ = 0;
  double solve_time_cache_ = 0.0;
  double primal_obj_cache_ = 0.0, dual_obj_cache_ = 0.0;
  double primal_feas_abs_cache_ = 0.0, dual_feas_abs_cache_ = 0.0, gap_abs_cache_ = 0.0;
  double primal_feas_rel_cache_ = 0.0, dual_feas_rel_cache_ = 0.0, gap_rel_cache_ = 0.0;
  int saveinfo_cache_ = 0;
  double beta_cache_ = 0.0;
  double primal_step_cache_ = 0.0, dual_step_cache_ = 0.0;
  int last_restart_iter_cache_ = -1;

  py::array_t<double> x0_buf_;
  py::array_t<double> y0_buf_;
  bool has_init_iterate_ = false;
};

static py::dict make_default_params()
{
  pdhg_parameters_t p;
  set_default_parameters(&p);
  py::dict d;
  d["l_inf_ruiz_iterations"] = p.l_inf_ruiz_iterations;
  d["has_pock_chambolle_alpha"] = p.has_pock_chambolle_alpha;
  d["pock_chambolle_alpha"] = p.pock_chambolle_alpha;
  d["bound_objective_rescaling"] = p.bound_objective_rescaling;
  d["verbose"] = p.verbose;
  d["verbose_time"] = p.verbose_time;
  d["termination_evaluation_frequency"] = p.termination_evaluation_frequency;
  d["reflection_coefficient"] = p.reflection_coefficient;
  d["time_sec_limit"] = p.termination_criteria.time_sec_limit;
  d["iteration_limit"] = p.termination_criteria.iteration_limit;
  d["eps_optimal_relative"] = p.termination_criteria.eps_optimal_relative;
  d["eps_feasible_relative"] = p.termination_criteria.eps_feasible_relative;
  d["eps_infeasible"] = p.termination_criteria.eps_infeasible;
  d["use_dual_nnz_gate"] = p.termination_criteria.use_dual_nnz_gate; // 0
  d["dual_nnz_factor"] = p.termination_criteria.dual_nnz_factor;     // 2.0
  d["dual_nnz_abs_tol"] = p.termination_criteria.dual_nnz_abs_tol;   // 0.0

  d["step_size_method"] = 0;
  d["step_size_safety"] = 0.998;

  d["power_max_iterations"] = 20;
  d["power_tolerance"] = 1e-4;
  d["hybrid_refine_iterations"] = 10;

  d["stepsize_power_reference"] = false;
  d["stepsize_reference_max_iterations"] = 5000;
  d["stepsize_reference_tolerance"] = 1e-4;
  return d;
}
static void reset_cuda_device()
{
  cudaDeviceSynchronize();
  cudaDeviceReset();
}
static void set_free_mode(int mode)
{
  if (mode < 0 || mode > 2)
    throw std::runtime_error("free_mode must be 0,1,2.");
  g_free_mode = mode;
}

// ====== PYBIND11 ======
PYBIND11_MODULE(pycupdlpx, m)
{
  m.doc() = "Python bindings for cuPDLPx (direct CSC input, safe device->host copy, unscale)";

  py::class_<CupdlpxHolder>(m, "cupdlpx")
      .def(py::init<>())
      .def("loadData_csc", &CupdlpxHolder::loadData_csc,
           py::arg("indptr"), py::arg("indices"), py::arg("data"),
           py::arg("m"), py::arg("n"),
           py::arg("c"), py::arg("rhs"),
           py::arg("lb"), py::arg("ub"),
           py::arg("nEqs"),
           "Load problem in CSC (indptr, indices, data) + vectors (c, rhs, lb, ub).")
      .def("loadData", &CupdlpxHolder::loadData,
           py::arg("A"), py::arg("c"), py::arg("rhs"), py::arg("lb"), py::arg("ub"), py::arg("nEqs"),
           "Load problem from a scipy.sparse.csc_matrix A plus vectors (c, rhs, lb, ub).")
      .def("solve", &CupdlpxHolder::solve, py::arg("params") = py::dict(),
           "Solve and return a dict with summary and unscaled (x,y).")
      .def("getSolution", &CupdlpxHolder::getSolution,
           "Return solver stats (iters/time/feasibility/gap and placeholders).")
      .def("setInitSol", &CupdlpxHolder::setInitSol,
           py::arg("x0"), py::arg("y0"),
           R"doc(Set initial iterate (unscaled) for PDHG: x0 length = n, y0 length = m. 
Pass None for either to skip.)doc")
      .def_property_readonly("num_variables", &CupdlpxHolder::num_variables)
      .def_property_readonly("num_constraints", &CupdlpxHolder::num_constraints)
      .def_property_readonly("nnz", &CupdlpxHolder::nnz);

  m.def("make_default_params", &make_default_params);
  m.def("reset_cuda_device", &reset_cuda_device);
  m.def("set_free_mode", &set_free_mode, "0:free all, 1:free state only (default), 2:free problem only");
  m.def("print_solver_summary", [](py::capsule state_capsule)
        {
    auto* state = state_capsule.get_pointer<pdhg_solver_state_t>();
    if (!state) throw std::runtime_error("Invalid solver state capsule.");
    print_solver_summary(state); });

  // CUDA_ERROR_CONTEXT_IS_DESTROYED / CUDA_ERROR_INVALID_HANDLE。
}
