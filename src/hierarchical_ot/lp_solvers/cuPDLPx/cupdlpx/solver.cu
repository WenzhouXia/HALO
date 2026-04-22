/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "solver.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <vector>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cmath>    // std::fabs


#include <cfloat>   // DBL_MAX

#include <chrono>
#include <cuda_runtime.h>
#include <cstdio>
// =================== SMART PATCH START ===================
// 仅在 CUDA 版本低于 11.7 (11070) 时启用此补丁
#include <cuda_runtime_api.h>

#if defined(CUDART_VERSION) && CUDART_VERSION < 11070

#include <cublas_v2.h>

// 1. 补丁：cublasGetStatusName
#ifndef CUBLAS_GET_STATUS_NAME_H_
#define CUBLAS_GET_STATUS_NAME_H_
static const char* cublasGetStatusName(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}
#endif

// 2. 补丁：cublasDnrm2_v2_64
// 旧版本没有 64 位接口，回退到标准接口
#ifndef cublasDnrm2_v2_64
#define cublasDnrm2_v2_64 cublasDnrm2
#endif

#endif
// =================== SMART PATCH END ===================

using clk = std::chrono::high_resolution_clock;
__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub,
    int n, double step_size);
__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack);
__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual, const double *primal_product,
    const double *const_lb, const double *const_ub, int n, double step_size);
__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb, const double *const_ub,
    int n, double step_size);
__global__ void halpern_update_kernel(
    const double *initial_primal, double *current_primal, const double *reflected_primal,
    const double *initial_dual, double *current_dual, const double *reflected_dual,
    int n_vars, int n_cons, double weight, double reflection_coeff);
__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_vars, int n_cons);
static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state);
static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state);
static void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient);
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *cudaMemsetParams);
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

static pdhg_solver_state_t *initialize_solver_state(
    const lp_problem_t *original_problem,
    const rescale_info_t *rescale_info,
    const pdhg_parameters_t *params);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);


static inline double toMiB(std::size_t b){ return b / (1024.0 * 1024.0); }

// =================== [新增] 全局内存统计变量 ===================
static size_t g_solver_peak_mem_bytes = 0;

extern "C" {
    // 供 binding 调用的接口
    size_t get_last_solver_peak_mem() {
        return g_solver_peak_mem_bytes;
    }
}
// =============================================================

// 在任意位置调用以打印当前 GPU 显存用量；建议在主循环前调用一次
static inline void print_gpu_mem_once(const char* tag = "before loop", bool sync_before_query = true) {
    if (sync_before_query) cudaDeviceSynchronize();   // 确保前面 kernel/异步分配都完成
    std::size_t freeB = 0, totalB = 0;
    cudaError_t st = cudaMemGetInfo(&freeB, &totalB);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "[GPU MEM] %s: cudaMemGetInfo failed: %s\n",
                     tag, cudaGetErrorString(st));
        return;
    }
    std::size_t usedB = totalB - freeB;
    int dev = 0; cudaGetDevice(&dev);
    std::printf("[GPU MEM][dev=%d] %s: used=%.2f MiB  free=%.2f MiB  total=%.2f MiB\n",
                dev, tag, toMiB(usedB), toMiB(freeB), toMiB(totalB));
    std::fflush(stdout);
}
pdhg_solver_state_t *optimize(const pdhg_parameters_t *params, const lp_problem_t *original_problem)
{
    cudaFree(0);

    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // -------- setup 分解计时 --------
    double t_setup = 0.0, t_print_init = 0.0, t_rescale = 0.0, t_init_state = 0.0, t_step_init = 0.0;

    auto t0 = clk::now();
    auto t_a = clk::now();
    print_initial_info(params, original_problem);
    cudaDeviceSynchronize();
    auto t_b = clk::now();
    t_print_init = sec(t_a, t_b);

    t_a = clk::now();
    rescale_info_t *rescale_info = rescale_problem(params, original_problem);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_rescale = sec(t_a, t_b);

    t_a = clk::now();
    pdhg_solver_state_t *state = initialize_solver_state(original_problem, rescale_info, params);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_init_state = sec(t_a, t_b);

    rescale_info_free(rescale_info);

    t_a = clk::now();
    initialize_step_size_and_primal_weight(state, params);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_step_init = sec(t_a, t_b);

    t_setup = t_print_init + t_rescale + t_init_state + t_step_init;

    // -------- 循环阶段计时（累计）--------
    double t_eval = 0.0;    // 计算残差、(早期)不可行信息、检查停止、打印
    double t_restart = 0.0; // 自适应 restart 判定与执行
    double t_kernel = 0.0;  // 核心一步：primal/dual 更新 + (必要时) fixed point + Halpern

    clock_t start_time_clock = clock();
    auto t_loop_start = clk::now();

    bool do_restart = false;
    // =================== [修改] 统计峰值显存 ===================
    // 在主循环开始前，所有 heavy allocation (向量/矩阵) 均已完成
    // 这是统计 "算法运行时占用" 的最佳时刻
    {
        cudaDeviceSynchronize();
        size_t free_byte, total_byte;
        if (cudaMemGetInfo(&free_byte, &total_byte) == cudaSuccess) {
            g_solver_peak_mem_bytes = total_byte - free_byte; // 记录当前实际占用
            
            // 如果 verbose，顺便打印出来供核对
            if (params->verbose) {
                printf("[GPU MEM STATS] Peak Memory at Loop Start: %.2f MiB (Used: %.2f / Total: %.2f)\n", 
                       g_solver_peak_mem_bytes / (1024.0 * 1024.0),
                       (double)(total_byte - free_byte) / (1024.0 * 1024.0),
                       (double)total_byte / (1024.0 * 1024.0));
            }
        }
    }
    // =========================================================
    if (params->verbose){
    print_gpu_mem_once("before main loop (after allocs)", /*sync_before_query=*/true);
    }
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        // ---- 评估/打印块 ----
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            auto t1 = clk::now();
            compute_residual(state);
            cudaDeviceSynchronize();

            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
                cudaDeviceSynchronize();
            }

            state->cumulative_time_sec = (double)(clock() - start_time_clock) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            cudaDeviceSynchronize();

            auto t2 = clk::now();
            t_eval += sec(t1, t2);
        }

        // ---- restart 判定/执行块 ----
        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            auto t1 = clk::now();
            do_restart = should_do_adaptive_restart(
                state, &params->restart_params,
                params->termination_evaluation_frequency, 
                params->verbose);
            if (do_restart)
            {
                perform_restart(state, params);
                cudaDeviceSynchronize();
            }
            auto t2 = clk::now();
            t_restart += sec(t1, t2);
        }

        // 下一个 major 的布尔标志
        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        // ---- 核心更新块（primal/dual/fixed-point/Halpern）----
        auto t1 = clk::now();

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);
        cudaDeviceSynchronize();

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            cudaDeviceSynchronize();

            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }

        halpern_update(state, params->reflection_coefficient);
        cudaDeviceSynchronize();

        auto t2 = clk::now();
        t_kernel += sec(t1, t2);

        state->inner_count++;
        state->total_count++;
    }

    // -------- finalize 计时 --------
    auto t_loop_end = clk::now();
    auto t1 = clk::now();
    pdhg_final_log(state, params->verbose, state->termination_reason);
    cudaDeviceSynchronize();
    auto t2 = clk::now();
    double t_finalize = sec(t1, t2);

    auto t_end = clk::now();
    double t_total = sec(t0, t_end);
    double t_loop  = sec(t_loop_start, t_loop_end);

    // -------- 汇总打印 --------
    // if (params->verbose_time == true){
    //     fprintf(stdout,
    //     "[TIMING] setup=%.3fs  (print_init=%.3f, rescale=%.3f, init_state=%.3f, step_init=%.3f)\n",
    //     t_setup, t_print_init, t_rescale, t_init_state, t_step_init);
    // fprintf(stdout,
    //     "[TIMING] loop: eval=%.3fs, kernel=%.3fs, restart=%.3fs  (loop_total=%.3fs)\n",
    //     t_eval, t_kernel, t_restart, t_loop);
    // fprintf(stdout,
    //     "[TIMING] finalize=%.3fs   total=%.3fs\n",
    //     t_finalize, t_total);
    // fflush(stdout);
    // }
    

    return state;
}



static pdhg_solver_state_t *initialize_solver_state(
    const lp_problem_t *original_problem,
    const rescale_info_t *rescale_info,
    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = original_problem->num_variables;
    int n_cons = original_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = original_problem->objective_constant;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(state->constraint_matrix->row_ptr, rescale_info->scaled_problem->constraint_matrix_row_pointers, (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind, rescale_info->scaled_problem->constraint_matrix_col_indices, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val, rescale_info->scaled_problem->constraint_matrix_values, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    // CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
    //     state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
    //     state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
    //     state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
    //     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    //     CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    // CUSPARSE_CHECK(cusparseCsr2cscEx2(
    //     state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
    //     state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
    //     state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
    //     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    //     CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer));

    CUDA_CHECK(cudaFree(buffer));

    ALLOC_AND_COPY(state->variable_lower_bound, rescale_info->scaled_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, rescale_info->scaled_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, rescale_info->scaled_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, rescale_info->scaled_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, rescale_info->scaled_problem->constraint_upper_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_rescaling, rescale_info->con_rescale, con_bytes);
    ALLOC_AND_COPY(state->variable_rescaling, rescale_info->var_rescale, var_bytes);

    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    // ---- Warm start: map UN-SCALED x0/y0 -> INTERNAL SCALED and write to all start buffers ----
    if (params && params->has_initial_iterate &&
        (params->initial_primal_unscaled || params->initial_dual_unscaled)) {

        const double *s_var = rescale_info->var_rescale; // len = n_vars (host)
        const double *s_con = rescale_info->con_rescale; // len = n_cons (host)

        // 这两个“标量缩放”也要乘上，否则会导致 Ax-b 在内部空间上来就巨大
        const double alpha_x = (state->constraint_bound_rescaling  != 0.0)
                                ? state->constraint_bound_rescaling  : 1.0;
        const double alpha_y = (state->objective_vector_rescaling != 0.0)
                                ? state->objective_vector_rescaling : 1.0;

        // 用“未缩放空间”的边界先夹紧 x0（避免跨尺度延拓后出界造成大残差）
        const double *lb_u = original_problem->variable_lower_bound; // 未缩放 lb
        const double *ub_u = original_problem->variable_upper_bound; // 未缩放 ub

        // 1) 处理 x0：x_internal = clip_unscaled(x0, lb, ub) * (var_rescale[i] * alpha_x)
        if (params->initial_primal_unscaled) {
            double *x_scaled_h = (double *)safe_malloc(var_bytes);
            for (int i = 0; i < n_vars; ++i) {
                double xu = params->initial_primal_unscaled[i];   // 未缩放
                if (lb_u) xu = (xu < lb_u[i]) ? lb_u[i] : xu;
                if (ub_u) xu = (xu > ub_u[i]) ? ub_u[i] : xu;
                const double sv = s_var ? s_var[i] : 1.0;
                x_scaled_h[i] = xu * (sv * alpha_x);              // ✅ 正确映射
            }

            CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,   x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->current_primal_solution,   x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution,      x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution, x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            free(x_scaled_h);
        }

        // 2) 处理 y0：y_internal = y0_unscaled * (con_rescale[j] * alpha_y)
        if (params->initial_dual_unscaled) {
            double *y_scaled_h = (double *)safe_malloc(con_bytes);
            for (int j = 0; j < n_cons; ++j) {
                const double yu = params->initial_dual_unscaled[j]; // 未缩放
                const double sc = s_con ? s_con[j] : 1.0;
                y_scaled_h[j] = yu * (sc * alpha_y);                // ✅ 正确映射
            }

            CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,   y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->current_dual_solution,   y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution,      y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution, y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            free(y_scaled_h);
        }
    }

    

    double *temp_host = (double *)safe_malloc(fmax(var_bytes, con_bytes));
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_lower_bound[i]) ? rescale_info->scaled_problem->constraint_lower_bound[i] : 0.0;
    ALLOC_AND_COPY(state->constraint_lower_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_upper_bound[i]) ? rescale_info->scaled_problem->constraint_upper_bound[i] : 0.0;
    ALLOC_AND_COPY(state->constraint_upper_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_lower_bound[i]) ? rescale_info->scaled_problem->variable_lower_bound[i] : 0.0;
    ALLOC_AND_COPY(state->variable_lower_bound_finite_val, temp_host, var_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_upper_bound[i]) ? rescale_info->scaled_problem->variable_upper_bound[i] : 0.0;
    ALLOC_AND_COPY(state->variable_upper_bound_finite_val, temp_host, var_bytes);
    free(temp_host);

    double sum_of_squares = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        sum_of_squares += original_problem->objective_vector[i] * original_problem->objective_vector[i];
    }
    state->objective_vector_norm = sqrt(sum_of_squares);

    sum_of_squares = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = original_problem->constraint_lower_bound[i];
        double upper = original_problem->constraint_upper_bound[i];

        if (isfinite(lower) && (lower != upper))
        {
            sum_of_squares += lower * lower;
        }

        if (isfinite(upper))
        {
            sum_of_squares += upper * upper;
        }
    }

    state->constraint_bound_norm = sqrt(sum_of_squares);
    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual = (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    state->best_primal_dual_residual_gap = INFINITY;
    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matA, state->num_constraints, state->num_variables, state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind, state->constraint_matrix->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt, state->num_variables, state->num_constraints, state->constraint_matrix_t->num_nonzeros, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind, state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    // CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
    //                                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    // CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
    //                                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_primal_d, ones_primal_h, state->num_variables * sizeof(double), cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_dual_d, ones_dual_h, state->num_constraints * sizeof(double), cudaMemcpyHostToDevice));
    free(ones_dual_h);

    state->k_p = params->restart_params.k_p;
    state->k_i = params->restart_params.k_i;
    state->k_d = params->restart_params.k_d;
    state->previous_restart_dual_residual = DBL_MAX;
    state->previous_restart_gap = DBL_MAX;
    return state;
}

__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub,
    int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
    }
}

__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual, const double *primal_product,
    const double *const_lb, const double *const_ub, int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        reflected_dual[i] = 2.0 * (temp - temp_proj) * step_size - current_dual[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb, const double *const_ub,
    int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
    }
}

__global__ void halpern_update_kernel(
    const double *initial_primal, double *current_primal, const double *reflected_primal,
    const double *initial_dual, double *current_dual, const double *reflected_dual,
    int n_vars, int n_cons, double weight, double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double reflected = reflection_coeff * reflected_dual[idx] + (1.0 - reflection_coeff) * current_dual[idx];
        current_dual[idx] = weight * reflected + (1.0 - weight) * initial_dual[idx];
    }
}

__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_vars, int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
}

static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->current_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    double step = state->step_size / state->primal_weight;

    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->pdhg_primal_solution, state->reflected_primal_solution,
            state->dual_product, state->objective_vector, state->variable_lower_bound,
            state->variable_upper_bound, state->num_variables, step, state->dual_slack);
    }
    else
    {
        compute_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->reflected_primal_solution, state->dual_product,
            state->objective_vector, state->variable_lower_bound, state->variable_upper_bound,
            state->num_variables, step);
    }
}

static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->reflected_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    double step = state->step_size * state->primal_weight;

    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->pdhg_dual_solution, state->reflected_dual_solution,
            state->primal_product, state->constraint_lower_bound, state->constraint_upper_bound,
            state->num_constraints, step);
    }
    else
    {
        compute_next_pdhg_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->reflected_dual_solution, state->primal_product,
            state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints, step);
    }
}

static void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient)
{
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    halpern_update_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->current_primal_solution, state->reflected_primal_solution,
        state->initial_dual_solution, state->current_dual_solution, state->reflected_dual_solution,
        state->num_variables, state->num_constraints, weight, reflection_coefficient);
}

// static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
// {
//     compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
//         state->initial_primal_solution, state->pdhg_primal_solution, state->delta_primal_solution,
//         state->initial_dual_solution, state->pdhg_dual_solution, state->delta_dual_solution,
//         state->num_variables, state->num_constraints);

//     double primal_dist, dual_dist;
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

//     double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

//     if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
//     {
//         double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
//         state->primal_weight_error_sum *= params->restart_params.i_smooth;
//         state->primal_weight_error_sum += error;
//         double delta_error = error - state->primal_weight_last_error;
//         state->primal_weight *= exp(params->restart_params.k_p * error +
//                                     params->restart_params.k_i * state->primal_weight_error_sum +
//                                     params->restart_params.k_d * delta_error);
//         state->primal_weight_last_error = error;
//     }
//     else
//     {
//         state->primal_weight = state->best_primal_weight;
//         state->primal_weight_error_sum = 0.0;
//         state->primal_weight_last_error = 0.0;
//     }

//     double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
//     if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
//     {
//         state->best_primal_dual_residual_gap = primal_dual_residual_gap;
//         state->best_primal_weight = state->primal_weight;
//     }

//     CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));

//     state->inner_count = 0;
//     state->last_trial_fixed_point_error = INFINITY;
// }
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->pdhg_primal_solution, state->delta_primal_solution,
        state->initial_dual_solution, state->pdhg_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    double primal_dist, dual_dist;
    // CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    // CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    // --- PID DEBUG: 保存旧权重以便打印 ---
    const double old_primal_weight = state->primal_weight;
    // 确保在 C++ 中包含 stdio.h (solver.cu 顶部已有)
    // #include <stdio.h> 

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
    {
        // if (params->verbose) {
        //     printf("\n[PID DEBUG] Iter %d: PID controller ACTIVE.\n", state->total_count);
        //     printf("  [PID INPUT] primal_dist (||x_new-x_old||) = %.6e\n", primal_dist);
        //     printf("  [PID INPUT] dual_dist   (||y_new-y_old||) = %.6e\n", dual_dist);
        //     printf("  [PID INPUT] old_primal_weight             = %.6e\n", old_primal_weight);
        //     printf("  [PID INPUT] old_primal_weight_last_error  = %.6e\n", state->primal_weight_last_error);
        //     printf("  [PID INPUT] old_primal_weight_error_sum   = %.6e\n", state->primal_weight_error_sum);
        // }
        // --- PID DEBUG: 打印 PID 输入 ---
        
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        
        // --- PID DEBUG: 打印 P, I, D 系数和中间值 ---
        const double Kp = params->restart_params.k_p;
        const double Ki = params->restart_params.k_i;
        const double Kd = params->restart_params.k_d;
        const double i_smooth = params->restart_params.i_smooth;
        
        state->primal_weight_error_sum *= i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;

        const double term_P = Kp * error;
        const double term_I = Ki * state->primal_weight_error_sum;
        const double term_D = Kd * delta_error;
        const double exp_term = term_P + term_I + term_D;

        // if (params->verbose) {
        //         printf("  [PID CALC]  Kp=%.2e, Ki=%.2e, Kd=%.2e, i_smooth=%.2e\n", Kp, Ki, Kd, i_smooth);
        //     printf("  [PID CALC]  error (P term base)           = %.6e\n", error);
        //     printf("  [PID CALC]  delta_error (D term base)     = %.6e\n", delta_error);
        //     printf("  [PID CALC]  new_error_sum (I term base)   = %.6e\n", state->primal_weight_error_sum);
        //     printf("  [PID CALC]  Term P (Kp * error)           = %.6e\n", term_P);
        //     printf("  [PID CALC]  Term I (Ki * sum)             = %.6e\n", term_I);
        //     printf("  [PID CALC]  Term D (Kd * delta)           = %.6e\n", term_D);
        //     printf("  [PID CALC]  exp_term (P+I+D)              = %.6e\n", exp_term);

        // }
        
        state->primal_weight *= exp(exp_term);
        state->primal_weight_last_error = error;

        // --- PID DEBUG: 打印最终结果 ---
        // if (params->verbose) { 
        //     printf("  [PID OUTPUT] new_primal_weight            = %.6e\n", state->primal_weight);
        //     fflush(stdout); // 确保立即刷新缓冲区
        // }   
    }
    else
    {
        // // --- PID DEBUG: 打印重置情况 ---
        // if (params->verbose) {
        //     printf("\n[PID DEBUG] Iter %d: PID controller SKIPPED (safety check fail).\n", state->total_count);
        //     printf("  [PID INFO]  primal_dist = %.6e, dual_dist = %.6e, ratio_infeas = %.6e\n",
        //             primal_dist, dual_dist, ratio_infeas);
        //     printf("  [PID RESET] Resetting primal_weight from %.6e to best_primal_weight %.6e\n",
        //             old_primal_weight, state->best_primal_weight);
        //     fflush(stdout); // 确保立即刷新缓冲区
        // }
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}
// static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
// {
//     // ... (compute_delta_solution_kernel, cublasDnrm2_v2_64, ...)
//     compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
//         state->initial_primal_solution, state->pdhg_primal_solution, state->delta_primal_solution,
//         state->initial_dual_solution, state->pdhg_dual_solution, state->delta_dual_solution,
//         state->num_variables, state->num_constraints);

//     double primal_dist, dual_dist;
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

//     double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;
//     // ...

//     const double GAP_EXPLOSION_FACTOR = 5.0;
//     const double current_gap = state->relative_objective_gap;
//     const double previous_gap = state->previous_restart_gap; // (假设已添加)
//     bool is_exploding = false; 

//     if (state->total_count > params->termination_evaluation_frequency &&
//         previous_gap > 1e-12 && 
//         current_gap > GAP_EXPLOSION_FACTOR * previous_gap)
//     {
//         is_exploding = true;
//         printf("\n[ROLLBACK] Iter %d: Gap exploded! Current=%.2e > %.1f * Previous=%.2e\n",
//                state->total_count, current_gap, GAP_EXPLOSION_FACTOR, previous_gap);
//     }

//     if (is_exploding)
//     {
//         // === [ROLLBACK] ===
//         printf("  [ROLLBACK] Reverting to previous state...\n");
//         printf("    [ROLLBACK STATE] Old Kp = %.6e, Old K_i = %.6e\n", state->k_p, state->k_i);
//         printf("    [ROLLBACK STATE] Old primal_weight = %.6e\n", state->primal_weight);
        
//         // --- 【你的关键修正】 ---
//         // 我们回退 total_count，就好像这个坏的周期 (inner_count) 从未发生过
//         printf("    [ROLLBACK] Rewinding total_count from %d by %d iterations.\n", 
//                state->total_count, state->inner_count);
//         state->total_count -= state->inner_count; // <-- 完 全 回 滚
//         // ---

//         // 1. 永久禁用 PID
//         state->k_p = 0.0;
//         state->k_i = 0.0;
//         state->k_d = 0.0;

//         // 2. 强制降低 primal_weight
//         state->primal_weight *= 0.1; 

//         // 3. 清空 PID 累加器
//         state->primal_weight_error_sum = 0.0;
//         state->primal_weight_last_error = 0.0;

//         // 4. 重置 inner counter
//         state->inner_count = 0;
//         state->last_trial_fixed_point_error = INFINITY;

//         // 5. 恢复解
//         CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->initial_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//         CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        
//         printf("    [ROLLBACK] Action: Kp, Ki, Kd set to 0. PID RESET. primal_weight OVERRIDDEN to %.2e. Solution restored.\n", 
//                state->primal_weight);
//         fflush(stdout);
//     }
//     else
//     {
//         // === [NORMAL RESTART] ===
//         const double old_primal_weight = state->primal_weight;

//         if (state->k_p == 0.0) // 检查 PID 是否已被禁用
//         {
//              printf("\n[PID DEBUG] Iter %d: PID permanently disabled. Holding primal_weight constant at %.6e\n", 
//                     state->total_count, state->primal_weight);
//              fflush(stdout);
//         }
//         else if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
//         {
//             // ... (正常的 PID 逻辑) ...
//             printf("\n[PID DEBUG] Iter %d: PID controller ACTIVE.\n", state->total_count);
            
//             double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
            
//             const double Kp = state->k_p;
//             const double Ki = state->k_i;
//             const double Kd = state->k_d;
//             const double i_smooth = params->restart_params.i_smooth;
            
//             state->primal_weight_error_sum *= i_smooth;
//             state->primal_weight_error_sum += error;
//             double delta_error = error - state->primal_weight_last_error;
//             const double exp_term = (Kp * error) + (Ki * state->primal_weight_error_sum) + (Kd * delta_error);
//             state->primal_weight *= exp(exp_term);
//             state->primal_weight_last_error = error;

//             printf("  [PID OUTPUT] new_primal_weight            = %.6e\n", state->primal_weight);
//             fflush(stdout);
//         }
//         else
//         {
//             // ... (PID SKIPPED 逻辑) ...
//             printf("\n[PID DEBUG] Iter %d: PID controller SKIPPED (safety check fail).\n", state->total_count);
//             printf("  [PID RESET] Resetting primal_weight from %.6e to best_primal_weight %.6e\n",
//                     old_primal_weight, state->best_primal_weight);
//             fflush(stdout);
//             state->primal_weight = state->best_primal_weight;
//             state->primal_weight_error_sum = 0.0;
//             state->primal_weight_last_error = 0.0;
//         }

//         // --- 更新 Best Weight ---
//         double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
//         if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
//         {
//             state->best_primal_dual_residual_gap = primal_dual_residual_gap;
//             state->best_primal_weight = state->primal_weight;
//         }

//         // --- 正常提交状态 ---
//         state->previous_restart_gap = state->relative_objective_gap; // 保存基准 Gap

//         CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//         CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//         CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
//         CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));

//         state->inner_count = 0;
//         state->last_trial_fixed_point_error = INFINITY;
//     }
// }


// __global__ void row_abs_sum_kernel(const int* __restrict__ row_ptr,
//                                    const double* __restrict__ val,
//                                    int m, double* __restrict__ out) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < m) {
//         double s = 0.0;
//         int beg = row_ptr[i];
//         int end = row_ptr[i + 1];
//         #pragma unroll 4
//         for (int p = beg; p < end; ++p) s += fabs(val[p]);
//         out[i] = s;
//     }
// }

// // 返回 CSR 的 “最大行绝对值和”
// static double csr_max_row_abs_sum(const cu_sparse_matrix_csr_t* M) {
//     const int m = M->num_rows;
//     double* d_row_sums = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_row_sums, sizeof(double) * (size_t)m));
//     dim3 block(256);
//     dim3 grid((m + block.x - 1) / block.x);
//     row_abs_sum_kernel<<<grid, block>>>(M->row_ptr, M->val, m, d_row_sums);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
//     thrust::device_ptr<double> p(d_row_sums);
//     double mx = thrust::reduce(p, p + m, 0.0, thrust::maximum<double>());
//     CUDA_CHECK(cudaFree(d_row_sums));
//     return mx;
// }

// // \|A\|_2 上界：sqrt(\|A\|_1 * \|A\|_inf)，其中 \|A\|_1 = max col sum(A) = max row sum(AT)
// static double estimate_sigma_max_upper_1inf(const cu_sparse_matrix_csr_t* A,
//                                             const cu_sparse_matrix_csr_t* AT) {
//     double infA = csr_max_row_abs_sum(A);   // \|A\|_inf
//     double oneA = csr_max_row_abs_sum(AT);  // \|A\|_1
//     return sqrt(oneA * infA);
// }

// // ——替换你的步长初始化——
// static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
//                                                    const pdhg_parameters_t *params)
// {
//     // 便宜上界；如需更保守/更激进可调 safety
//     const double safety = 0.95;
//     double L = estimate_sigma_max_upper_1inf(state->constraint_matrix,
//                                              state->constraint_matrix_t);
//     state->step_size = safety / L;

//     if (params->bound_objective_rescaling) {
//         state->primal_weight = 1.0;
//     } else {
//         state->primal_weight = state->objective_vector_norm / state->constraint_bound_norm;
//     }
//     state->best_primal_weight = state->primal_weight;
// }
////////////////////////////////////////////////////////////////////////////////////////////
// // 行绝对值和：对 CSR 的每一行 i 计算 sum_{p=row_ptr[i]}^{row_ptr[i+1]-1} |val[p]|
// __global__ void row_abs_sum_kernel(const int* __restrict__ row_ptr,
//                                    const double* __restrict__ val,
//                                    int m, double* __restrict__ out) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= m) return;
//     int beg = row_ptr[i];
//     int end = row_ptr[i + 1];
//     double s = 0.0;
//     // 手写展开一点点有利于吞吐，但保持简单更稳妥
//     for (int p = beg; p < end; ++p) {
//         s += fabs(val[p]);
//     }
//     out[i] = s;
// }

// // 块内最大值归约（带 grid-stride，写每块的最大值到 out[blockIdx.x]）
// // 注意：用 shared mem 做块内归约；多轮调用直至只剩 1 个 block 输出
// __global__ void reduce_max_kernel(const double* __restrict__ in, int n,
//                                   double* __restrict__ out) {
//     extern __shared__ double sdata[];  // 大小 = blockDim.x * sizeof(double)

//     // 每线程本地最大值（支持 grid-stride）
//     double local_max = -DBL_MAX;
//     // 全局线性线程 id（跨 block）
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int idx = tid; idx < n; idx += stride) {
//         double v = in[idx];
//         // 允许 NaN：将其视为 -inf，避免干扰（也可选择 isfinite 判定）
//         if (v > local_max) local_max = v;
//     }

//     // 写入共享内存并做块内归约
//     sdata[threadIdx.x] = local_max;
//     __syncthreads();

//     // 标准二分归约
//     for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//         if (threadIdx.x < offset) {
//             double other = sdata[threadIdx.x + offset];
//             if (other > sdata[threadIdx.x]) sdata[threadIdx.x] = other;
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x == 0) {
//         out[blockIdx.x] = sdata[0];
//     }
// }

// // 简单 CSR 健壮性检查：row_ptr[0]==0 && row_ptr[m]==nnz
// static bool csr_basic_check(const int* d_row_ptr, int m, int nnz, cudaStream_t stream) {
//     int h0 = -1, hN = -1;
//     // 只拷两个整数，避免大块 D2H
//     if (cudaMemcpyAsync(&h0, d_row_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
//     if (cudaMemcpyAsync(&hN, d_row_ptr + m, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
//     if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
//     return (h0 == 0) && (hN == nnz);
// }

// // 利用上面两个 kernel：||M||_∞ = max_i sum_j |M_ij|（CSR）
// static double csr_max_row_abs_sum_stream(const cu_sparse_matrix_csr_t* M,
//                                          cudaStream_t stream) {
//     const int m   = M->num_rows;
//     // const int nnz = M->num_nonzeros;
//     if (m <= 0) return 0.0;

//     // 1) 行和
//     double* d_row_sums = nullptr;
//     CUDA_CHECK(cudaMalloc((void**)&d_row_sums, sizeof(double) * (size_t)m));

//     const int TPB = 256;
//     int grid_rows = (m + TPB - 1) / TPB;
//     // grid 限制，防止过大
//     grid_rows = (grid_rows > 65535) ? 65535 : grid_rows;

//     row_abs_sum_kernel<<<grid_rows, TPB, 0, stream>>>(
//         M->row_ptr, M->val, m, d_row_sums);
//     CUDA_CHECK(cudaGetLastError());

//     // 2) 归约最大值（多轮）
//     // 预估第一轮输出块数，并为“ping-pong”准备一个临时缓冲区
//     int curr_n = m;
//     double* d_in  = d_row_sums;
//     double* d_tmp = nullptr;

//     // 最大需要的块数：ceil(curr_n / (TPB * grid_rows)) 每轮会大幅减少
//     // 简化：分配一个上界大小的临时缓冲 = 至多 grid_rows 个块的输出
//     int max_blocks = grid_rows;
//     if (max_blocks < 1) max_blocks = 1;
//     CUDA_CHECK(cudaMalloc((void**)&d_tmp, sizeof(double) * (size_t)max_blocks));

//     double* d_out = d_tmp;

//     while (curr_n > 1) {
//         int num_blocks = (curr_n + TPB - 1) / TPB;
//         if (num_blocks > 65535) num_blocks = 65535;
//         size_t smem_bytes = TPB * sizeof(double);

//         reduce_max_kernel<<<num_blocks, TPB, smem_bytes, stream>>>(d_in, curr_n, d_out);
//         CUDA_CHECK(cudaGetLastError());

//         // 下一轮输入就是 d_out；如果下一轮还需要，再在 d_in 和 d_out 之间切换
//         curr_n = num_blocks;

//         // 若下一轮还大于 1，就交换 in/out 并确保 d_out 指向可以容纳下一轮的输出
//         if (curr_n > 1) {
//             // 交换指针：让 d_in 指向刚产生的输出
//             double* tmp = d_in;
//             d_in  = d_out;
//             d_out = tmp; // d_out 指回 d_row_sums（容量足够），避免重复分配
//         }
//     }

//     // 结果在 d_out[0] 或 d_in[0]，取决于循环结束时是否交换
//     double h_max = 0.0;
//     const double* d_result = (curr_n == 1) ? d_out : d_in;

//     CUDA_CHECK(cudaMemcpyAsync(&h_max, d_result, sizeof(double), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     CUDA_CHECK(cudaFree(d_row_sums));
//     CUDA_CHECK(cudaFree(d_tmp));
//     return h_max;
// }

// // sqrt(||A||_1 * ||A||_inf)，其中 ||A||_1 = ||A^T||_inf
// // ★ 完全不依赖 Thrust/CUB
// static double estimate_sigma_max_upper_1inf(const cu_sparse_matrix_csr_t* A,
//                                             const cu_sparse_matrix_csr_t* AT,
//                                             cudaStream_t stream) {
//     // 轻量自检：防止把 CSC 当成 CSR 传进来、或 row_ptr 损坏
//     bool okA  = csr_basic_check(A->row_ptr,  A->num_rows,  A->num_nonzeros,  stream);
//     bool okAT = csr_basic_check(AT->row_ptr, AT->num_rows, AT->num_nonzeros, stream);
//     if (!okA || !okAT) {
//         fprintf(stderr,
//             "[StepSize] CSR check failed (A.ok=%d, AT.ok=%d). "
//             "row_ptr[0] must be 0 and row_ptr[m]==nnz.\n", (int)okA, (int)okAT);
//         // 让上层决定是否回退到幂迭代
//         return NAN;
//     }

//     double infA = csr_max_row_abs_sum_stream(A,  stream);
//     double oneA = csr_max_row_abs_sum_stream(AT, stream); // = ||A||_1

//     // 健壮性检查
//     if (!(std::isfinite(infA) && infA >= 0.0) ||
//         !(std::isfinite(oneA) && oneA >= 0.0)) {
//         fprintf(stderr, "[StepSize] invalid one/inf norm: oneA=%.6e, infA=%.6e\n", oneA, infA);
//         return NAN;
//     }

//     double bound = sqrt(oneA * infA);
//     return bound;
// }

// static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
//                                                    const pdhg_parameters_t *params)
// {
//     const cu_sparse_matrix_csr_t* A  = state->constraint_matrix;
//     const cu_sparse_matrix_csr_t* AT = state->constraint_matrix_t;

//     // 统一流
//     cudaStream_t stream = 0;
//     CUSPARSE_CHECK(cusparseGetStream(state->sparse_handle, &stream));
//     CUBLAS_CHECK(cublasSetStream(state->blas_handle,   stream));

//     const double safety = (params->step_size_safety > 0.0 && params->step_size_safety < 1.0)
//                           ? params->step_size_safety : 0.998;

//     auto method_name = [](int m)->const char* {
//         switch ((step_size_method_t)m) {
//             case STEP_SIZE_POWER_ITERATION: return "power";
//             case STEP_SIZE_ONE_INF_UPPER  : return "one_inf";
//             case STEP_SIZE_HYBRID         : return "hybrid";
//             default: return "unknown";
//         }
//     };

//     // —— 计算 L —— 
//     double L = 0.0;

//     switch ((step_size_method_t)params->step_size_method)
//     {
//     case STEP_SIZE_POWER_ITERATION:
//     {
//         int    it  = (params->power_max_iterations > 0) ? params->power_max_iterations : 20;
//         double tol = (params->power_tolerance > 0.0)     ? params->power_tolerance     : 1e-4;
//         L = estimate_maximum_singular_value(
//                 state->sparse_handle, state->blas_handle, A, AT, it, tol);
//         break;
//     }
//     case STEP_SIZE_ONE_INF_UPPER:
//     {
//         L = estimate_sigma_max_upper_1inf(A, AT, stream);  // ★ 纯 CUDA 实现
//         // 健壮性：失败回退 power(20,1e-3)
//         if (!std::isfinite(L) || L <= 0.0 || L > 1e12) {
//             fprintf(stderr,
//                 "[StepSize] one_inf bound suspicious (L=%.6e). Fallback to power(20,1e-3).\n", L);
//             L = estimate_maximum_singular_value(
//                     state->sparse_handle, state->blas_handle, A, AT, 20, 1e-3);
//         }
//         break;
//     }
//     case STEP_SIZE_HYBRID:
//     default:
//     {
//         double L1 = estimate_sigma_max_upper_1inf(A, AT, stream);  // 上界
//         int    it = (params->hybrid_refine_iterations > 0) ? params->hybrid_refine_iterations : 10;
//         double tl = (params->power_tolerance > 0.0)        ? params->power_tolerance        : 1e-3;
//         double L2 = (it > 0)
//                     ? estimate_maximum_singular_value(state->sparse_handle, state->blas_handle, A, AT, it, tl)
//                     : L1;
//         // 正常取更小的那个；若上界失败（NaN/Inf/≤0/离谱），用 L2
//         if (!std::isfinite(L1) || L1 <= 0.0 || L1 > 1e12) L = L2;
//         else                                              L = fmin(L1, L2);
//         break;
//     }
//     }

//     if (!std::isfinite(L) || L <= 0.0) {
//         fprintf(stderr, "[StepSize] INVALID L (method=%s): L=%.17g -> fallback L=1.\n",
//                 method_name(params->step_size_method), L);
//         L = 1.0;
//     }

//     state->step_size = safety / L;

//     if (params->verbose) {
//         fprintf(stdout, "[StepSize] method=%s  safety=%.6g  L=%.6e  step=%.6e\n",
//                 method_name(params->step_size_method), safety, L, state->step_size);
//         fflush(stdout);
//     }

//     // 可选：计算参考幂迭代，仅打印
//     if (params->stepsize_power_reference) {
//         int    itR  = (params->stepsize_reference_max_iterations > 0) ? params->stepsize_reference_max_iterations : 5000;
//         double tolR = (params->stepsize_reference_tolerance  > 0.0)   ? params->stepsize_reference_tolerance      : 1e-4;
//         double Lref = estimate_maximum_singular_value(state->sparse_handle, state->blas_handle, A, AT, itR, tolR);
//         if (Lref > 0.0 && std::isfinite(Lref)) {
//             double step     = state->step_size;
//             double step_ref = safety / Lref;
//             double L_est    = safety / step;
//             double rel_s    = fabs(step - step_ref) / step_ref;
//             double rel_L    = fabs(L_est - Lref)    / Lref;
//             fprintf(stdout,
//                 "[StepSize] ref(power) L_ref=%.6e  step_ref=%.6e  rel_err_step=%.3e  rel_err_L=%.3e\n",
//                 Lref, step_ref, rel_s, rel_L);
//         }
//     }

//     // 原来的 primal weight 逻辑
//     if (params->bound_objective_rescaling) {
//         state->primal_weight = 1.0;
//     } else {
//         state->primal_weight = state->objective_vector_norm / state->constraint_bound_norm;
//     }
//     state->best_primal_weight = state->primal_weight;
// }
////////////////////////////////////////////////////////////////////////////////////////////
// === 计算 CSR 每行绝对值和的 kernel ===
__global__ void row_abs_sum_kernel(const int* __restrict__ row_ptr,
                                   const double* __restrict__ val,
                                   int m, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double s = 0.0;
        int beg = row_ptr[i];
        int end = row_ptr[i + 1];
        #pragma unroll 4
        for (int p = beg; p < end; ++p) s += fabs(val[p]);
        out[i] = s;
    }
}

// === 用 Thrust 取“最大行绝对值和”（即 ||M||_inf）===
// 要求：M->row_ptr / M->val 是设备指针（本项目就是）
static double csr_max_row_abs_sum_thrust(const cu_sparse_matrix_csr_t* M, cudaStream_t stream) {
    const int m = M->num_rows;
    if (m <= 0) return 0.0;

    double* d_row_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_sums, sizeof(double) * (size_t)m));

    dim3 block(256);
    dim3 grid((m + block.x - 1) / block.x);
    row_abs_sum_kernel<<<grid, block, 0, stream>>>(M->row_ptr, M->val, m, d_row_sums);
    CUDA_CHECK(cudaGetLastError());

    // 用与 cuSPARSE 同一条 stream 的执行策略
    thrust::device_ptr<double> beg(d_row_sums);
    thrust::device_ptr<double> end = beg + m;
    double mx = thrust::reduce(thrust::cuda::par.on(stream), beg, end, 0.0, thrust::maximum<double>());

    CUDA_CHECK(cudaFree(d_row_sums));
    return mx;
}

// === \|A\|_2 的上界：sqrt(\|A\|_1 * \|A\|_inf)
// 其中 \|A\|_1 = max col sum(A) = \|A^T\|_inf（AT 为 A 的转置，已是 CSR）
static double estimate_sigma_max_upper_1inf_thrust(const cu_sparse_matrix_csr_t* A,
                                                   const cu_sparse_matrix_csr_t* AT,
                                                   cudaStream_t stream) {
    double infA = csr_max_row_abs_sum_thrust(A,  stream); // \|A\|_inf
    double oneA = csr_max_row_abs_sum_thrust(AT, stream); // \|A\|_1 = \|A^T\|_inf

    if (!std::isfinite(infA) || infA < 0.0) infA = 0.0;
    if (!std::isfinite(oneA) || oneA < 0.0) oneA = 0.0;

    double prod = oneA * infA;
    if (!(std::isfinite(prod)) || prod <= 0.0) return 0.0;
    return sqrt(prod);
}
// ||A||_2 的一阶上界：sqrt(||A||_1 * ||A||_inf)
// 其中 ||A||_1 用 AT 的“最大行和”（= A 的最大列和）来等价计算
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
                                                   const pdhg_parameters_t *params)
{
    const cu_sparse_matrix_csr_t* A  = state->constraint_matrix;
    const cu_sparse_matrix_csr_t* AT = state->constraint_matrix_t;

    // —— 与 cuSPARSE 同流，并让 cuBLAS 使用同一条流 —— //
    cudaStream_t stream = 0;
    CUSPARSE_CHECK(cusparseGetStream(state->sparse_handle, &stream));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle, stream));

    auto valid_pos = [](double x)->bool { return std::isfinite(x) && (x > 0.0); };
    auto name_of = [](int m)->const char* {
        switch ((step_size_method_t)m) {
            case STEP_SIZE_POWER_ITERATION: return "power";
            case STEP_SIZE_ONE_INF_UPPER  : return "one_inf";
            case STEP_SIZE_CONSTANT       : return "constant";
            case STEP_SIZE_HYBRID         : return "hybrid";
            default: return "unknown";
        }
    };

    // —— 安全系数：兜底为 0.998（或按需改成 0.95） —— //
    double safety = params->step_size_safety;
    if (!std::isfinite(safety) || safety <= 1e-6 || safety >= 1.0) safety = 0.998;

    // —— 计时事件 —— //
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    float  t_oneinf_ms = 0.0f;
    float  t_power_ms  = 0.0f;
    float  t_ref_ms    = 0.0f;
    float t_constant_ms = 0.0f;

    // —— 结果容器 —— //
    double L_final = 0.0;
    double L_oneinf = 0.0;
    double L_power  = 0.0;
    // double L_constant = 0.0;

    // ===== 分支：计算 L =====
    switch ((step_size_method_t)params->step_size_method) {
    case STEP_SIZE_POWER_ITERATION: {
        // 纯幂迭代
        const int    it  = (params->power_max_iterations > 0) ? params->power_max_iterations : 5000;
        const double tol = (params->power_tolerance > 0.0)     ? params->power_tolerance     : 1e-4;
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_power = estimate_maximum_singular_value(
                    state->sparse_handle, state->blas_handle, A, AT, it, tol);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_power_ms, e0, e1));
        L_final = valid_pos(L_power) ? L_power : 1.0;
        break;
    }

    case STEP_SIZE_ONE_INF_UPPER: {
        // 上界 sqrt(||A||_1 * ||A||_inf)
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_oneinf = estimate_sigma_max_upper_1inf_thrust(A, AT, stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_oneinf_ms, e0, e1));
        L_final = valid_pos(L_oneinf) ? L_oneinf : 1.0;
        break;
    }

    case STEP_SIZE_CONSTANT: {
        // L_constant = 1.0;
        // const int    it  = (params->power_max_iterations > 0) ? params->power_max_iterations : 5000;
        // const double tol = (params->power_tolerance > 0.0)     ? params->power_tolerance     : 1e-4;
        // CUDA_CHECK(cudaEventRecord(e0, stream));
        // L_power = estimate_maximum_singular_value(
        //             state->sparse_handle, state->blas_handle, A, AT, it, tol);
        // CUDA_CHECK(cudaEventRecord(e1, stream));
        // CUDA_CHECK(cudaEventSynchronize(e1));
        // CUDA_CHECK(cudaEventElapsedTime(&t_constant_ms, e0, e1));
        // L_final = min(L_constant, L_power);
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_final = 1.0;
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_constant_ms, e0, e1));
        break;
    }
    case STEP_SIZE_HYBRID:
    default: {
        // 1) 先算 one-inf 上界（保证不低估）
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_oneinf = estimate_sigma_max_upper_1inf_thrust(A, AT, stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_oneinf_ms, e0, e1));
        if (!valid_pos(L_oneinf)) L_oneinf = 1.0;

        // 2) 少量幂迭代 refine
        const int    it_ref  = (params->hybrid_refine_iterations > 0) ? params->hybrid_refine_iterations : 10;
        const double tol_ref = (params->power_tolerance > 0.0)        ? params->power_tolerance        : 1e-3;

        if (it_ref > 0) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            L_power = estimate_maximum_singular_value(
                        state->sparse_handle, state->blas_handle, A, AT, it_ref, tol_ref);
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&t_power_ms, e0, e1));
            if (!valid_pos(L_power)) L_power = L_oneinf;
        } else {
            L_power = L_oneinf;
        }

        // 3) “min-guarded”策略（原 hybrid 精神：尽量用幂迭代更准的值，但**带安全栅**）
        //    只有当 L_power 与上界足够接近时才允许把 L 降低，否则仍用上界。
        //    guard_rel 默认 2%，也可用 power_tolerance 作为替代来源（限制在 [1%, 10%]）
        double guard_rel = 0.02;
        if (params->power_tolerance > 0.0 && std::isfinite(params->power_tolerance)) {
            guard_rel = fmax(0.01, fmin(0.10, params->power_tolerance));
        }
        const double L_guard = (1.0 - guard_rel) * L_oneinf;

        if (L_power >= L_guard) {
            // 幂迭代结果“足够接近”上界，采用较小的 L（步长更大，但仍受 guard 约束）
            L_final = L_power;
        } else {
            // 幂迭代远低于上界，可能低估谱范数 → 回退到上界，保证完全稳定
            L_final = L_oneinf;
        }
        break;
    }
    }

    if (!valid_pos(L_final)) L_final = 1.0;
    state->step_size = safety / L_final;

    // ===== 可选：参考幂迭代（只打印，不参与步长） =====
    double L_ref = 0.0;
    bool have_ref = false;
    if (params->stepsize_power_reference) {
        const int    itR  = (params->stepsize_reference_max_iterations > 0)
                            ? params->stepsize_reference_max_iterations : 5000;
        const double tolR = (params->stepsize_reference_tolerance  > 0.0)
                            ? params->stepsize_reference_tolerance      : 1e-4;

        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_ref = estimate_maximum_singular_value(
                    state->sparse_handle, state->blas_handle, A, AT, itR, tolR);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_ref_ms, e0, e1));

        have_ref = valid_pos(L_ref);
    }

    // ===== 打印 =====
    if (params->verbose) {
        const char* mname = name_of(params->step_size_method);
        if ((step_size_method_t)params->step_size_method == STEP_SIZE_ONE_INF_UPPER) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_oneinf=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_oneinf_ms);
        } else if ((step_size_method_t)params->step_size_method == STEP_SIZE_POWER_ITERATION) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_power=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_power_ms);
        } else if ((step_size_method_t)params->step_size_method == STEP_SIZE_CONSTANT) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_constant=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_constant_ms);
        } else {
            // HYBRID：把两条都打出来，便于对比
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L_oneinf=%.6e  L_power=%.6e  -> L=%.6e  step=%.6e  "
                "(t_oneinf=%.3f ms, t_power=%.3f ms)\n",
                mname, safety, L_oneinf, L_power, L_final, state->step_size, t_oneinf_ms, t_power_ms);
        }

        if (have_ref) {
            const double step_ref = safety / L_ref;
            const double rel_err_step = fabs(state->step_size - step_ref) / step_ref;
            const double rel_err_L    = fabs(L_final - L_ref) / L_ref;
            fprintf(stdout,
                "[StepSize] ref(power) L_ref=%.6e  step_ref=%.6e  rel_err_step=%.3e  rel_err_L=%.3e  (t_ref=%.3f ms)\n",
                L_ref, step_ref, rel_err_step, rel_err_L, t_ref_ms);
        }
        fflush(stdout);
    }

    // ===== 原有 primal_weight 逻辑 =====
    if (params->bound_objective_rescaling) {
        state->primal_weight = 1.0;
    } else {
        state->primal_weight = state->objective_vector_norm / state->constraint_bound_norm;
    }
    state->best_primal_weight = state->primal_weight;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}


////////////////////////////////////////////////////////////////////////////////////////////
// static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
// {
//     double max_sv = estimate_maximum_singular_value(state->sparse_handle, state->blas_handle, state->constraint_matrix, state->constraint_matrix_t, 5000, 1e-4);
//     state->step_size = 0.998 / max_sv;

//     if (params->bound_objective_rescaling)
//     {
//         state->primal_weight = 1.0;
//     }
//     else
//     {
//         state->primal_weight = state->objective_vector_norm / state->constraint_bound_norm;
//     }
//     state->best_primal_weight = state->primal_weight;
// }
////////////////////////////////////////////////////////////////////////////////////////////
static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->current_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_constraints,
                                   state->delta_dual_solution,
                                   1,
                                   &dual_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_variables,
                                   state->delta_primal_solution,
                                   1,
                                   &primal_norm));
    movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_product, 1, state->delta_primal_solution, 1, &cross_term));
    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));

    if (state->primal_spmv_buffer)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
    if (state->dual_spmv_buffer)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer));

    // 2) 销毁 cuSPARSE 稠密向量/稀疏矩阵句柄（先 vec，后 mat）
    if (state->vec_primal_sol)  CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_sol));
    if (state->vec_dual_sol)    CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_sol));
    if (state->vec_primal_prod) CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_prod));
    if (state->vec_dual_prod)   CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_prod));

    if (state->matA)   CUSPARSE_CHECK(cusparseDestroySpMat(state->matA));
    if (state->matAt)  CUSPARSE_CHECK(cusparseDestroySpMat(state->matAt));

    // 3) 销毁 cuSPARSE/cuBLAS handle（最后销毁 handle）
    if (state->sparse_handle) CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle));
    if (state->blas_handle)   CUBLAS_CHECK(cublasDestroy(state->blas_handle));

    // 4) 释放 host 侧壳体结构体
    if (state->constraint_matrix)   free(state->constraint_matrix);
    if (state->constraint_matrix_t) free(state->constraint_matrix_t);



    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    lp_problem_free(info->scaled_problem);
    free(info->con_rescale);
    free(info->var_rescale);

    free(info);
}