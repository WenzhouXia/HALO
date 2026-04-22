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

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <cusparse.h>
#include <cublas_v2.h>

typedef struct
{
	int num_rows;
	int num_cols;
	int num_nonzeros;
	int *row_ptr;
	int *col_ind;
	double *val;
} cu_sparse_matrix_csr_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	double *objective_vector;
	double objective_constant;

	int *constraint_matrix_row_pointers;
	int *constraint_matrix_col_indices;
	double *constraint_matrix_values;
	int constraint_matrix_num_nonzeros;

	double *constraint_lower_bound;
	double *constraint_upper_bound;
} lp_problem_t;

typedef enum
{
	TERMINATION_REASON_UNSPECIFIED,
	TERMINATION_REASON_OPTIMAL,
	TERMINATION_REASON_PRIMAL_INFEASIBLE,
	TERMINATION_REASON_DUAL_INFEASIBLE,
	TERMINATION_REASON_TIME_LIMIT,
	TERMINATION_REASON_ITERATION_LIMIT
} termination_reason_t;

typedef struct
{
	lp_problem_t *scaled_problem;
	double *con_rescale;
	double *var_rescale;
	double con_bound_rescale;
	double obj_vec_rescale;
	double rescaling_time_sec;
} rescale_info_t;

typedef struct
{
	double artificial_restart_threshold;
	double sufficient_reduction_for_restart;
	double necessary_reduction_for_restart;
	double k_p;
	double k_i;
	double k_d;
	double i_smooth;
} restart_parameters_t;

typedef struct
{
	double eps_optimal_relative;
	double eps_feasible_relative;
	double eps_infeasible;
	double time_sec_limit;
	int iteration_limit;
	double eps_feasible_relative_primal;
	double eps_feasible_relative_dual;
	int use_dual_nnz_gate;	 // 0=关闭(默认)，1=开启
	double dual_nnz_factor;	 // 阈值系数，默认 2.0 (阈值 = factor * num_variables)
	double dual_nnz_abs_tol; // 非零判定的 |y_i| > tol，默认 0.0
} termination_criteria_t;

typedef enum
{
	STEP_SIZE_POWER_ITERATION = 0, // 幂迭代
	STEP_SIZE_ONE_INF_UPPER = 1,   // sqrt(||A||_1 * ||A||_inf)
	STEP_SIZE_HYBRID = 2,		   // 上界 + 少量幂迭代 refine
	STEP_SIZE_CONSTANT = 3,
} step_size_method_t;

typedef struct
{
	int l_inf_ruiz_iterations;
	bool has_pock_chambolle_alpha;
	double pock_chambolle_alpha;
	bool bound_objective_rescaling;
	bool verbose;
	bool verbose_time;
	int termination_evaluation_frequency;
	termination_criteria_t termination_criteria;
	restart_parameters_t restart_params;
	double reflection_coefficient;
	const double *initial_primal_unscaled;
	const double *initial_dual_unscaled;
	bool has_initial_iterate;

	///
	int step_size_method;		  // 0/1/2，见上 enum
	double step_size_safety;	  // 安全系数，默认 0.95（或保留你原来的 0.998）
	int power_max_iterations;	  // 幂迭代最大步数（power/hybrid）
	double power_tolerance;		  // 幂迭代相对/残差容忍
	int hybrid_refine_iterations; // hybrid 时的幂迭代步数（更小，例如 5~10）
	int stepsize_power_reference; // bool
	int stepsize_reference_max_iterations;
	double stepsize_reference_tolerance;
	///
} pdhg_parameters_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	double *objective_vector;
	double objective_constant;
	cu_sparse_matrix_csr_t *constraint_matrix;
	cu_sparse_matrix_csr_t *constraint_matrix_t;
	double *constraint_lower_bound;
	double *constraint_upper_bound;
	int num_blocks_primal;
	int num_blocks_dual;
	int num_blocks_primal_dual;
	double objective_vector_norm;
	double constraint_bound_norm;
	double *constraint_lower_bound_finite_val;
	double *constraint_upper_bound_finite_val;
	double *variable_lower_bound_finite_val;
	double *variable_upper_bound_finite_val;

	double *initial_primal_solution;
	double *current_primal_solution;
	double *pdhg_primal_solution;
	double *reflected_primal_solution;
	double *dual_product;
	double *initial_dual_solution;
	double *current_dual_solution;
	double *pdhg_dual_solution;
	double *reflected_dual_solution;
	double *primal_product;
	double step_size;
	double primal_weight;
	int total_count;
	bool is_this_major_iteration;
	double primal_weight_error_sum;
	double primal_weight_last_error;
	double best_primal_weight;
	double best_primal_dual_residual_gap;

	double *constraint_rescaling;
	double *variable_rescaling;
	double constraint_bound_rescaling;
	double objective_vector_rescaling;
	double *primal_slack;
	double *dual_slack;
	double rescaling_time_sec;
	double gpu_to_cpu_time_sec;
	double basic_time_sec;
	double cumulative_time_sec;

	double *primal_residual;
	double absolute_primal_residual;
	double relative_primal_residual;
	double *dual_residual;
	double absolute_dual_residual;
	double relative_dual_residual;
	double primal_objective_value;
	double dual_objective_value;
	double objective_gap;
	double relative_objective_gap;
	double max_primal_ray_infeasibility;
	double max_dual_ray_infeasibility;
	double primal_ray_linear_objective;
	double dual_ray_objective;
	termination_reason_t termination_reason;

	double *delta_primal_solution;
	double *delta_dual_solution;
	double fixed_point_error;
	double initial_fixed_point_error;
	double last_trial_fixed_point_error;
	int inner_count;

	cusparseHandle_t sparse_handle;
	cublasHandle_t blas_handle;
	size_t spmv_buffer_size;
	size_t primal_spmv_buffer_size;
	size_t dual_spmv_buffer_size;
	void *primal_spmv_buffer;
	void *dual_spmv_buffer;
	void *spmv_buffer;

	cusparseSpMatDescr_t matA;
	cusparseSpMatDescr_t matAt;
	cusparseDnVecDescr_t vec_primal_sol;
	cusparseDnVecDescr_t vec_dual_sol;
	cusparseDnVecDescr_t vec_primal_prod;
	cusparseDnVecDescr_t vec_dual_prod;

	double *ones_primal_d;
	double *ones_dual_d;

	// 新增，为了避免PID导致的爆炸
	double k_p;
	double k_i;
	double k_d;
	double previous_restart_dual_residual;
	double previous_restart_gap;
} pdhg_solver_state_t;
