#include "HPRLP.h"
#include "version.h"
#include "mps_reader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <chrono>   // 用于高精度计时
#include <stdio.h>    // 用于 printf
#include <cuda_runtime.h> // 用于 cudaMemGetInfo
/**
 * Print HPRLP startup banner with version information
 */
static void print_banner() {
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "                          HPR-LP Solver                           \n";
    std::cout << "     Halpern Peaceman-Rachford Linear Programming Solver          \n";
    std::cout << "                                                                  \n";
    std::cout << "  Version: " << HPRLP_VERSION_STRING << "                          \n";
    std::cout << "                                                                  \n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
}

/**
 * Print solver parameters
 */
static void print_parameters(const HPRLP_parameters *param) {
    std::cout << "Solver Parameters:\n";
    std::cout << "  Device:              GPU " << param->device_number << "\n";
    std::cout << "  Max Iterations:      " << param->max_iter << "\n";
    // std::cout << "  Stopping Tolerance:  " << std::scientific << std::setprecision(1) << param->stop_tol << "\n" << std::defaultfloat;
    std::cout << "  Primal Tolerance:   " << std::scientific << std::setprecision(1) << param->primal_tol << "\n" << std::defaultfloat;
    std::cout << "  Dual Tolerance:     " << std::scientific << std::setprecision(1) << param->dual_tol << "\n" << std::defaultfloat;
    std::cout << "  Gap Tolerance:      " << std::scientific << std::setprecision(1) << param->gap_tol << "\n" << std::defaultfloat;
    std::cout << "  Time Limit:          " << std::fixed << std::setprecision(1) << param->time_limit << " seconds\n" << std::defaultfloat;
    std::cout << "  Check Interval:      " << param->check_iter << " iterations\n";
    std::cout << "  Scaling:\n";
    std::cout << "    - Ruiz:            " << (param->use_Ruiz_scaling ? "Enabled" : "Disabled") << "\n";
    std::cout << "    - Pock-Chambolle:  " << (param->use_Pock_Chambolle_scaling ? "Enabled" : "Disabled") << "\n";
    std::cout << "    - Bounds/Cost:     " << (param->use_bc_scaling ? "Enabled" : "Disabled") << "\n";
    std::cout << "\n";
}

/**
 * Initialize CUDA device (internal function)
 */
static int initialize_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[error] Failed to set CUDA device " << device_id << ": " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Note: cudaDeviceReset() removed to avoid segfaults during Python exit
    // The CUDA runtime will clean up automatically when the process exits
    
    // std::cout << "[info] Using CUDA device " << device_id << std::endl;
    return 0;
}

// // 定义一个统一的时钟时间点类型别名
// using HprlpClockTimePoint = std::chrono::high_resolution_clock::time_point;

// /**
//  * @brief 获取当前高精度时间点
//  * (这是 C++11 兼容写法, 请用它替换掉您旧的 time_now)
//  */
// HprlpClockTimePoint time_now() {
//     return std::chrono::high_resolution_clock::now();
// }

// /**
//  * @brief 计算从 'start' 到现在的秒数 (double)
//  * (请用它替换掉您旧的 time_since)
//  */
// double time_since(HprlpClockTimePoint start) {
//     // 'auto' 在函数内部是 C++11 安全的
//     auto end = std::chrono::high_resolution_clock::now(); 
//     return std::chrono::duration<double>(end - start).count();
// }


// --- 内存打印辅助 (来自 solver.cu) ---
static inline double toMiB(std::size_t b){ return b / (1024.0 * 1024.0); }

// 在任意位置调用以打印当前 GPU 显存用量
static inline void print_gpu_mem_once(const char* tag = "mem_tag", bool sync_before_query = true) {
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
static inline double get_gpu_mem_usage_mib(bool sync_before_query = true) {
    if (sync_before_query) cudaDeviceSynchronize();
    std::size_t freeB = 0, totalB = 0;
    cudaError_t st = cudaMemGetInfo(&freeB, &totalB);
    if (st != cudaSuccess) {
        return 0.0;
    }
    std::size_t usedB = totalB - freeB;
    return usedB / (1024.0 * 1024.0);
}
// void debug_print_model(const LP_info_cpu* info) {
//     std::cout << "=== Debug LP_info_cpu ===\n";
//     std::cout << "m = " << info->m << ", n = " << info->n << "\n";

//     // 打印 A 的 CSR（通过 info->A）
//     std::cout << "A (CSR):\n";
//     for (int i = 0; i < info->m; ++i) {
//         int start = info->A->rowPtr[i];
//         int end   = info->A->rowPtr[i+1];
//         for (int k = start; k < end; ++k) {
//             int j = info->A->colIndex[k];
//             double v = info->A->value[k];
//             std::cout << "  A[" << i << "," << j << "] = " << v << "\n";
//         }
//     }

//     std::cout << "AL: ";
//     for (int i = 0; i < info->m; ++i) std::cout << info->AL[i] << " ";
//     std::cout << "\nAU: ";
//     for (int i = 0; i < info->m; ++i) std::cout << info->AU[i] << " ";
//     std::cout << "\nl: ";
//     for (int j = 0; j < info->n; ++j) std::cout << info->l[j] << " ";
//     std::cout << "\nu: ";
//     for (int j = 0; j < info->n; ++j) std::cout << info->u[j] << " ";
//     std::cout << "\nc: ";
//     for (int j = 0; j < info->n; ++j) std::cout << info->c[j] << " ";
//     std::cout << "\n=========================\n";
// }
// HPRLP_results HPRLP_main_solve(const LP_info_cpu *lp_info_cpu, const HPRLP_parameters *param) {

//     // Print startup banner
//     print_banner();
    
//     // Print parameters
//     print_parameters(param);

//     // Initialize CUDA device
//     initialize_device(param->device_number);

//     // load LP param (A, c, l, u, AL, AU) from host to device
//     LP_info_gpu lp_info_gpu;
//     copy_lpinfo_to_device(lp_info_cpu, &lp_info_gpu);

    
//     // allocate memory for the variables used in the algorithm
//     HPRLP_workspace_gpu workspace;
//     workspace.m = lp_info_cpu->m;
//     workspace.n = lp_info_cpu->n;
//     allocate_memory(&workspace, &lp_info_gpu, lp_info_cpu);

//     // Scaling part: 1.Ruiz_Scaling, 2.PC_Scaling, 3.bc_scaling
//     Scaling_info scaling_info;
//     // scaling(&lp_info_gpu, &scaling_info, param, workspace.cublasHandle);
//     scaling(&lp_info_gpu, &scaling_info, param, &workspace); // 传入整个 workspace

//     /* ----------The alg time is started!---------- */
//     auto t_start_alg = time_now();

//     // debug_print_model(lp_info_cpu);
//     // workspace.lambda_max = power_method_cusparse(&workspace, 6) * 1.01;
//     workspace.lambda_max = power_method_cusparse(&workspace) * 1.01;

//     // ### Initialization ###
HPRLP_results HPRLP_main_solve(const LP_info_cpu *lp_info_cpu, const HPRLP_parameters *param) {

    // ===== 计时：第 1 步 - 声明耗时变量 =====
    double t_init_device = 0.0, t_copy_lp = 0.0, t_alloc_mem = 0.0;
    double t_scaling = 0.0, t_power_method = 0.0, t_setup = 0.0;
    
    // 记录总启动时间
    auto t_start_setup = time_now();

    // Print startup banner
    print_banner();
    
    // Print parameters
    print_parameters(param);

    // ===== 计时：`initialize_device` =====
    auto t_a = time_now();
    initialize_device(param->device_number);
    t_init_device = time_since(t_a);

    // load LP param (A, c, l, u, AL, AU) from host to device
    LP_info_gpu lp_info_gpu;
    
    // ===== 计时：`copy_lpinfo_to_device` =====
    t_a = time_now();
    copy_lpinfo_to_device(lp_info_cpu, &lp_info_gpu);
    t_copy_lp = time_since(t_a);
    
    // allocate memory for the variables used in the algorithm
    HPRLP_workspace_gpu workspace;
    workspace.m = lp_info_cpu->m;
    workspace.n = lp_info_cpu->n;

    // ===== 计时：`allocate_memory` =====
    t_a = time_now();
    allocate_memory(&workspace, &lp_info_gpu, lp_info_cpu);
    t_alloc_mem = time_since(t_a);

    // ===== 内存：第 1 步 - 在分配后打印内存 =====
    // 这是测量峰值占用的最佳位置
    print_gpu_mem_once("After allocate_memory", /*sync_before_query=*/true);
    double peak_mem_recorded = get_gpu_mem_usage_mib(false); // 刚刚已经同步过了，这里设为 false 即可
    std::printf("[GPU MEM]: peak=%.2f MiB \n", peak_mem_recorded);

    // Scaling part: 1.Ruiz_Scaling, 2.PC_Scaling, 3.bc_scaling
    Scaling_info scaling_info;
    
    // ===== 计时：`scaling` =====
    t_a = time_now();
    scaling(&lp_info_gpu, &scaling_info, param, &workspace); // 传入整个 workspace
    t_scaling = time_since(t_a);

    /* ----------The alg time is started!---------- */
    // 您的 `t_start_alg` 计时器现在用于测量 *循环* 时间，这很好
    auto t_start_alg = time_now(); 

    // ===== 计时：`power_method_cusparse` =====
    t_a = time_now();
    workspace.lambda_max = power_method_cusparse(&workspace) * 1.01;
    t_power_method = time_since(t_a);

    // ===== 计时：第 2 步 - 计算总 setup 时间 =====
    t_setup = time_since(t_start_setup); // 这是总的 setup 时间
    HPRLP_residuals residuals;
    if (scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8) {
        workspace.sigma = scaling_info.norm_b / scaling_info.norm_c;
    }
    else{
        workspace.sigma = 1.0;
    }
    HPRLP_restart restart_info;
    restart_info.best_sigma = workspace.sigma;

    HPRLP_results output;
    bool first_4 = true;
    bool first_6 = true;
    bool first_8 = true;

    std::cout << " iter     errRp        errRd         p_obj            d_obj          gap         sigma       time\n" << std::flush;

    for (int iter = 0 ; iter < param->max_iter ; iter ++) {

        bool print_flag = ((iter % step(iter) == 0) ||                
                            (iter == param->max_iter) ||                    // reach maximum allowed iterations
                            time_since(t_start_alg) > param->time_limit);   // reach maximum run time

        if (iter % param->check_iter == 0 || print_flag) {
            collect_residuals(&workspace, &lp_info_gpu, &scaling_info, &residuals, iter);    // KKT residuals computation
            residuals.is_updated = true;
        } 
        else {
            residuals.is_updated = false;
        }

        // check stopping criterion
        std::string status = check_stopping(&residuals, iter, t_start_alg, param);

        // check restart
        check_restart(&restart_info, iter, param->check_iter, workspace.sigma);

        if (print_flag || status != "CONTINUE") {
            std::cout << std::setw(5) << iter << "    "
                      << std::scientific << std::setprecision(2)
                      << residuals.err_Rp_org_bar << "    "
                      << residuals.err_Rd_org_bar << "    "
                      << std::scientific << std::setprecision(6) << std::showpos << residuals.primal_obj_bar << "    "
                      << residuals.dual_obj_bar << "    "
                      << std::scientific << std::setprecision(2) << std::noshowpos << residuals.rel_gap_bar << "    "
                      << workspace.sigma << "      "
                      << std::fixed << std::setprecision(2)
                      << time_since(t_start_alg) << "\n" << std::flush;
        }

        if (first_4 && residuals.KKTx_and_gap_org_bar < 1e-4) {
            output.iter4 = iter;
            output.time4 = time_since(t_start_alg);
            first_4 = false;
            std::cout << "Residual < 1e-4 at iter = " << iter << "\n" << std::flush;
        }
        if (first_6 && residuals.KKTx_and_gap_org_bar < 1e-6) {
            output.iter6 = iter;
            output.time6 = time_since(t_start_alg);
            first_6 = false;
            std::cout << "Residual < 1e-6 at iter = " << iter << "\n" << std::flush;
        }
        if (first_8 && residuals.KKTx_and_gap_org_bar < 1e-8) {
            output.iter8 = iter;
            output.time8 = time_since(t_start_alg);
            first_8 = false;
            std::cout << "Residual < 1e-8 at iter = " << iter << "\n" << std::flush;
        }

        if (status != "CONTINUE") {
            strncpy(output.status, status.c_str(), sizeof(output.status) - 1);
            output.status[sizeof(output.status) - 1] = '\0';  // Ensure null termination
            output.iter = iter;
            output.gap = residuals.rel_gap_bar;
            output.residuals = residuals.KKTx_and_gap_org_bar;
            output.primal_residuals = residuals.err_Rp_org_bar;
            output.dual_residuals = residuals.err_Rd_org_bar;
            output.primal_obj = residuals.primal_obj_bar;
            output.time = time_since(t_start_alg);
            
            // =================== [新增] ===================
            output.peak_mem = peak_mem_recorded;
            // =============================================

            output.time4 = (output.time4 == 0.0) ? output.time : output.time4;
            output.time6 = (output.time6 == 0.0) ? output.time : output.time6;
            output.time8 = (output.time8 == 0.0) ? output.time : output.time8;
            output.iter4 = (output.iter4 == 0) ? output.iter : output.iter4;
            output.iter6 = (output.iter6 == 0) ? output.iter : output.iter6;
            output.iter8 = (output.iter8 == 0) ? output.iter : output.iter8;

            // Collect solution vectors from GPU workspace
            collect_solution(&workspace, &scaling_info, &output);

            // Print solution summary
            std::cout << "\n=== Solution Summary ===\n";
            std::cout << "Status: " << output.status << "\n";
            std::cout << "Iterations: " << output.iter << "\n";
            std::cout << "Time: " << output.time << " seconds\n";
            std::cout << "Primal Objective: " << std::scientific << std::setprecision(12) << output.primal_obj << "\n";
            std::cout << "Residual: " << output.residuals << "\n";
            std::cout << "\n";
            // ===== 计时：第 3 步 - 打印初始化耗时 =====
            // 在返回结果前，打印详细的 setup 耗时
            std::printf("[TIMING] Setup Total=%.3fs (InitDevice=%.3f, CopyLP=%.3f, AllocMem=%.3f, Scaling=%.3f, PowerMethod=%.3f)\n",
                   t_setup, t_init_device, t_copy_lp, t_alloc_mem, t_scaling, t_power_method);
            std::fflush(stdout);
            // Clean up GPU resources to prevent memory leaks
            // Synchronize to ensure all CUDA operations complete before cleanup
            cudaDeviceSynchronize();
            
            free_workspace(&workspace);
            free_scaling_info(&scaling_info);
            free_lp_info(&lp_info_gpu);
            
            // Final synchronization to ensure cleanup is complete
            cudaDeviceSynchronize();

            return output;
        }


        // update sigma
        update_sigma(&restart_info, &workspace, &residuals);

        do_restart(&workspace, &restart_info);

        workspace.check = ((iter + 1) % param->check_iter == 0 || restart_info.restart_flag > 0);
        workspace.check = (workspace.check || (iter + 1) % step(iter + 1) == 0);

        HPRLP_FLOAT fact1 = 1.0 / (restart_info.inner + 2.0);
        HPRLP_FLOAT fact2 = 1.0 - fact1;

        update_z_x(&workspace, fact1, fact2);
        update_y(&workspace, fact1, fact2);

        if ((iter + 1) % param->check_iter == 0) {
            restart_info.current_gap = compute_weighted_norm(&workspace);
        }

        if(restart_info.restart_flag > 0) {
            restart_info.last_gap = compute_weighted_norm(&workspace);
        }

        restart_info.inner += 1;
    }
}

/* ============================================================================
 * New Model-Based API Implementation (v0.2+)
 * ============================================================================
 */

/**
 * Create an LP model from raw arrays
 */
LP_info_cpu* create_model_from_arrays(int m, int n, int nnz,
                                      const int *rowPtr, const int *colIndex, 
                                      const HPRLP_FLOAT *values,
                                      const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                      const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                      const HPRLP_FLOAT *c,
                                      const HPRLP_FLOAT *x_init_data,
                                      const HPRLP_FLOAT *y_init_data,
                                      bool is_csc)
{
    // Validate inputs
    if (m <= 0 || n <= 0 || nnz <= 0) {
        std::cerr << "[error] Invalid dimensions: m=" << m << ", n=" << n << ", nnz=" << nnz << std::endl;
        return nullptr;
    }
    
    if (!rowPtr || !colIndex || !values || !AL || !AU || !l || !u || !c) {
        std::cerr << "[error] Null pointer in input arrays" << std::endl;
        return nullptr;
    }
    
    // Allocate model structure
    LP_info_cpu* model = new LP_info_cpu;
    if (!model) {
        std::cerr << "[error] Failed to allocate model structure" << std::endl;
        return nullptr;
    }
    
    // Initialize fields
    model->A = nullptr;
    model->m = 0;
    model->n = 0;
    model->obj_constant = 0.0;
    model->x_init = nullptr; // 默认初始化为 nullptr
    model->y_init = nullptr; // 默认初始化为 nullptr
    // Create CSRMatrix structure
    CSRMatrix csr_input;
    if (is_csc) {
        // Convert CSC to CSR: need to transpose
        // CSC format means rowPtr is actually column pointers, colIndex is row indices
        // We need to convert this to CSR format
        
        // Create a temporary sparseMatrix for CSC (which is CSR of A^T)
        sparseMatrix csc_sparse;
        csc_sparse.row = n;  // CSC: rows in storage = columns in A^T  
        csc_sparse.col = m;  // CSC: cols in storage = rows in A^T
        csc_sparse.numElements = nnz;
        csc_sparse.rowPtr = (int*)malloc((n + 1) * sizeof(int));
        csc_sparse.colIndex = (int*)malloc(nnz * sizeof(int));
        csc_sparse.value = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));
        
        if (!csc_sparse.rowPtr || !csc_sparse.colIndex || !csc_sparse.value) {
            std::cerr << "[error] Memory allocation failed for CSC matrix" << std::endl;
            if (csc_sparse.rowPtr) free(csc_sparse.rowPtr);
            if (csc_sparse.colIndex) free(csc_sparse.colIndex);
            if (csc_sparse.value) free(csc_sparse.value);
            delete model;
            return nullptr;
        }
        
        memcpy(csc_sparse.rowPtr, rowPtr, (n + 1) * sizeof(int));
        memcpy(csc_sparse.colIndex, colIndex, nnz * sizeof(int));
        memcpy(csc_sparse.value, values, nnz * sizeof(HPRLP_FLOAT));
        
        // Transpose to get CSR (transpose of A^T = A)
        sparseMatrix csr_sparse;
        CSR_transpose_host(csc_sparse, &csr_sparse);
        
        // Free temporary CSC
        free(csc_sparse.rowPtr);
        free(csc_sparse.colIndex);
        free(csc_sparse.value);
        
        // Now convert sparseMatrix to CSRMatrix
        csr_input.nrows = m;
        csr_input.ncols = n;
        csr_input.nnz = nnz;
        csr_input.row_ptr = csr_sparse.rowPtr;
        csr_input.col_idx = csr_sparse.colIndex;
        csr_input.values = csr_sparse.value;
    } else {
        // Already CSR, just copy
        csr_input.nrows = m;
        csr_input.ncols = n;
        csr_input.nnz = nnz;
        csr_input.row_ptr = (int*)malloc((m + 1) * sizeof(int));
        csr_input.col_idx = (int*)malloc(nnz * sizeof(int));
        csr_input.values = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));
        
        if (!csr_input.row_ptr || !csr_input.col_idx || !csr_input.values) {
            std::cerr << "[error] Memory allocation failed for CSR matrix" << std::endl;
            if (csr_input.row_ptr) free(csr_input.row_ptr);
            if (csr_input.col_idx) free(csr_input.col_idx);
            if (csr_input.values) free(csr_input.values);
            delete model;
            return nullptr;
        }
        
        memcpy(csr_input.row_ptr, rowPtr, (m + 1) * sizeof(int));
        memcpy(csr_input.col_idx, colIndex, nnz * sizeof(int));
        memcpy(csr_input.values, values, nnz * sizeof(HPRLP_FLOAT));
    }
    
    // Call build_model_from_arrays to build the model directly (no preprocessing)
    try {
        build_model_from_arrays(&csr_input, AL, AU, l, u, c, 0.0, model);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to build model: " << e.what() << std::endl;
        free(csr_input.row_ptr);
        free(csr_input.col_idx);
        free(csr_input.values);
        delete model;
        return nullptr;
    }
    
    // Free temporary CSR matrix
    free(csr_input.row_ptr);
    free(csr_input.col_idx);
    free(csr_input.values);
    
    // Verify model was created successfully
    if (!model->A || model->m <= 0 || model->n <= 0) {
        std::cerr << "[error] Model creation failed" << std::endl;
        free_lp_info_cpu(model);
        delete model;
        return nullptr;
    }
    // --- 新增逻辑：复制初始解 ---
    if (x_init_data != nullptr) {
        model->x_init = (HPRLP_FLOAT*)malloc(n * sizeof(HPRLP_FLOAT));
        if (model->x_init) {
            memcpy(model->x_init, x_init_data, n * sizeof(HPRLP_FLOAT));
        } else {
            std::cerr << "[error] Failed to allocate memory for x_init" << std::endl;
            // ... (执行错误处理, 释放已分配资源)
        }
    }

    if (y_init_data != nullptr) {
        model->y_init = (HPRLP_FLOAT*)malloc(m * sizeof(HPRLP_FLOAT));
        if (model->y_init) {
            memcpy(model->y_init, y_init_data, m * sizeof(HPRLP_FLOAT));
        } else {
            std::cerr << "[error] Failed to allocate memory for y_init" << std::endl;
            // ... (执行错误处理)
        }
    }
    return model;
}

/**
 * Create an LP model from an MPS file
 */
LP_info_cpu* create_model_from_mps(const char* mps_file_path) {
    // Validate input
    if (!mps_file_path) {
        std::cerr << "[error] Null MPS file path pointer" << std::endl;
        return nullptr;
    }
    
    // Allocate model structure
    LP_info_cpu* model = new LP_info_cpu;
    if (!model) {
        std::cerr << "[error] Failed to allocate model structure" << std::endl;
        return nullptr;
    }
    
    // Initialize fields
    model->A = nullptr;
    model->m = 0;
    model->n = 0;
    
    // Call build_model_from_mps to read MPS file and populate model (no preprocessing)
    try {
        build_model_from_mps(mps_file_path, model);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to read MPS file: " << e.what() << std::endl;
        delete model;
        return nullptr;
    }
    
    // Verify model was created successfully
    if (!model->A || model->m <= 0 || model->n <= 0) {
        std::cerr << "[error] Invalid model from MPS file" << std::endl;
        free_lp_info_cpu(model);
        delete model;
        return nullptr;
    }
    
    return model;
}

/**
 * Solve an LP model with given parameters
 */
HPRLP_results solve(const LP_info_cpu *model, const HPRLP_parameters *param) {
    // Validate input
    if (!model) {
        std::cerr << "[error] Null model pointer" << std::endl;
        HPRLP_results error_result;
        strncpy(error_result.status, "ERROR", sizeof(error_result.status) - 1);
        error_result.status[sizeof(error_result.status) - 1] = '\0';
        error_result.iter = 0;
        error_result.time = 0.0;
        error_result.primal_obj = 0.0;
        error_result.residuals = 0.0;
        error_result.gap = 0.0;
        error_result.x = nullptr;
        error_result.y = nullptr;
        return error_result;
    }
    
    // Use default parameters if param is NULL
    HPRLP_parameters default_param;
    const HPRLP_parameters *actual_param = param ? param : &default_param;
    
    // Solve the LP problem
    return HPRLP_main_solve(model, actual_param);
}

/**
 * Free an LP model
 */
void free_model(LP_info_cpu *model) {
    if (!model) return;
    
    // Free LP data structures
    free_lp_info_cpu(model);
    
    // Free the model structure itself
    delete model;
}