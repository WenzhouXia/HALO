#include "scaling.h"

void debug_print_gpu_vector(HPRLP_FLOAT* d_vec, int size, const char* vecName) {
    if (size <= 0) {
        std::cout << "--- Debug Print Vector: " << vecName << " (size 0 or invalid)" << std::endl;
        return;
    }

    // 1. 分配主机 (CPU) 内存
    HPRLP_FLOAT* h_vec = new HPRLP_FLOAT[size];

    // 2. 将数据从 GPU 拷贝到 CPU
    cudaError_t err = cudaMemcpy(h_vec, d_vec, size * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        std::cerr << "!!! 错误: cudaMemcpy " << vecName << " 失败" << std::endl;
        delete[] h_vec;
        return;
    }

    // 3. 打印向量内容 (最多打印 10 个)
    std::cout << "--- Debug Print Vector: " << vecName << " (size " << size << ") ---" << std::endl;
    std::cout << "  [";
    int print_n = (size > 10) ? 10 : size;
    for (int i = 0; i < print_n; ++i) {
        // 确保使用高精度打印
        std::cout << std::fixed << std::setprecision(6) << h_vec[i];
        if (i < print_n - 1) {
            std::cout << ", ";
        }
    }
    if (size > print_n) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;

    // 4. 清理主机内存
    delete[] h_vec;
}
void debug_print_sparse_matrix(sparseMatrix* A, const char* matrixName) {
    if (A == nullptr) {
        std::cout << "--- Debug Print: " << matrixName << " (ERROR: sparseMatrix* A is null) ---" << std::endl;
        return;
    }

    std::cout << "--- Debug Print: " << matrixName << " ---" << std::endl;
    
    // 1. 从结构体中获取元数据和 GPU (device) 指针
    int rows = A->row;
    int cols = A->col;
    int nnz = A->numElements;
    
    int* d_rowOffsets = A->rowPtr;
    int* d_colInd = A->colIndex;
    HPRLP_FLOAT* d_values = A->value;

    if (d_rowOffsets == nullptr || d_colInd == nullptr || d_values == nullptr) {
        std::cout << "!!! 错误: " << matrixName << " 的 device 指针 (rowPtr/colIndex/value) 为空。" << std::endl;
        return;
    }

    std::cout << matrixName << " (CSR): " << rows << "x" << cols << " with " << nnz << " non-zeros." << std::endl;

    if (nnz <= 0) {
        std::cout << "  (Matrix has no non-zero values)" << std::endl;
        return;
    }

    // 2. 分配主机 (CPU) 内存
    int* h_rowOffsets = new int[rows + 1]; 
    int* h_colInd = new int[nnz];
    HPRLP_FLOAT* h_values = new HPRLP_FLOAT[nnz];

    // === 修正点：将 print_count 声明移到 goto 之前 ===
    int print_count = 0;
    // ===========================================


    // 3. 将数据从 GPU 拷贝到 CPU
    cudaError_t err;
    err = cudaMemcpy(h_rowOffsets, d_rowOffsets, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "!!! 错误: cudaMemcpy h_rowOffsets 失败 (cudaError: " << err << ")" << std::endl;
        goto cleanup; // 跳转
    }
    
    err = cudaMemcpy(h_colInd, d_colInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "!!! 错误: cudaMemcpy h_colInd 失败" << std::endl;
        goto cleanup; // 跳转
    }

    err = cudaMemcpy(h_values, d_values, nnz * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "!!! 错误: cudaMemcpy h_values 失败" << std::endl;
        goto cleanup; // 跳转
    }

    // 4. 遍历并打印 CSR 数据 (最多打印 20 个)
    std::cout << matrixName << " (CSR data):" << std::endl;
    // (print_count 已在上面声明)
    for (int i = 0; i < rows; ++i) { // 遍历每一行
        int rowStart = h_rowOffsets[i];
        int rowEnd = h_rowOffsets[i+1];
        
        for (int k = rowStart; k < rowEnd; ++k) {
            if (print_count < 20) { // 限制打印数量
                int col = h_colInd[k];
                if (k >= nnz) {
                     std::cerr << "!!! 错误: rowPtr 索引 " << k << " 超出 nnz " << nnz << std::endl;
                     break;
                }
                std::cout << "  " << matrixName << "[" << i << "," << col << "] = " 
                          << h_values[k] << std::endl;
                print_count++;
            } else if (print_count == 20) {
                 std::cout << "  ... (omitting remaining " << (nnz - 20) << " values)" << std::endl;
                 print_count++;
            }
        }
        if (print_count > 20) break; 
    }
    std::cout << "-----------------------------------" << std::endl;

// 5. 清理主机内存
cleanup: // 标签
    delete[] h_rowOffsets;
    delete[] h_colInd;
    delete[] h_values;
}

// void scaling(LP_info_gpu *lp_info_gpu, Scaling_info* scaling_info, const HPRLP_parameters *param, cublasHandle_t cublasHandle) {
void scaling(LP_info_gpu *lp_info_gpu, Scaling_info *scaling_info, const HPRLP_parameters *param, HPRLP_workspace_gpu *ws) {   
    // ==================================================================
    // === 在这里插入调试代码 ===
    
    // (您需要将下面的 debug_print_sparse_matrix 辅助函数定义在某个地方,
    //  例如这个文件的顶部。请注意: 这与我们上次用的 debug_print_csr_matrix 不同)
    
    // std::cout << std::defaultfloat << std::setprecision(6); // 恢复默认打印格式
    // if (lp_info_gpu != nullptr && lp_info_gpu->A != nullptr) {
    //     debug_print_sparse_matrix(lp_info_gpu->A, "A_at_start_of_scaling");
    // } else {
    //     std::cout << "!!! 错误: lp_info_gpu 或 lp_info_gpu->A 为空，无法打印矩阵。" << std::endl;
    // }
    // std::cout << std::scientific << std::setprecision(2); // 恢复您原来的打印格式
    // //
    // // === 调试代码结束 ===
    // // ==================================================================

    int m = lp_info_gpu->m;
    int n = lp_info_gpu->n;
    // printf("Scaling: m = %d, n = %d\n", m, n);

    // ... 函数的其余部分 ...

    create_zero_vector_device(scaling_info->row_norm, m);
    create_zero_vector_device(scaling_info->col_norm, n);
    create_zero_vector_device(scaling_info->l_org, n);
    create_zero_vector_device(scaling_info->u_org, n);

    HPRLP_FLOAT *rowNormA = scaling_info->row_norm;
    HPRLP_FLOAT *colNormA = scaling_info->col_norm;
    HPRLP_FLOAT *tempNorm1;
    HPRLP_FLOAT *tempNorm2;

    create_zero_vector_device(tempNorm1, m);
    create_zero_vector_device(tempNorm2, n);

    set_vector_value_device(rowNormA, m, 1.0);
    set_vector_value_device(colNormA, n, 1.0);

    vMemcpy_device(scaling_info->l_org, lp_info_gpu->l, n);
    vMemcpy_device(scaling_info->u_org, lp_info_gpu->u, n);

    HPRLP_FLOAT *b;
    CUDA_CHECK(cudaMalloc(&b, m * sizeof(HPRLP_FLOAT)));

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);
    
    scaling_info->norm_b_org = 1 + l2_norm(b, m, ws->cublasHandle);
    scaling_info->norm_c_org = 1 + l2_norm(lp_info_gpu->c, n, ws->cublasHandle);

    if (param->use_Ruiz_scaling){
        // printf("Using Ruiz scaling\n");
        for (int i = 0; i < 10; ++i) {
            // find the max value of each row of A
            CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 99);
            // === 在这里插入调试代码 ===
            // debug_print_gpu_vector(tempNorm1, m, "Ruiz_Row_Norms (tempNorm1)");
            vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);
            
            // AL = AL / tempNorm   ;   AU = AU / tempNorm
            vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
            vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

            // find the max value of each column of A which is equivalent to each row of AT
            CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 99);
            // === 在这里插入调试代码 ===
            // debug_print_gpu_vector(tempNorm2, n, "Ruiz_Col_Norms (tempNorm2)");
            vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
            mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
            mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

            // c = c / tempNorm
            vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

            // l = l * tempNorm, u = u * tempNorm
            vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
            vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
        }
    }
    
    // Pock and Chambolle scaling
    // compute the sum of each row of A, in tempNorm
    if(param->use_Pock_Chambolle_scaling){
        // printf("Using Pock and Chambolle scaling\n");
        CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 1);
        vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);

        // AL = AL / tempNorm   ;   AU = AU / tempNorm
        vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
        vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

        // compute the sum of each column of A
        CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 1);
        vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
        mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
        mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

        // c = c / tempNorm
        vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

        // l = l * tempNorm, u = u * tempNorm
        vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
        vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
    }
    // === ⬇️ 1. 在此处添加 Warm-Start 向量的 D, E 缩放 ===
    // 此时 rowNormA 存储了总的 D 缩放, colNormA 存储了总的 E 缩放
    
    // x' = E * x   (colNormA 是 E)
    // 使用 vector_dot_product (false 代表逐元素乘法)
    vector_dot_product(ws->x,      colNormA, ws->x,      n, false);
    vector_dot_product(ws->last_x, colNormA, ws->last_x, n, false);
    vector_dot_product(ws->x_bar,  colNormA, ws->x_bar,  n, false);

    // y' = D * y   (rowNormA 是 D)
    vector_dot_product(ws->y,      rowNormA, ws->y,      m, false);
    vector_dot_product(ws->last_y, rowNormA, ws->last_y, m, false);
    vector_dot_product(ws->y_bar,  rowNormA, ws->y_bar,  m, false);
    // === ⬆️ 缩放结束 ===


    if (param->use_bc_scaling){
        // printf("Using BC scaling\n");
        gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

        scaling_info->b_scale = 1 + l2_norm(b, m, ws->cublasHandle);
        scaling_info->c_scale = 1 + l2_norm(lp_info_gpu->c, n, ws->cublasHandle);

        
        // b = b / b_scale, c = c / c_scale
        const HPRLP_FLOAT bs = 1.0 / scaling_info->b_scale;
        const HPRLP_FLOAT cs = 1.0 / scaling_info->c_scale;

        CUBLAS_CHECK(cublasDscal(ws->cublasHandle, m, &bs, lp_info_gpu->AU, 1));
        CUBLAS_CHECK(cublasDscal(ws->cublasHandle, m, &bs, lp_info_gpu->AL, 1));
        CUBLAS_CHECK(cublasDscal(ws->cublasHandle, n, &bs, lp_info_gpu->l, 1));
        CUBLAS_CHECK(cublasDscal(ws->cublasHandle, n, &bs, lp_info_gpu->u, 1));
        CUBLAS_CHECK(cublasDscal(ws->cublasHandle, n, &cs, lp_info_gpu->c, 1));
    }
    else{
        scaling_info->b_scale = 1.0;
        scaling_info->c_scale = 1.0;
    }

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

    scaling_info->norm_b = l2_norm(b, m, ws->cublasHandle);
    scaling_info->norm_c = l2_norm(lp_info_gpu->c, n, ws->cublasHandle);

    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(tempNorm1));
    CUDA_CHECK(cudaFree(tempNorm2));
}


void free_scaling_info(Scaling_info *scaling_info) {
    /*
     * Free device memory allocated in scaling function.
     */
    if (!scaling_info) return;
    
    if (scaling_info->row_norm) cudaFree(scaling_info->row_norm);
    if (scaling_info->col_norm) cudaFree(scaling_info->col_norm);
    if (scaling_info->l_org) cudaFree(scaling_info->l_org);
    if (scaling_info->u_org) cudaFree(scaling_info->u_org);
}