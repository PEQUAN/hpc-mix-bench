#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>

// ========== 稀疏矩阵数据结构 ==========

// COO格式 (Coordinate format)
struct COOMatrix {
    int rows;
    int cols;
    int nnz;
    int* row_indices;
    int* col_indices;
    double* values;
    
    COOMatrix() : rows(0), cols(0), nnz(0), 
                  row_indices(nullptr), col_indices(nullptr), values(nullptr) {}
    
    ~COOMatrix() {
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
    }
};

// CSR格式 (Compressed Sparse Row)
struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    int* row_ptr;
    int* col_indices;
    double* values;
    
    CSRMatrix() : rows(0), cols(0), nnz(0),
                  row_ptr(nullptr), col_indices(nullptr), values(nullptr) {}
    
    ~CSRMatrix() {
        delete[] row_ptr;
        delete[] col_indices;
        delete[] values;
    }
};

// ========== MatrixMarket文件读取 ==========

bool readMatrixMarket(const char* filename, COOMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // 跳过注释行
    std::string line;
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    // 读取矩阵维度
    sscanf(line.c_str(), "%d %d %d", &mat.rows, &mat.cols, &mat.nnz);
    
    std::cout << "Matrix dimensions: " << mat.rows << " x " << mat.cols 
              << ", NNZ: " << mat.nnz << std::endl;
    
    // 分配内存
    mat.row_indices = new int[mat.nnz];
    mat.col_indices = new int[mat.nnz];
    mat.values = new double[mat.nnz];
    
    // 读取数据 (MatrixMarket使用1-based索引)
    for (int i = 0; i < mat.nnz; i++) {
        int row, col;
        double val;
        file >> row >> col >> val;
        mat.row_indices[i] = row - 1;
        mat.col_indices[i] = col - 1;
        mat.values[i] = val;
    }
    
    file.close();
    return true;
}

// ========== COO转CSR格式 ==========

void COOtoCSR(const COOMatrix& coo, CSRMatrix& csr) {
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    
    csr.row_ptr = new int[csr.rows + 1];
    csr.col_indices = new int[csr.nnz];
    csr.values = new double[csr.nnz];
    
    for (int i = 0; i <= csr.rows; i++) {
        csr.row_ptr[i] = 0;
    }
    
    for (int i = 0; i < coo.nnz; i++) {
        csr.row_ptr[coo.row_indices[i] + 1]++;
    }
    
    for (int i = 0; i < csr.rows; i++) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }
    
    int* current_pos = new int[csr.rows];
    for (int i = 0; i < csr.rows; i++) {
        current_pos[i] = csr.row_ptr[i];
    }
    
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_indices[i];
        int pos = current_pos[row];
        csr.col_indices[pos] = coo.col_indices[i];
        csr.values[pos] = coo.values[i];
        current_pos[row]++;
    }
    
    delete[] current_pos;
    
    // 对每行按列索引排序
    for (int i = 0; i < csr.rows; i++) {
        int start = csr.row_ptr[i];
        int end = csr.row_ptr[i + 1];
        
        for (int j = start; j < end - 1; j++) {
            for (int k = j + 1; k < end; k++) {
                if (csr.col_indices[j] > csr.col_indices[k]) {
                    int temp_col = csr.col_indices[j];
                    csr.col_indices[j] = csr.col_indices[k];
                    csr.col_indices[k] = temp_col;
                    
                    double temp_val = csr.values[j];
                    csr.values[j] = csr.values[k];
                    csr.values[k] = temp_val;
                }
            }
        }
    }
}

// ========== 动态稀疏行结构 ==========

struct DynamicSparseRow {
    int* cols;
    double* vals;
    int count;
    int capacity;
    
    DynamicSparseRow() : cols(nullptr), vals(nullptr), count(0), capacity(0) {}
    
    ~DynamicSparseRow() {
        delete[] cols;
        delete[] vals;
    }
    
    void reserve(int cap) {
        if (cap > capacity) {
            int* new_cols = new int[cap];
            double* new_vals = new double[cap];
            
            if (cols != nullptr) {
                for (int i = 0; i < count; i++) {
                    new_cols[i] = cols[i];
                    new_vals[i] = vals[i];
                }
                delete[] cols;
                delete[] vals;
            }
            
            cols = new_cols;
            vals = new_vals;
            capacity = cap;
        }
    }
    
    void append(int col, double val) {
        if (count >= capacity) {
            reserve(capacity == 0 ? 8 : capacity * 2);
        }
        cols[count] = col;
        vals[count] = val;
        count++;
    }
    
    void clear() {
        count = 0;
    }
    
    void sort() {
        for (int i = 0; i < count - 1; i++) {
            for (int j = i + 1; j < count; j++) {
                if (cols[i] > cols[j]) {
                    int temp_col = cols[i];
                    cols[i] = cols[j];
                    cols[j] = temp_col;
                    
                    double temp_val = vals[i];
                    vals[i] = vals[j];
                    vals[j] = temp_val;
                }
            }
        }
    }
    
    void swap(DynamicSparseRow& other) {
        int* temp_cols = cols;
        double* temp_vals = vals;
        int temp_count = count;
        int temp_capacity = capacity;
        
        cols = other.cols;
        vals = other.vals;
        count = other.count;
        capacity = other.capacity;
        
        other.cols = temp_cols;
        other.vals = temp_vals;
        other.count = temp_count;
        other.capacity = temp_capacity;
    }
    
    // 查找指定列的值，返回指针
    double getValue(int col) const {
        for (int i = 0; i < count; i++) {
            if (cols[i] == col) return vals[i];
            if (cols[i] > col) break;
        }
        return 0.0;
    }
    
    // 修改或添加元素
    void setValue(int col, double val) {
        for (int i = 0; i < count; i++) {
            if (cols[i] == col) {
                vals[i] = val;
                return;
            }
        }
        append(col, val);
        sort();
    }
};

// ========== 稀疏LU分解（带部分主元） ==========

struct SparseLU {
    CSRMatrix L;
    CSRMatrix U;
    int* perm;
    int* inv_perm;
    int n;
    
    SparseLU() : perm(nullptr), inv_perm(nullptr), n(0) {}
    
    ~SparseLU() {
        delete[] perm;
        delete[] inv_perm;
    }
};

bool sparseLU_with_pivoting(const CSRMatrix& A, SparseLU& lu) {
    lu.n = A.rows;
    
    // 初始化置换向量
    lu.perm = new int[lu.n];
    lu.inv_perm = new int[lu.n];
    for (int i = 0; i < lu.n; i++) {
        lu.perm[i] = i;
        lu.inv_perm[i] = i;
    }
    
    // 使用动态数组存���L和U的每一行
    DynamicSparseRow* L_rows = new DynamicSparseRow[lu.n];
    DynamicSparseRow* U_rows = new DynamicSparseRow[lu.n];
    
    // 工作矩阵：复制A并在原地修改
    DynamicSparseRow* work_rows = new DynamicSparseRow[lu.n];
    for (int i = 0; i < lu.n; i++) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            work_rows[i].append(A.col_indices[p], A.values[p]);
        }
    }
    
    std::cout << "Starting sparse LU factorization with partial pivoting...\n";
    
    int pivot_count = 0;
    
    // 高斯消元带主元选取
    for (int k = 0; k < lu.n; k++) {
        if (k % 100 == 0 && k > 0) {
            std::cout << "Processing pivot " << k << "/" << lu.n 
                      << " (pivots so far: " << pivot_count << ")" << std::endl;
        }
        
        // ========== 部分主元选取 ==========
        int pivot_row = k;
        double max_val = 0.0;
        
        for (int i = k; i < lu.n; i++) {
            double val = work_rows[i].getValue(k);
            double abs_val = std::abs(val);
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = i;
            }
        }
        
        if (max_val < 1e-14) {
            std::cerr << "Error: Cannot find valid pivot at column " << k 
                      << " (max value: " << max_val << ")" << std::endl;
            delete[] L_rows;
            delete[] U_rows;
            delete[] work_rows;
            return false;
        }
        
        // 如果需要，交换第k行和pivot_row
        if (pivot_row != k) {
            pivot_count++;
            
            // 交换work_rows
            work_rows[k].swap(work_rows[pivot_row]);
            
            // 交换L_rows（已计算的部分）
            L_rows[k].swap(L_rows[pivot_row]);
            
            // 更新置换向量
            int temp = lu.perm[k];
            lu.perm[k] = lu.perm[pivot_row];
            lu.perm[pivot_row] = temp;
            
            lu.inv_perm[lu.perm[k]] = k;
            lu.inv_perm[lu.perm[pivot_row]] = pivot_row;
        }
        
        // ========== 消元步骤 ==========
        double u_kk = work_rows[k].getValue(k);
        
        if (std::abs(u_kk) < 1e-14) {
            std::cerr << "Error: Zero pivot after selection at k=" << k 
                      << " (value: " << u_kk << ")" << std::endl;
            delete[] L_rows;
            delete[] U_rows;
            delete[] work_rows;
            return false;
        }
        
        // 将work_rows[k]存储到U[k,:]（只存储 >= k的列）
        for (int p = 0; p < work_rows[k].count; p++) {
            if (work_rows[k].cols[p] >= k) {
                U_rows[k].append(work_rows[k].cols[p], work_rows[k].vals[p]);
            }
        }
        
        // 对第k列下方的所有行进行消元
        for (int i = k + 1; i < lu.n; i++) {
            double a_ik = work_rows[i].getValue(k);
            
            if (std::abs(a_ik) < 1e-15) {
                continue;  // 该行在第k列已经是0
            }
            
            // 计算乘数
            double l_ik = a_ik / u_kk;
            L_rows[i].append(k, l_ik);
            
            // 执行行消元: work_rows[i] -= l_ik * work_rows[k]
            // 创建新行
            DynamicSparseRow new_row;
            
            int p1 = 0;  // work_rows[i]的指针
            int p2 = 0;  // work_rows[k]的指针
            
            while (p1 < work_rows[i].count || p2 < work_rows[k].count) {
                int col1 = (p1 < work_rows[i].count) ? work_rows[i].cols[p1] : lu.n + 1;
                int col2 = (p2 < work_rows[k].count) ? work_rows[k].cols[p2] : lu.n + 1;
                
                if (col1 == col2) {
                    double new_val = work_rows[i].vals[p1] - l_ik * work_rows[k].vals[p2];
                    if (std::abs(new_val) > 1e-15) {
                        new_row.append(col1, new_val);
                    }
                    p1++;
                    p2++;
                } else if (col1 < col2) {
                    new_row.append(col1, work_rows[i].vals[p1]);
                    p1++;
                } else {
                    double new_val = -l_ik * work_rows[k].vals[p2];
                    if (std::abs(new_val) > 1e-15) {
                        new_row.append(col2, new_val);
                    }
                    p2++;
                }
            }
            
            // 替换work_rows[i]
            work_rows[i].swap(new_row);
        }
    }
    
    std::cout << "Factorization complete (row pivots performed: " << pivot_count << ")\n";
    std::cout << "Converting to CSR format...\n";
    
    // 转换L为CSR格式（添加单位对角线）
    lu.L.rows = lu.n;
    lu.L.cols = lu.n;
    
    int L_total_nnz = lu.n;
    for (int i = 0; i < lu.n; i++) {
        L_total_nnz += L_rows[i].count;
    }
    
    lu.L.nnz = L_total_nnz;
    lu.L.row_ptr = new int[lu.n + 1];
    lu.L.col_indices = new int[L_total_nnz];
    lu.L.values = new double[L_total_nnz];
    
    lu.L.row_ptr[0] = 0;
    int pos = 0;
    for (int i = 0; i < lu.n; i++) {
        L_rows[i].sort();
        
        for (int j = 0; j < L_rows[i].count; j++) {
            lu.L.col_indices[pos] = L_rows[i].cols[j];
            lu.L.values[pos] = L_rows[i].vals[j];
            pos++;
        }
        lu.L.col_indices[pos] = i;
        lu.L.values[pos] = 1.0;
        pos++;
        
        lu.L.row_ptr[i + 1] = pos;
    }
    
    // 转换U为CSR格式
    lu.U.rows = lu.n;
    lu.U.cols = lu.n;
    
    int U_total_nnz = 0;
    for (int i = 0; i < lu.n; i++) {
        U_total_nnz += U_rows[i].count;
    }
    
    lu.U.nnz = U_total_nnz;
    lu.U.row_ptr = new int[lu.n + 1];
    lu.U.col_indices = new int[U_total_nnz];
    lu.U.values = new double[U_total_nnz];
    
    lu.U.row_ptr[0] = 0;
    pos = 0;
    for (int i = 0; i < lu.n; i++) {
        U_rows[i].sort();
        
        for (int j = 0; j < U_rows[i].count; j++) {
            lu.U.col_indices[pos] = U_rows[i].cols[j];
            lu.U.values[pos] = U_rows[i].vals[j];
            pos++;
        }
        lu.U.row_ptr[i + 1] = pos;
    }
    
    // 清理临时内存
    delete[] L_rows;
    delete[] U_rows;
    delete[] work_rows;
    
    return true;
}

// ========== 稀疏矩阵向量乘法 ==========

void sparseMatVec(const CSRMatrix& A, const double* x, double* y) {
    for (int i = 0; i < A.rows; i++) {
        y[i] = 0.0;
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            y[i] += A.values[p] * x[A.col_indices[p]];
        }
    }
}

// ========== 应用行置换: y = P*x ==========

void applyPermutation(const int* perm, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[perm[i]];
    }
}

// ========== 前向替换: L*y = b ==========

void forwardSubstitution(const CSRMatrix& L, const double* b, double* y) {
    for (int i = 0; i < L.rows; i++) {
        double sum = 0.0;
        
        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; p++) {
            int col = L.col_indices[p];
            if (col < i) {
                sum += L.values[p] * y[col];
            }
        }
        
        y[i] = b[i] - sum;
    }
}

// ========== 后向替换: U*x = y ==========

void backwardSubstitution(const CSRMatrix& U, const double* y, double* x) {
    for (int i = 0; i < U.rows; i++) {
        x[i] = 0.0;
    }
    
    for (int i = U.rows - 1; i >= 0; i--) {
        double sum = 0.0;
        double diag = 0.0;
        
        for (int p = U.row_ptr[i]; p < U.row_ptr[i + 1]; p++) {
            int col = U.col_indices[p];
            if (col == i) {
                diag = U.values[p];
            } else if (col > i) {
                sum += U.values[p] * x[col];
            }
        }
        
        if (std::abs(diag) < 1e-14) {
            std::cerr << "Error: Zero diagonal in U at row " << i << std::endl;
            x[i] = 0.0;
        } else {
            x[i] = (y[i] - sum) / diag;
        }
    }
}

// ========== 范数计算 ==========

double vectorNorm(const double* v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

double maxNorm(const double* v, int n) {
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        double abs_val = std::abs(v[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

// ========== 打印矩阵信息 ==========

void printCSRInfo(const CSRMatrix& M, const char* name) {
    std::cout << name << " matrix: " << M.rows << " x " << M.cols 
              << ", NNZ: " << M.nnz << std::endl;
    
    int diag_count = 0;
    double min_diag = 1e100;
    double max_diag = 0.0;
    
    for (int i = 0; i < M.rows; i++) {
        for (int p = M.row_ptr[i]; p < M.row_ptr[i + 1]; p++) {
            if (M.col_indices[p] == i) {
                diag_count++;
                double abs_val = std::abs(M.values[p]);
                if (abs_val < min_diag) min_diag = abs_val;
                if (abs_val > max_diag) max_diag = abs_val;
                break;
            }
        }
    }
    
    std::cout << "  Diagonal elements: " << diag_count << "/" << M.rows << std::endl;
    if (diag_count > 0) {
        std::cout << "  Min |diag|: " << min_diag << ", Max |diag|: " << max_diag << std::endl;
    }
}

// ========== 主程序 ==========

int main() {
    const char* filename = "sherman1.mtx";
    
    std::cout << "========== Sparse LU Solver with Partial Pivoting ==========\n\n";
    
    // 读取矩阵
    COOMatrix coo;
    if (!readMatrixMarket(filename, coo)) {
        return 1;
    }
    
    // 转换为CSR格式
    CSRMatrix A;
    COOtoCSR(coo, A);
    
    std::cout << "\n========== Original Matrix (CSR) ==========\n";
    printCSRInfo(A, "A");
    
    std::cout << "\nFirst 5 rows structure:\n";
    for (int i = 0; i < std::min(5, A.rows); i++) {
        std::cout << "Row " << i << " (nnz=" << (A.row_ptr[i+1] - A.row_ptr[i]) << "): ";
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            std::cout << "(" << A.col_indices[p] << "," << A.values[p] << ") ";
        }
        std::cout << std::endl;
    }
    
    // 创建右端向量b = A*x_true，其中x_true全为1
    double* x_true = new double[A.rows];
    double* b = new double[A.rows];
    
    for (int i = 0; i < A.rows; i++) {
        x_true[i] = 1.0;
    }
    
    sparseMatVec(A, x_true, b);
    
    std::cout << "\nRight-hand side b = A * x_true (first 10 elements):\n";
    for (int i = 0; i < std::min(10, A.rows); i++) {
        std::cout << "b[" << i << "] = " << b[i] << std::endl;
    }
    
    // 执行LU分解
    std::cout << "\n========== Performing Sparse LU Decomposition ==========\n";
    SparseLU lu;
    
    if (!sparseLU_with_pivoting(A, lu)) {
        std::cerr << "LU decomposition failed!" << std::endl;
        delete[] x_true;
        delete[] b;
        return 1;
    }
    
    std::cout << "\n========== LU Decomposition Results ==========\n";
    printCSRInfo(lu.L, "L");
    printCSRInfo(lu.U, "U");
    
    double fill_ratio = (double)(lu.L.nnz + lu.U.nnz - lu.n) / A.nnz;
    std::cout << "Fill ratio: " << fill_ratio << " (total nnz: " 
              << (lu.L.nnz + lu.U.nnz - lu.n) << " vs original: " << A.nnz << ")\n";
    
    std::cout << "\nPermutation vector (first 20): ";
    for (int i = 0; i < std::min(20, lu.n); i++) {
        std::cout << lu.perm[i] << " ";
    }
    std::cout << "\n";
    
    // 检查是否有非平凡的置换
    int perm_diff_count = 0;
    for (int i = 0; i < lu.n; i++) {
        if (lu.perm[i] != i) perm_diff_count++;
    }
    std::cout << "Non-trivial permutations: " << perm_diff_count << "/" << lu.n << "\n";
    
    std::cout << "\nFirst 5 rows of L:\n";
    for (int i = 0; i < std::min(5, lu.L.rows); i++) {
        std::cout << "L[" << i << "] (nnz=" << (lu.L.row_ptr[i+1] - lu.L.row_ptr[i]) << "): ";
        int printed = 0;
        for (int p = lu.L.row_ptr[i]; p < lu.L.row_ptr[i + 1] && printed < 10; p++, printed++) {
            std::cout << "(" << lu.L.col_indices[p] << "," << lu.L.values[p] << ") ";
        }
        if (lu.L.row_ptr[i + 1] - lu.L.row_ptr[i] > 10) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nFirst 5 rows of U:\n";
    for (int i = 0; i < std::min(5, lu.U.rows); i++) {
        std::cout << "U[" << i << "] (nnz=" << (lu.U.row_ptr[i+1] - lu.U.row_ptr[i]) << "): ";
        int printed = 0;
        for (int p = lu.U.row_ptr[i]; p < lu.U.row_ptr[i + 1] && printed < 10; p++, printed++) {
            std::cout << "(" << lu.U.col_indices[p] << "," << lu.U.values[p] << ") ";
        }
        if (lu.U.row_ptr[i + 1] - lu.U.row_ptr[i] > 10) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    
    // 求解 A*x = b
    // 由于 P*A = L*U，所以 A = P^T*L*U
    // A*x = b => P^T*L*U*x = b => L*U*x = P*b
    
    double* b_permuted = new double[A.rows];
    double* y = new double[A.rows];
    double* x_solved = new double[A.rows];
    
    for (int i = 0; i < A.rows; i++) {
        y[i] = 0.0;
        x_solved[i] = 0.0;
    }
    
    std::cout << "\n========== Solving Linear System ==========\n";
    std::cout << "System: A*x = b, with P*A = L*U\n";
    
    // 应用行置换到右端向量: b_permuted = P*b
    applyPermutation(lu.perm, b, b_permuted, A.rows);
    
    std::cout << "Applied row permutation to b\n";
    
    // 前向替换: L*y = P*b
    std::cout << "Forward substitution (L*y = P*b)...\n";
    forwardSubstitution(lu.L, b_permuted, y);
    
    // 后向替换: U*x = y
    std::cout << "Backward substitution (U*x = y)...\n";
    backwardSubstitution(lu.U, y, x_solved);
    
    std::cout << "\nSolution x (first 20 elements):\n";
    for (int i = 0; i < std::min(20, A.rows); i++) {
        std::cout << "x[" << i << "] = " << x_solved[i] 
                  << " (true: " << x_true[i] << ", error: " 
                  << std::abs(x_solved[i] - x_true[i]) << ")" << std::endl;
    }
    
    std::cout << "\nLast 10 elements:\n";
    for (int i = std::max(0, A.rows - 10); i < A.rows; i++) {
        std::cout << "x[" << i << "] = " << x_solved[i] 
                  << " (true: " << x_true[i] << ", error: " 
                  << std::abs(x_solved[i] - x_true[i]) << ")" << std::endl;
    }
    
    // 验证解的正确性: 计算 residual = A*x - b
    double* residual = new double[A.rows];
    sparseMatVec(A, x_solved, residual);
    
    for (int i = 0; i < A.rows; i++) {
        residual[i] -= b[i];
    }
    
    double residual_norm = vectorNorm(residual, A.rows);
    double residual_max = maxNorm(residual, A.rows);
    double b_norm = vectorNorm(b, A.rows);
    double relative_error = residual_norm / (b_norm + 1e-15);
    
    std::cout << "\n========== Verification ==========\n";
    std::cout << "||b||_2 = " << b_norm << std::endl;
    std::cout << "||A*x - b||_2 = " << residual_norm << std::endl;
    std::cout << "||A*x - b||_inf = " << residual_max << std::endl;
    std::cout << "Relative residual (||A*x - b||/||b||) = " << relative_error << std::endl;
    
    // 计算与真实解的误差
    double* error = new double[A.rows];
    for (int i = 0; i < A.rows; i++) {
        error[i] = x_solved[i] - x_true[i];
    }
    double error_norm = vectorNorm(error, A.rows);
    double error_max = maxNorm(error, A.rows);
    double x_norm = vectorNorm(x_true, A.rows);
    
    std::cout << "||x_solved - x_true||_2 = " << error_norm << std::endl;
    std::cout << "||x_solved - x_true||_inf = " << error_max << std::endl;
    std::cout << "Relative solution error = " << error_norm / x_norm << std::endl;
    
    if (relative_error < 1e-10) {
        std::cout << "\n✓✓✓ Solution is highly accurate!\n";
    } else if (relative_error < 1e-6) {
        std::cout << "\n✓✓ Solution is accurate!\n";
    } else if (relative_error < 1e-3) {
        std::cout << "\n✓ Solution has acceptable accuracy.\n";
    } else {
        std::cout << "\n✗ Warning: Solution may be inaccurate!\n";
    }
    
    // 清理内存
    delete[] x_true;
    delete[] b;
    delete[] b_permuted;
    delete[] y;
    delete[] x_solved;
    delete[] residual;
    delete[] error;
    
    std::cout << "\n========== Program Complete ==========\n";
    std::cout << "Memory cleaned up successfully.\n";
    
    return 0;
}