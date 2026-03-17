#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>

// ========== 稀疏矩阵数据结构 ==========

struct COOMatrix {
    int rows, cols, nnz;
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

struct CSRMatrix {
    int rows, cols, nnz;
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

// ========== MatrixMarket读取 ==========

bool readMatrixMarket(const char* filename, COOMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    sscanf(line.c_str(), "%d %d %d", &mat.rows, &mat.cols, &mat.nnz);
    
    std::cout << "Matrix: " << mat.rows << " x " << mat.cols 
              << ", NNZ: " << mat.nnz << std::endl;
    
    mat.row_indices = new int[mat.nnz];
    mat.col_indices = new int[mat.nnz];
    mat.values = new double[mat.nnz];
    
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

// ========== COO转CSR ==========

void COOtoCSR(const COOMatrix& coo, CSRMatrix& csr) {
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    
    csr.row_ptr = new int[csr.rows + 1];
    csr.col_indices = new int[csr.nnz];
    csr.values = new double[csr.nnz];
    
    for (int i = 0; i <= csr.rows; i++) csr.row_ptr[i] = 0;
    
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
    
    // 排序每行
    for (int i = 0; i < csr.rows; i++) {
        int start = csr.row_ptr[i];
        int end = csr.row_ptr[i + 1];
        
        for (int j = start; j < end - 1; j++) {
            for (int k = j + 1; k < end; k++) {
                if (csr.col_indices[j] > csr.col_indices[k]) {
                    int tc = csr.col_indices[j];
                    csr.col_indices[j] = csr.col_indices[k];
                    csr.col_indices[k] = tc;
                    
                    double tv = csr.values[j];
                    csr.values[j] = csr.values[k];
                    csr.values[k] = tv;
                }
            }
        }
    }
}




void reverseCuthillMcKee(const CSRMatrix& A, int* perm, int* invperm) {
    int n = A.rows;
    bool* visited = new bool[n];
    int* degree = new int[n];
    int* queue = new int[n];
    
    for (int i = 0; i < n; i++) {
        visited[i] = false;
        degree[i] = A.row_ptr[i + 1] - A.row_ptr[i];
    }
    
    std::cout << "Computing RCM ordering...\n";
    
    int start = 0;
    int min_degree = degree[0];
    for (int i = 1; i < n; i++) {
        if (degree[i] < min_degree) {
            min_degree = degree[i];
            start = i;
        }
    }
    
    int front = 0, back = 0;
    queue[back++] = start;
    visited[start] = true;
    
    while (front < back) {
        int node = queue[front++];
        
        int* neighbors = new int[n];
        int neighbor_count = 0;
        
        for (int p = A.row_ptr[node]; p < A.row_ptr[node + 1]; p++) {
            int neighbor = A.col_indices[p];
            if (neighbor != node && !visited[neighbor]) {
                bool already_added = false;
                for (int k = 0; k < neighbor_count; k++) {
                    if (neighbors[k] == neighbor) {
                        already_added = true;
                        break;
                    }
                }
                if (!already_added) {
                    neighbors[neighbor_count++] = neighbor;
                }
            }
        }
        
        for (int i = 0; i < neighbor_count - 1; i++) {
            for (int j = i + 1; j < neighbor_count; j++) {
                if (degree[neighbors[i]] > degree[neighbors[j]]) {
                    int temp = neighbors[i];
                    neighbors[i] = neighbors[j];
                    neighbors[j] = temp;
                }
            }
        }
        
        for (int i = 0; i < neighbor_count; i++) {
            if (!visited[neighbors[i]]) {
                visited[neighbors[i]] = true;
                queue[back++] = neighbors[i];
            }
        }
        
        delete[] neighbors;
    }
    
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            queue[back++] = i;
        }
    }
    
    for (int i = 0; i < n; i++) {
        perm[i] = queue[n - 1 - i];
        invperm[perm[i]] = i;
    }
    
    delete[] visited;
    delete[] degree;
    delete[] queue;
    
    std::cout << "RCM ordering complete.\n";
}

void naturalOrdering(int n, int* perm, int* invperm) {
    for (int i = 0; i < n; i++) {
        perm[i] = i;
        invperm[i] = i;
    }
}

// ========== 动态稀疏行 ==========

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
    
    void append(int col, flx::floatx<5, 10> val) {
        if (count >= capacity) {
            reserve(capacity == 0 ? 16 : capacity * 2);
        }
        cols[count] = col;
        vals[count] = val;
        count++;
    }
    
    void sort() {
        for (int i = 0; i < count - 1; i++) {
            for (int j = i + 1; j < count; j++) {
                if (cols[i] > cols[j]) {
                    int tc = cols[i]; cols[i] = cols[j]; cols[j] = tc;
                    flx::floatx<4, 3> tv = vals[i]; vals[i] = vals[j]; vals[j] = tv;
                }
            }
        }
    }
    
    double* find(int col) {
        for (int i = 0; i < count; i++) {
            if (cols[i] == col) return &vals[i];
            if (cols[i] > col) break;
        }
        return nullptr;
    }
};

// ========== 稀疏累积器 ==========

struct SparseAccumulator {
    double* values;
    bool* flags;
    int* pattern;
    int count;
    int size;
    
    SparseAccumulator(int n) : count(0), size(n) {
        values = new double[n];
        flags = new bool[n];
        pattern = new int[n];
        for (int i = 0; i < n; i++) {
            values[i] = 0.0;
            flags[i] = false;
        }
    }
    
    ~SparseAccumulator() {
        delete[] values;
        delete[] flags;
        delete[] pattern;
    }
    
    void scatter(int col, flx::floatx<4, 3> val) {
        if (!flags[col]) {
            flags[col] = true;
            pattern[count++] = col;
        }
        values[col] += val;
    }
    
    void clear() {
        for (int i = 0; i < count; i++) {
            values[pattern[i]] = 0.0;
            flags[pattern[i]] = false;
        }
        count = 0;
    }
};

// ========== 稀疏LU分解 ==========

struct SparseLU {
    CSRMatrix L;
    CSRMatrix U;
    int* P;
    int* invP;
    int n;
    
    SparseLU() : P(nullptr), invP(nullptr), n(0) {}
    
    ~SparseLU() {
        delete[] P;
        delete[] invP;
    }
};

bool sparseLUwithPivoting(const CSRMatrix& A, SparseLU& lu) {
    lu.n = A.rows;
    lu.P = new int[lu.n];
    lu.invP = new int[lu.n];
    
    for (int i = 0; i < lu.n; i++) {
        lu.P[i] = i;
        lu.invP[i] = i;
    }
    
    DynamicSparseRow* work_rows = new DynamicSparseRow[lu.n];
    
    for (int i = 0; i < A.rows; i++) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            work_rows[i].append(A.col_indices[p], A.values[p]);
        }
    }
    
    DynamicSparseRow* L_rows = new DynamicSparseRow[lu.n];
    DynamicSparseRow* U_rows = new DynamicSparseRow[lu.n];
    
    SparseAccumulator acc(lu.n);
    
    std::cout << "LU factorization with partial pivoting...\n";
    
    int total_swaps = 0;
    
    for (int k = 0; k < lu.n; k++) {
        if (k % 100 == 0) {
            std::cout << "  Step " << k << "/" << lu.n << std::endl;
        }
        
        // ========== Partial Pivoting ==========
        int pivot_row = k;
        flx::floatx<4, 3> max_pivot = 0.0;
        
        for (int i = k; i < lu.n; i++) {
            double* val_ptr = work_rows[i].find(k);
            if (val_ptr != nullptr && abs(*val_ptr) > max_pivot) {
                max_pivot = abs(*val_ptr);
                pivot_row = i;
            }
        }
        
        if (max_pivot < 1e-14) {
            std::cerr << "Error: Singular matrix at column " << k << std::endl;
            delete[] work_rows;
            delete[] L_rows;
            delete[] U_rows;
            return false;
        }
        
        // 行交换（只交换work_rows，不交换L_rows！）
        if (pivot_row != k) {
            total_swaps++;
            
            // 交换work_rows
            DynamicSparseRow temp_work = work_rows[k];
            work_rows[k] = work_rows[pivot_row];
            work_rows[pivot_row] = temp_work;
            
            // 交换已有的L元素
            DynamicSparseRow temp_L = L_rows[k];
            L_rows[k] = L_rows[pivot_row];
            L_rows[pivot_row] = temp_L;
            
            // 更新置换
            int temp_p = lu.P[k];
            lu.P[k] = lu.P[pivot_row];
            lu.P[pivot_row] = temp_p;
            
            lu.invP[lu.P[k]] = k;
            lu.invP[lu.P[pivot_row]] = pivot_row;
        }
        
        double* pivot_ptr = work_rows[k].find(k);
        if (pivot_ptr == nullptr) {
            std::cerr << "Error: Pivot not found at (" << k << "," << k << ")" << std::endl;
            delete[] work_rows;
            delete[] L_rows;
            delete[] U_rows;
            return false;
        }
        flx::floatx<4, 3> pivot = *pivot_ptr;
        
        // 保存U[k,:]（所有 >= k的列）
        for (int p = 0; p < work_rows[k].count; p++) {
            if (work_rows[k].cols[p] >= k) {
                U_rows[k].append(work_rows[k].cols[p], work_rows[k].vals[p]);
            }
        }
        
        // 对k行下方的所有行进行消元
        for (int i = k + 1; i < lu.n; i++) {
            double* aik_ptr = work_rows[i].find(k);
            if (aik_ptr == nullptr || abs(*aik_ptr) < 1e-15) {
                continue;
            }
            
            flx::floatx<4, 3> multiplier = (*aik_ptr) / pivot;
            L_rows[i].append(k, multiplier);
            
            // row_i -= multiplier * row_k
            acc.clear();
            
            for (int p = 0; p < work_rows[i].count; p++) {
                acc.scatter(work_rows[i].cols[p], work_rows[i].vals[p]);
            }
            
            for (int p = 0; p < work_rows[k].count; p++) {
                acc.scatter(work_rows[k].cols[p], -multiplier * work_rows[k].vals[p]);
            }
            
            work_rows[i].count = 0;
            for (int p = 0; p < acc.count; p++) {
                int col = acc.pattern[p];
                flx::floatx<4, 3> val = acc.values[col];
                if (abs(val) > 1e-15) {
                    work_rows[i].append(col, val);
                }
            }
            work_rows[i].sort();
        }
    }
    
    std::cout << "Total row swaps: " << total_swaps << std::endl;
    
    // 转换L为CSR
    lu.L.rows = lu.n;
    lu.L.cols = lu.n;
    
    int L_nnz = lu.n;
    for (int i = 0; i < lu.n; i++) {
        L_nnz += L_rows[i].count;
    }
    
    lu.L.nnz = L_nnz;
    lu.L.row_ptr = new int[lu.n + 1];
    lu.L.col_indices = new int[L_nnz];
    lu.L.values = new double[L_nnz];
    
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
    
    // 转换U为CSR
    lu.U.rows = lu.n;
    lu.U.cols = lu.n;
    
    int U_nnz = 0;
    for (int i = 0; i < lu.n; i++) {
        U_nnz += U_rows[i].count;
    }
    
    lu.U.nnz = U_nnz;
    lu.U.row_ptr = new int[lu.n + 1];
    lu.U.col_indices = new int[U_nnz];
    lu.U.values = new double[U_nnz];
    
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
    
    
    return true;
}

// ========== 应用置换到矩阵 ==========

void permuteCSR(const CSRMatrix& A, const int* rperm, const int* cperm, CSRMatrix& P) {
    P.rows = A.rows;
    P.cols = A.cols;
    P.nnz = A.nnz;
    
    P.row_ptr = new int[P.rows + 1];
    P.col_indices = new int[P.nnz];
    P.values = new double[P.nnz];
    
    int* row_counts = new int[P.rows]();
    
    for (int old_row = 0; old_row < A.rows; old_row++) {
        int new_row = rperm[old_row];
        row_counts[new_row] = A.row_ptr[old_row + 1] - A.row_ptr[old_row];
    }
    
    P.row_ptr[0] = 0;
    for (int i = 0; i < P.rows; i++) {
        P.row_ptr[i + 1] = P.row_ptr[i] + row_counts[i];
    }
    
    int* current_pos = new int[P.rows];
    for (int i = 0; i < P.rows; i++) {
        current_pos[i] = P.row_ptr[i];
    }
    
    for (int old_row = 0; old_row < A.rows; old_row++) {
        int new_row = rperm[old_row];
        
        for (int p = A.row_ptr[old_row]; p < A.row_ptr[old_row + 1]; p++) {
            int old_col = A.col_indices[p];
            int new_col = cperm[old_col];
            
            int pos = current_pos[new_row];
            P.col_indices[pos] = new_col;
            P.values[pos] = A.values[p];
            current_pos[new_row]++;
        }
    }
    
    delete[] row_counts;
    delete[] current_pos;
    
    for (int i = 0; i < P.rows; i++) {
        int start = P.row_ptr[i];
        int end = P.row_ptr[i + 1];
        
        for (int j = start; j < end - 1; j++) {
            for (int k = j + 1; k < end; k++) {
                if (P.col_indices[j] > P.col_indices[k]) {
                    int tc = P.col_indices[j];
                    P.col_indices[j] = P.col_indices[k];
                    P.col_indices[k] = tc;
                    
                    flx::floatx<5, 10> tv = P.values[j];
                    P.values[j] = P.values[k];
                    P.values[k] = tv;
                }
            }
        }
    }
}

void sparseMatVec(const CSRMatrix& A, const double* x, double* y) {
    for (int i = 0; i < A.rows; i++) {
        y[i] = 0.0;
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            y[i] += A.values[p] * x[A.col_indices[p]];
        }
    }
}

void forwardSubstitution(const CSRMatrix& L, const double* b, double* y) {
    for (int i = 0; i < L.rows; i++) {
        flx::floatx<4, 3> sum = 0.0;
        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; p++) {
            if (L.col_indices[p] < i) {
                sum += L.values[p] * y[L.col_indices[p]];
            }
        }
        y[i] = b[i] - sum;
    }
}

void backwardSubstitution(const CSRMatrix& U, const double* y, double* x) {
    for (int i = 0; i < U.rows; i++) x[i] = 0.0;
    
    for (int i = U.rows - 1; i >= 0; i--) {
        flx::floatx<5, 10> sum = 0.0;
        flx::floatx<5, 10> diag = 0.0;
        
        for (int p = U.row_ptr[i]; p < U.row_ptr[i + 1]; p++) {
            int col = U.col_indices[p];
            if (col == i) {
                diag = U.values[p];
            } else if (col > i) {
                sum += U.values[p] * x[col];
            }
        }
        
        if (abs(diag) < 1e-14) {
            x[i] = 0.0;
        } else {
            x[i] = (y[i] - sum) / diag;
        }
    }
}

// 正确的求解函数：P*Q*A*Q^T = L*U，求解 A*x=b
void solveLUSystem(const SparseLU& lu, const int* Q, const int* invQ, 
                   const double* b_orig, double* x_orig) {
    // 我们有: P * (Q*A*Q^T) * (Q*x) = P * (Q*b)
    // 即: L * U * z = c，其中 z = Q*x, c = P*Q*b
    
    double* c = new double[lu.n];      // c = P * Q * b
    double* y = new double[lu.n];      // L * y = c
    double* z = new double[lu.n];      // U * z = y
    double* x_perm = new double[lu.n]; // x_perm = Q * x_orig
    
    // Step 1: 应用列排序 Q 到 b: b_temp = Q * b_orig
    for (int i = 0; i < lu.n; i++) {
        c[i] = b_orig[Q[i]];
    }
    
    // Step 2: 应用行置换 P: c = P * b_temp
    double* temp = new double[lu.n];
    for (int i = 0; i < lu.n; i++) {
        temp[i] = c[i];
    }
    for (int i = 0; i < lu.n; i++) {
        c[i] = temp[lu.P[i]];
    }
    delete[] temp;
    
    // Step 3: 前向替换 L * y = c
    forwardSubstitution(lu.L, c, y);
    
    // Step 4: 后向替换 U * z = y
    backwardSubstitution(lu.U, y, z);
    
    // Step 5: 应用 Q^T: x_orig = Q^T * z
    for (int i = 0; i < lu.n; i++) {
        x_orig[Q[i]] = z[i];
    }
    
    delete[] c;
    delete[] y;
    delete[] z;
    delete[] x_perm;
}


flx::floatx<4, 3> norm2(const double* v, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrt(sum);
}

flx::floatx<4, 3> normInf(const flx::floatx<4, 3>* v, int n) {
    flx::floatx<4, 3> max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (abs(v[i]) > max_val) max_val = abs(v[i]);
    }
    return max_val;
}
void printInfo(const CSRMatrix& M, const char* name) {
    int diag_count = 0;
    double min_diag = 1e100, max_diag = 0.0;
    
    for (int i = 0; i < M.rows; i++) {
        for (int p = M.row_ptr[i]; p < M.row_ptr[i + 1]; p++) {
            if (M.col_indices[p] == i) {
                diag_count++;
                double av = abs(M.values[p]);
                if (av < min_diag) min_diag = av;
                if (av > max_diag) max_diag = av;
                break;
            }
        }
    }
    
    std::cout << name << ": " << M.rows << "x" << M.cols 
              << ", NNZ=" << M.nnz;
    if (diag_count > 0) {
        std::cout << ", |diag|∈[" << min_diag << "," << max_diag << "]";
    }
    std::cout << std::endl;
}


int main() {
    
    COOMatrix coo;
    if (!readMatrixMarket("sherman1.mtx", coo)) return 1;
    
    CSRMatrix A;
    COOtoCSR(coo, A);
    
    std::cout << "\n"; 
    printInfo(A, "Original A");
    
    // 准备测试：x_true全为1，b = A*x_true
    double* x_true = new double[A.rows];
    double* b = new double[A.rows];
    
    for (int i = 0; i < A.rows; i++) x_true[i] = 1.0;
    sparseMatVec(A, x_true, b);
    
    double b_norm = norm2(b, A.rows);
    double x_norm = norm2(x_true, A.rows);
    
    std::cout << "Test: x_true = [1,1,...,1], ||b||₂ = " << b_norm << "\n";
    
    // ========== 测试1: 仅Partial Pivoting ==========
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 1: Partial Pivoting Only\n";
    std::cout << std::string(60, '=') << "\n";
    
    SparseLU lu1;
    if (!sparseLUwithPivoting(A, lu1)) return 1;
    
    printInfo(lu1.L, "L");
    printInfo(lu1.U, "U");
    
    double fill1 = (double)(lu1.L.nnz + lu1.U.nnz - lu1.n) / A.nnz;
    std::cout << "Fill ratio: " << fill1 << " (" << (lu1.L.nnz + lu1.U.nnz - lu1.n) << " nnz)\n";
    
    int* nat_perm = new int[A.rows];
    int* nat_invperm = new int[A.rows];
    naturalOrdering(A.rows, nat_perm, nat_invperm);
    
    double* x1 = new double[A.rows];
    solveLUSystem(lu1, nat_perm, nat_invperm, b, x1);
    
    double* res1 = new double[A.rows];
    sparseMatVec(A, x1, res1);
    for (int i = 0; i < A.rows; i++) res1[i] -= b[i];
    
    double rel_res1 = norm2(res1, A.rows) / b_norm;
    for (int i = 0; i < A.rows; i++) res1[i] = x1[i] - x_true[i];
    double rel_err1 = norm2(res1, A.rows) / x_norm;
    
    std::cout << "Relative residual: " << rel_res1 << "\n";
    std::cout << "Relative error:    " << rel_err1 << "\n";
    
    std::cout << "\nSolution (first 10):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  x[" << i << "] = " << x1[i] 
                  << " (err: " << abs(x1[i]-1.0) << ")\n";
    }
    
    // ========== 测试2: RCM + Partial Pivoting ==========
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST 2: RCM Ordering + Partial Pivoting\n";
    std::cout << std::string(60, '=') << "\n";
    
    int* rcm_perm = new int[A.rows];
    int* rcm_invperm = new int[A.rows];
    reverseCuthillMcKee(A, rcm_perm, rcm_invperm);
    
    std::cout << "RCM permutation (first 20): ";
    for (int i = 0; i < 20; i++) {
        std::cout << rcm_perm[i] << " ";
    }
    std::cout << "\n";
    
    CSRMatrix A_rcm;
    permuteCSR(A, rcm_perm, rcm_perm, A_rcm);
    
    printInfo(A_rcm, "Q*A*Q^T (RCM)");
    
    SparseLU lu2;
    if (!sparseLUwithPivoting(A_rcm, lu2)) return 1;
    
    printInfo(lu2.L, "L");
    printInfo(lu2.U, "U");
    
    double fill2 = (double)(lu2.L.nnz + lu2.U.nnz - lu2.n) / A.nnz;
    std::cout << "Fill ratio: " << fill2 << " (" << (lu2.L.nnz + lu2.U.nnz - lu2.n) << " nnz)\n";
    
    if (fill1 > 0) {
        std::cout << "Fill reduction: " << (1.0 - fill2/fill1) * 100 << "%\n";
    }
    
    double* x2 = new double[A.rows];
    solveLUSystem(lu2, rcm_perm, rcm_invperm, b, x2);
    
    double* res2 = new double[A.rows];
    sparseMatVec(A, x2, res2);
    for (int i = 0; i < A.rows; i++) res2[i] -= b[i];
    
    double rel_res2 = norm2(res2, A.rows) / b_norm;
    for (int i = 0; i < A.rows; i++) res2[i] = x2[i] - x_true[i];
    double rel_err2 = norm2(res2, A.rows) / x_norm;
    
    std::cout << "Relative residual: " << rel_res2 << "\n";
    std::cout << "Relative error:    " << rel_err2 << "\n";
    
    std::cout << "\nSolution (first 10):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  x[" << i << "] = " << x2[i] 
                  << " (err: " << abs(x2[i]-1.0) << ")\n";
    }
    
    // ========== 总结 ==========
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    printf("%-20s | %10s | %8s | %8s | %12s | %10s\n", 
           "Method", "Fill", "L-NNZ", "U-NNZ", "Residual", "Error");
    printf("%s\n", std::string(85, '-').c_str());
    printf("%-20s | %10.4f | %8d | %8d | %12.2e | %10.2e\n", 
           "Natural+Pivot", fill1, lu1.L.nnz, lu1.U.nnz, rel_res1, rel_err1);
    printf("%-20s | %10.4f | %8d | %8d | %12.2e | %10.2e\n", 
           "RCM+Pivot", fill2, lu2.L.nnz, lu2.U.nnz, rel_res2, rel_err2);
    
    PROMISE_CHECK_ARRAY(x2, A.rows);
    std::cout << "\n";
    
    bool accurate = (rel_res1 < 1e-6 && rel_res2 < 1e-6);
    
    if (accurate) {
        std::cout << "✓✓✓ Both solutions are ACCURATE!\n";
    } else if (rel_res1 < 1e-3 && rel_res2 < 1e-3) {
        std::cout << "✓✓ Solutions have acceptable accuracy.\n";
    } else {
        std::cout << "✗ Warning: Check numerical stability!\n";
    }
    
    std::cout << "\nRecommendation: Use " 
              << (fill2 < fill1 ? "RCM+Pivot" : "Natural+Pivot")
              << " (lower fill-in)\n";
    
    // 清理
    delete[] x_true;
    delete[] b;
    delete[] nat_perm;
    delete[] nat_invperm;
    delete[] x1;
    delete[] res1;
    delete[] rcm_perm;
    delete[] rcm_invperm;
    delete[] x2;
    delete[] res2;
    
    std::cout << "\n✓ Memory freed successfully.\n";
    
    return 0;
}