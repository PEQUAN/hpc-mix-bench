#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // Store number of non-zeros
};

bool compare_by_column(const std::pair<int, half_float::half>& a, const std::pair<int, half_float::half>& b) {
    return a.first < b.first;
}

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return A;
    }

    std::string line;
    while (getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    // Temporary storage for entries
    struct Entry { int row, col; float val; };
    Entry* entries = new Entry[2 * nz]; // Max possible entries (symmetric)
    int entry_count = 0;

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            delete[] entries;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        float val;
        ss >> i >> j >> val;
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }

    // Count non-zeros per row
    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    // Allocate CSR arrays
    A.nnz = entry_count;
    A.values = new double[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    // Sort entries by row and column
    std::sort(entries, entries + entry_count, 
        [](const Entry& a, const Entry& b) { 
            return a.row == b.row ? a.col < b.col : a.row < b.row; 
        });

    // Fill CSR arrays
    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

    // Clean up
    delete[] nnz_per_row;
    delete[] entries;
    return A;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
}

double* matvec(const CSRMatrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

float dot(const double* a, const double* b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double* axpy(float alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

float norm(const double* v, int n) {
    float d = dot(v, v, n);
    return sqrt(d);
}

struct Solution {
    double* x;
    float residual;
    int iterations;
};

Solution bicgstab(const CSRMatrix& A, const double* b, int max_iter = 1000, float tol = 1e-5) {
    int n = A.n;
    double* x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    double* r_hat = new double[n];
    for (int i = 0; i < n; ++i) r_hat[i] = r[i];
    double* p = new double[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    double* v = new double[n]();
    float rho = 1.0, alpha = 1.0, omega = 1.0;
    float initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        Solution result = {x, -1.0, 0};
        delete[] r; delete[] r_hat; delete[] p; delete[] v;
        return result;
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    float tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        float rho_new = dot(r_hat, r, n);
        if (abs(rho_new) < 1e-10) {
            std::cerr << "Breakdown: rho = " << rho_new << " at iteration " << k << std::endl;
            break;
        }
        float beta = (rho_new / rho) * (alpha / omega);
        double* temp1 = axpy(-omega, v, p, n);
        double* temp2 = axpy(beta, temp1, r, n);
        delete[] temp1;
        for (int i = 0; i < n; ++i) p[i] = temp2[i];
        delete[] temp2;
        delete[] v;
        v = matvec(A, p);
        float rhat_v = dot(r_hat, v, n);
        if (abs(rhat_v) < 1e-10) {
            std::cerr << "Breakdown: r_hat^T v = " << rhat_v << " at iteration " << k << std::endl;
            break;
        }
        alpha = rho_new / rhat_v;
        double* s = axpy(-alpha, v, r, n);
        float s_norm = norm(s, n);
        if (s_norm < tol_abs) {
            double* temp = axpy(alpha, p, x, n);
            delete[] x; x = temp;
            delete[] r; r = s;
            break;
        }
        double* t = matvec(A, s);
        float t_t = dot(t, t, n);
        if (abs(t_t) < 1e-10) {
            std::cerr << "Breakdown: t^T t = " << t_t << " at iteration " << k << std::endl;
            delete[] s; delete[] t;
            break;
        }
        omega = dot(t, s, n) / t_t;
        double* temp3 = axpy(omega, s, x, n);
        double* temp4 = axpy(alpha, p, temp3, n);
        delete[] temp3; delete[] x; x = temp4;
        double* temp5 = axpy(-omega, t, s, n);
        delete[] r; r = temp5;
        float r_norm = norm(r, n);
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k + 1 << std::endl;
            break;
        }
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Residual = " << r_norm << std::endl;
        }
        if (r_norm < tol_abs) break;
        rho = rho_new;
        delete[] s; delete[] t;
    }

    float residual = norm(r, n);
    Solution result = {x, residual, k + 1};
    delete[] r; delete[] r_hat; delete[] p; delete[] v;
    return result;
}

double* generate_rhs(int n) {
    double* b = new double[n];
    std::mt19937 gen(2025);
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    double* b = generate_rhs(A.n);

    Solution result = bicgstab(A, b, 2 * A.n);
    
    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    // fail for checking PROMISE_CHECK_ARRAY(result.x, A.n);

    double check_result[A.n];
    for (int i=0; i<A.n; i++){
        check_result[i] = result.x[i];
    }

    PROMISE_CHECK_ARRAY(check_result, A.n);

    double* Ax = matvec(A, result.x);

    double* temp = axpy(-1.0, Ax, b, A.n);
    float verify_residual = norm(temp, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;
    delete[] Ax; delete[] temp;


    delete[] b;
    delete[] result.x;
    free_csr_matrix(A);
    return 0;
}