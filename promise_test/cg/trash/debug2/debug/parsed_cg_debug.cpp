#ifndef _Alignof
#define _Alignof(type) alignof(type)
#endif

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>


struct CSRMatrix {
    int n;    
    int nnz;  
    __PROMISE0__* values; 
    int* col_indices;
    int* row_ptr;   

    CSRMatrix() : n(0), nnz(0), values(nullptr), col_indices(nullptr), row_ptr(nullptr) {}

    ~CSRMatrix() {
        delete[] values;
        delete[] col_indices;
        delete[] row_ptr;
    }
};


struct CGResult {
    __PROMISE1__* x;
    int n;
    __PROMISE2__ residual;
    int iterations;

    ~CGResult() {
        delete[] x;
    }
};

// New function to generate random SPD matrix in CSR format
CSRMatrix generate_random_spd_matrix(int n, double density, unsigned int seed = 42) {
    CSRMatrix A;
    A.n = n;
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> val_dis(0.1, 1.0);
    std::bernoulli_distribution edge_dis(density);

    // First pass: count non-zeros
    A.nnz = 0;
    std::vector<std::vector<std::pair<int, double>>> temp(n);
    
    for (int i = 0; i < n; ++i) {
        // Ensure diagonal dominance
        temp[i].push_back({i, 0.0}); // Placeholder for diagonal
        for (int j = i + 1; j < n; ++j) {
            if (edge_dis(gen)) {
                double val = val_dis(gen);
                temp[i].push_back({j, val});
                temp[j].push_back({i, val}); // Symmetry
            }
        }
    }

    // Calculate nnz and set diagonal for SPD
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        __PROMISE3__ row_sum = 0.0;
        for (const auto& entry : temp[i]) {
            if (entry.first != i) {
                row_sum += abs(entry.second);
            }
        }
        // Set diagonal to ensure positive definiteness
        for (auto& entry : temp[i]) {
            if (entry.first == i) {


                entry.second = row_sum + 1.0; // Diagonal dominance
            }
        }
        A.nnz += temp[i].size();
        A.row_ptr[i + 1] = A.nnz;
    }

    // Allocate and fill CSR arrays
    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        // Sort by column index
        std::sort(temp[i].begin(), temp[i].end(), 
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& entry : temp[i]) {
            A.col_indices[pos] = entry.first;
            A.values[pos] = entry.second;
            pos++;
        }
    }

    std::cout << "Generated matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

// Rest of the code remains the same until main()
double* matvec(const CSRMatrix& A, const double* x, int n) {
    double* y = new double[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

double* axpy(double alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(const double* v, int n) {
    return sqrt(dot(v, v, n));
}


CGResult conjugate_gradient(const CSRMatrix& A, const __PROMISE4__* b, int max_iter = 1000, __PROMISE5__ tol = 1e-6) {
    int n = A.n;
    CGResult result;
    result.n = n;
    result.x = new __PROMISE6__[n]();
    __PR_3__* r = new __PR_3__[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    __PR_2__* p = new __PR_2__[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    __PROMISE7__ rtr = dot(r, r, n);
    __PROMISE8__ tol2 = tol * tol * dot(b, b, n);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        __PROMISE9__* Ap = matvec(A, p, n);
        __PROMISE10__ alpha = rtr / dot(p, Ap, n);
        __PROMISE11__* x_new = axpy(alpha, p, result.x, n);
        delete[] result.x;
        result.x = x_new;
        __PROMISE12__* r_new = axpy(-alpha, Ap, r, n);
        delete[] r;
        r = r_new;
        __PROMISE13__ rtr_new = dot(r, r, n);
        __PROMISE14__ beta = rtr_new / rtr;
        __PROMISE15__* p_new = axpy(beta, p, r, n);
        delete[] p;
        p = p_new;
        delete[] Ap;
        rtr = rtr_new;
    }

    result.residual = norm(r, n);
    result.iterations = k;

    delete[] r;
    delete[] p;

    return result;
}




__PROMISE16__* generate_rhs(int n) {
    __PR_1__* b = new __PR_1__[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const __PROMISE17__* x, int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
}

int main() {
    int n = 50;           // Matrix size
    double density = 0.011; // Controls sparsity (0 to 1)
    
    // Generate random SPD matrix
    CSRMatrix A = generate_random_spd_matrix(n, density);
    if (A.n == 0) return 1;

    double* b = generate_rhs(A.n);
    CGResult result = conjugate_gradient(A, b, A.n);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = matvec(A, result.x, A.n);
    double* verify_vec = axpy(-1.0, Ax, b, A.n);
    double verify_residual = norm(verify_vec, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;

    // write_solution(result.x, A.n, "results/cg/cg_solution.csv");
    PROMISE_CHECK_VAR(result.residual);

    delete[] b;
    delete[] Ax;
    delete[] verify_vec;

    return 0;
}