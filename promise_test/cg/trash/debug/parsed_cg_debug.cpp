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

// New function to generate random SPD matrix in CSR format
CSRMatrix generate_random_spd_matrix(int n, __PROMISE1__ density, unsigned int seed = 42) {
    CSRMatrix A;
    A.n = n;
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> val_dis(0.1, 1.0);
    std::bernoulli_distribution edge_dis(density);

    // First pass: count non-zeros
    A.nnz = 0;
    std::vector<std::vector<std::pair<int, __PROMISE2__>>> temp(n);
    
    for (int i = 0; i < n; ++i) {
        // Ensure diagonal dominance
        temp[i].push_back({i, 0.0}); // Placeholder for diagonal
        for (int j = i + 1; j < n; ++j) {
            if (edge_dis(gen)) {
                __PROMISE3__ val = val_dis(gen);
                temp[i].push_back({j, val});
                temp[j].push_back({i, val}); // Symmetry
            }
        }
    }

    // Calculate nnz and set diagonal for SPD
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        __PROMISE4__ row_sum = 0.0;
        for (const auto& entry : temp[i]) {
            if (entry.first != i) {
                row_sum += std::abs(entry.second);
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
    A.values = new __PROMISE5__[A.nnz];
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
__PROMISE6__* matvec(const CSRMatrix& A, const __PROMISE7__* x, int n) {
    __PR_4__* y = new __PR_4__[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

__PROMISE8__* axpy(__PROMISE9__ alpha, const __PROMISE10__* x, const __PROMISE11__* y, int n) {
    __PR_5__* result = new __PR_5__[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

__PROMISE12__ dot(const __PROMISE13__* a, const __PROMISE14__* b, int n) {
    __PROMISE15__ sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__PROMISE16__ norm(const __PROMISE17__* v, int n) {
    return std::sqrt(dot(v, v, n));
}

struct CGResult {
    __PROMISE18__* x;
    int n;
    __PROMISE19__ residual;
    int iterations;

    ~CGResult() {
        delete[] x;
    }
};

CGResult conjugate_gradient(const CSRMatrix& A, const __PROMISE20__* b, int max_iter = 1000, __PROMISE21__ tol = 1e-6) {
    int n = A.n;
    CGResult result;
    result.n = n;
    result.x = new __PROMISE22__[n]();
    __PR_3__* r = new __PR_3__[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    __PR_2__* p = new __PR_2__[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    __PROMISE23__ rtr = dot(r, r, n);
    __PROMISE24__ tol2 = tol * tol * dot(b, b, n);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        __PROMISE25__* Ap = matvec(A, p, n);
        __PROMISE26__ alpha = rtr / dot(p, Ap, n);
        __PROMISE27__* x_new = axpy(alpha, p, result.x, n);
        delete[] result.x;
        result.x = x_new;
        __PROMISE28__* r_new = axpy(-alpha, Ap, r, n);
        delete[] r;
        r = r_new;
        __PROMISE29__ rtr_new = dot(r, r, n);
        __PROMISE30__ beta = rtr_new / rtr;
        __PROMISE31__* p_new = axpy(beta, p, r, n);
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

__PROMISE32__* generate_rhs(int n) {
    __PR_1__* b = new __PR_1__[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const __PROMISE33__* x, int n, const std::string& filename) {
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
    int n = 5000;           // Matrix size
    __PROMISE34__ density = 0.001; // Controls sparsity (0 to 1)
    
    // Generate random SPD matrix
    CSRMatrix A = generate_random_spd_matrix(n, density);
    if (A.n == 0) return 1;

    __PROMISE35__* b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    CGResult result = conjugate_gradient(A, b, A.n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    __PROMISE36__* Ax = matvec(A, result.x, A.n);
    __PROMISE37__* verify_vec = axpy(-1.0, Ax, b, A.n);
    __PROMISE38__ verify_residual = norm(verify_vec, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;

    // write_solution(result.x, A.n, "results/cg/cg_solution.csv");
    PROMISE_CHECK_VAR(result.residual);

    delete[] b;
    delete[] Ax;
    delete[] verify_vec;

    return 0;
}