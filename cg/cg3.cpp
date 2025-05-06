
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>

struct CSRMatrix {
    int n;    
    int nnz;  
    double* values; 
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
CSRMatrix generate_random_spd_matrix(int n, double density, unsigned int seed = 42) {
    CSRMatrix A;
    A.n = n;
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> val_dis(0.1, 1.0);
    std::bernoulli_distribution edge_dis(density);

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
        double row_sum = 0.0;
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
    return std::sqrt(dot(v, v, n));
}

struct CGResult {
    double* x;
    int n;
    double residual;
    int iterations;

    ~CGResult() {
        delete[] x;
    }
};

CGResult conjugate_gradient(const CSRMatrix& A, const double* b, int max_iter = 1000, double tol = 1e-6) {
    int n = A.n;
    CGResult result;
    result.n = n;
    result.x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    double* p = new double[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    double rtr = dot(r, r, n);
    double tol2 = tol * tol * dot(b, b, n);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        double* Ap = matvec(A, p, n);
        double alpha = rtr / dot(p, Ap, n);
        double* x_new = axpy(alpha, p, result.x, n);
        delete[] result.x;
        result.x = x_new;
        double* r_new = axpy(-alpha, Ap, r, n);
        delete[] r;
        r = r_new;
        double rtr_new = dot(r, r, n);
        double beta = rtr_new / rtr;
        double* p_new = axpy(beta, p, r, n);
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

double* generate_rhs(int n) {
    double* b = new double[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const double* x, int n, const std::string& filename) {
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
    double density = 0.001; // Controls sparsity (0 to 1)
    
    // Generate random SPD matrix
    CSRMatrix A = generate_random_spd_matrix(n, density);
    if (A.n == 0) return 1;

    double* b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    CGResult result = conjugate_gradient(A, b, A.n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = matvec(A, result.x, A.n);
    double* verify_vec = axpy(-1.0, Ax, b, A.n);
    double verify_residual = norm(verify_vec, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;

    write_solution(result.x, A.n, "results/cg/cg_solution.csv");

    delete[] b;
    delete[] Ax;
    delete[] verify_vec;

    return 0;
}