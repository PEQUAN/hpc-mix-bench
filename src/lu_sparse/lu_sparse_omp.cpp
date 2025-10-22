
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>

double* create_vector(int size, bool initialize = true) {
    double* vec = initialize ? new double[size]() : new double[size];
    if (!vec) {
        throw std::runtime_error("Failed to allocate vector");
    }
    return vec;
}

void free_vector(double* vec) {
    delete[] vec;
}

int* create_int_vector(int size, bool initialize = true) {
    int* vec = initialize ? new int[size]() : new int[size];
    if (!vec) {
        throw std::runtime_error("Failed to allocate int vector");
    }
    return vec;
}

void free_int_vector(int* vec) {
    delete[] vec;
}

struct SparseMatrix {
    double* val; 
    int* col_ind; 
    int* row_ptr; 
    int rows; 
    int cols; 
    int nnz; 
};

void free_sparse_matrix(SparseMatrix& mat) {
    if (mat.val) delete[] mat.val;
    if (mat.col_ind) delete[] mat.col_ind;
    if (mat.row_ptr) delete[] mat.row_ptr;
    mat.val = nullptr;
    mat.col_ind = nullptr;
    mat.row_ptr = nullptr;
    mat.nnz = 0;
}

SparseMatrix read_matrix_market(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return {nullptr, nullptr, nullptr, 0, 0, 0};
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    int rows, cols, nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    if (rows != cols) {
        std::cerr << "Matrix must be square for this implementation\n";
        file.close();
        return {nullptr, nullptr, nullptr, 0, 0, 0};
    }
    n = rows;

    double* temp_val = create_vector(nnz, false);
    int* temp_row = create_int_vector(nnz, false);
    int* temp_col = create_int_vector(nnz, false);

    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading matrix entries\n";
            free_vector(temp_val);
            free_int_vector(temp_row);
            free_int_vector(temp_col);
            file.close();
            return {nullptr, nullptr, nullptr, 0, 0, 0};
        }
        std::istringstream entry(line);
        entry >> temp_row[i] >> temp_col[i] >> temp_val[i];
        temp_row[i]--; // Convert to 0-based indexing
        temp_col[i]--;
    }
    file.close();

    double* val = create_vector(nnz, false);
    int* col_ind = create_int_vector(nnz, false);
    int* row_ptr = create_int_vector(rows + 1);

    for (int i = 0; i < nnz; ++i) {
        row_ptr[temp_row[i] + 1]++;
    }
    for (int i = 1; i <= rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }
    int* row_counts = create_int_vector(rows);
    for (int i = 0; i < nnz; ++i) {
        int r = temp_row[i];
        int pos = row_ptr[r] + row_counts[r];
        val[pos] = temp_val[i];
        col_ind[pos] = temp_col[i];
        row_counts[r]++;
    }

    free_vector(temp_val);
    free_int_vector(temp_row);
    free_int_vector(temp_col);
    free_int_vector(row_counts);

    return {val, col_ind, row_ptr, rows, cols, nnz};
}

void sparse_matvec(const SparseMatrix& A, const double* x, double* y) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid sparse matrix in sparse_matvec");
    }
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.val[j] * x[A.col_ind[j]];
        }
    }
}

void dense_lu_factorization(const SparseMatrix& A, double*& L, double*& U, int* P, double& U_norm) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid input matrix");
    }
    int n = A.rows;

    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    double* dense_A = create_vector(n * n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense_A[i * n + A.col_ind[j]] = A.val[j];
        }
    }

    L = create_vector(n * n);
    U = create_vector(n * n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = dense_A[i * n + j];
            L[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double A_norm = 0.0;
    #pragma omp parallel for reduction(max:A_norm)
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += std::abs(A.val[j]);
        }
        if (row_sum > A_norm) A_norm = row_sum;
    }

    for (int k = 0; k < n; ++k) {
        double max_val = std::abs(U[k * n + k]);
        int pivot = k;
        #pragma omp parallel for reduction(max:max_val)
        for (int i = k + 1; i < n; ++i) {
            double val = std::abs(U[i * n + k]);
            if (val > max_val) {
                #pragma omp critical
                {
                    if (val > max_val) {
                        max_val = val;
                        pivot = i;
                    }
                }
            }
        }
        if (std::abs(max_val) < 1e-15 * A_norm) {
            throw std::runtime_error("Matrix singular or nearly singular during factorization");
        }
        if (pivot != k) {
            #pragma omp parallel for
            for (int j = 0; j < n; ++j) {
                std::swap(U[k * n + j], U[pivot * n + j]);
                if (j < k) {
                    std::swap(L[k * n + j], L[pivot * n + j]);
                }
            }
            std::swap(P[k], P[pivot]);
        }
        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            L[i * n + k] = U[i * n + k] / U[k * n + k];
            for (int j = k; j < n; ++j) {
                U[i * n + j] -= L[i * n + k] * U[k * n + j];
            }
        }
    }

    // Compute U_norm for use in backward substitution
    U_norm = 0.0;
    #pragma omp parallel for reduction(max:U_norm)
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = i; j < n; ++j) {
            row_sum += std::abs(U[i * n + j]);
        }
        if (row_sum > U_norm) U_norm = row_sum;
    }

    free_vector(dense_A);
}

double* dense_forward_substitution(const double* L, int n, const double* b, const int* P) {
    if (!L || !P) {
        throw std::runtime_error("Invalid inputs in dense_forward_substitution");
    }
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

double* dense_backward_substitution(const double* U, int n, const double* y, double U_norm) {
    if (!U) {
        throw std::runtime_error("Invalid inputs in dense_backward_substitution");
    }
    double* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        if (std::abs(U[i * n + i]) < 1e-15 * U_norm) {
            free_vector(x);
            throw std::runtime_error("U is singular or nearly singular");
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

// Iterative refinement to improve solution
void iterative_refinement(const SparseMatrix& A, double* x, const double* b, const double* L, const double* U, const int* P, int n, int max_iter = 2) {
    double* r = create_vector(n);
    double* z = create_vector(n);
    double* y = create_vector(n);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual: r = b - Ax
        sparse_matvec(A, x, r);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
        }

        // Solve Lz = Pr
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int j = 0; j < i; ++j) {
                sum += L[i * n + j] * z[j];
            }
            z[i] = r[P[i]] - sum;
        }

        // Solve Ux_new = z, update x = x + x_new
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int j = i + 1; j < n; ++j) {
                sum += U[i * n + j] * y[j];
            }
            y[i] = (z[i] - sum) / U[i * n + i];
        }

        // Update x
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] += y[i];
        }
    }

    free_vector(r);
    free_vector(z);
    free_vector(y);
}

void compute_errors(const SparseMatrix& A, int n, const double* b, const double* x, const double* x_true, double& ferr, double& nbe, double& cbe) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid sparse matrix in compute_errors");
    }
    double x_true_norm = 0.0;
    ferr = 0.0;
    #pragma omp parallel for reduction(max:ferr, x_true_norm)
    for (int i = 0; i < n; ++i) {
        double err = std::abs(x[i] - x_true[i]);
        if (err > ferr) ferr = err;
        if (std::abs(x_true[i]) > x_true_norm) x_true_norm = std::abs(x_true[i]);
    }
    ferr = x_true_norm > 0 ? ferr / x_true_norm : ferr;

    double* Ax = create_vector(n);
    sparse_matvec(A, x, Ax);
    double* r = create_vector(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }

    double norm_r = 0.0;
    #pragma omp parallel for reduction(+:norm_r)
    for (int i = 0; i < n; ++i) {
        norm_r += r[i] * r[i];
    }
    norm_r = std::sqrt(norm_r);
    double x_norm = 0.0;
    #pragma omp parallel for reduction(max:x_norm)
    for (int i = 0; i < n; ++i) {
        if (std::abs(x[i]) > x_norm) x_norm = std::abs(x[i]);
    }
    double A_norm = 0.0;
    #pragma omp parallel for reduction(max:A_norm)
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += std::abs(A.val[j]);
        }
        if (row_sum > A_norm) A_norm = row_sum;
    }
    double b_norm = 0.0;
    #pragma omp parallel for reduction(max:b_norm)
    for (int i = 0; i < n; ++i) {
        if (std::abs(b[i]) > b_norm) b_norm = std::abs(b[i]);
    }
    nbe = (A_norm * x_norm + b_norm) > 0 ? norm_r / (A_norm * x_norm + b_norm) : norm_r;

    cbe = 0.0;
    #pragma omp parallel for reduction(max:cbe)
    for (int i = 0; i < n; ++i) {
        double axb = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            axb += std::abs(A.val[j]) * std::abs(x[A.col_ind[j]]);
        }
        axb += std::abs(b[i]);
        double temp = axb > 0 ? std::abs(r[i]) / axb : 0.0;
        if (temp > cbe) cbe = temp;
    }

    free_vector(Ax);
    free_vector(r);
}

void write_solution(const double* x, int size, const std::string& filename, double ferr, double nbe, double cbe) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(6);
    file << "Dense LU Solution\n";
    for (int i = 0; i < size; ++i) {
        file << x[i] << "\n";
    }
    file << "\nErrors\n";
    file << "Forward Error: " << ferr << "\n";
    file << "Normwise Backward Error: " << nbe << "\n";
    file << "Componentwise Backward Error: " << cbe << "\n";
    file.close();
}

int main(int argc, char* argv[]) {
    // omp_set_num_threads(4); // Adjust based on system
    int n;
    std::string matrix_file = (argc > 1) ? argv[1] : "../../data/suitesparse/psmigr_2.mtx";

    SparseMatrix A = read_matrix_market(matrix_file, n);
    if (!A.val) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    double* x_true = nullptr;
    double* b = nullptr;
    try {
        x_true = create_vector(n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x_true[i] = 1.0;
        }
        b = create_vector(n);
        sparse_matvec(A, x_true, b);
    } catch (const std::exception& e) {
        std::cerr << "Error setting up true solution: " << e.what() << "\n";
        free_sparse_matrix(A);
        if (x_true) free_vector(x_true);
        if (b) free_vector(b);
        return 1;
    }

    double* L = nullptr;
    double* U = nullptr;
    int* P = nullptr;
    double U_norm = 0.0;
    try {
        P = create_int_vector(n);
        dense_lu_factorization(A, L, U, P, U_norm);
    } catch (const std::exception& e) {
        std::cerr << "Dense LU factorization failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_vector(x_true);
        free_vector(b);
        if (P) free_int_vector(P);
        if (L) free_vector(L);
        if (U) free_vector(U);
        return 1;
    }

    double* y = nullptr;
    double* x = nullptr;
    try {
        y = dense_forward_substitution(L, n, b, P);
        x = dense_backward_substitution(U, n, y, U_norm);
        // Perform iterative refinement
        iterative_refinement(A, x, b, L, U, P, n, 2);
    } catch (const std::exception& e) {
        std::cerr << "Dense LU solve failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_vector(x_true);
        free_vector(b);
        free_vector(y);
        free_vector(x);
        free_vector(L);
        free_vector(U);
        free_int_vector(P);
        return 1;
    }

    double ferr, nbe, cbe;
    try {
        compute_errors(A, n, b, x, x_true, ferr, nbe, cbe);
    } catch (const std::exception& e) {
        std::cerr << "Error computing errors: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_vector(x_true);
        free_vector(b);
        free_vector(y);
        free_vector(x);
        free_vector(L);
        free_vector(U);
        free_int_vector(P);
        return 1;
    }

    std::cout << "Dense LU Solver Results:\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Forward Error: " << ferr << "\n";
    std::cout << "Normwise Backward Error: " << nbe << "\n";
    std::cout << "Componentwise Backward Error: " << cbe << "\n";
    write_solution(x, n, "solution_lu.txt", ferr, nbe, cbe);

    free_sparse_matrix(A);
    free_vector(L);
    free_vector(U);
    free_int_vector(P);
    free_vector(b);
    free_vector(y);
    free_vector(x);
    free_vector(x_true);

    return 0;
}