#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

double* create_dense_matrix(int rows, int cols) {
    return new double[rows * cols]();
}

void free_dense_matrix(double* mat) {
    delete[] mat;
}

double* create_vector(int size) {
    return new double[size]();
}

void free_vector(double* vec) {
    delete[] vec;
}

template<typename T>
T* matrix_multiply(T* A, int rowsA, int colsA, T* B, int rowsB, int colsB) {
    if (colsA != rowsB) {
        std::cerr << "Matrix dimensions incompatible\n";
        return nullptr;
    }
    T* C = create_dense_matrix(rowsA, colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            T sum = T(0);
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
    return C;
}

template<typename T>
T* transpose(T* A, int rows, int cols) {
    T* T_mat = create_dense_matrix(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T_mat[j * rows + i] = A[i * cols + j];
        }
    }
    return T_mat;
}

void matvec(const double* A, int n, const double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void lu_factorization(const double* A, int n, double* L, double* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        double max_val = std::abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(U[i * n + k]) > max_val) {
                max_val = std::abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (std::abs(max_val) < 1e-15) {
            throw std::runtime_error("Matrix singular or nearly singular");
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(U[k * n + j], U[pivot * n + j]);
                if (j < k) {
                    std::swap(L[k * n + j], L[pivot * n + j]);
                }
            }
            std::swap(P[k], P[pivot]);
        }
        for (int i = k + 1; i < n; ++i) {
            L[i * n + k] = U[i * n + k] / U[k * n + k];
            for (int j = k; j < n; ++j) {
                U[i * n + j] -= L[i * n + k] * U[k * n + j];
            }
        }
    }
}

double* forward_substitution_init(const double* L, int n, const double* b, const int* P) {
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

double* backward_substitution_init(const double* U, int n, const double* y) {
    double* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

double* forward_substitution(const double* L, int n, const double* b, const int* P) {
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

double* backward_substitution(const double* U, int n, const double* y) {
    double* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

double* vec_sub(const double* a, const double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double* vec_add(const double* a, const double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

double* initial_solve(const double* L, const double* U, int n, const int* P, const double* b) {
    double* y = forward_substitution_init(L, n, b, P);
    double* x = backward_substitution_init(U, n, y);
    free_vector(y);
    return x;
}

double* compute_residual(const double* A, int n, const double* b, const double* x) {
    double* Ax = create_vector(n);
    matvec(A, n, x, Ax);
    double* r = vec_sub(b, Ax, n);
    free_vector(Ax);
    return r;
}

double* solve_correction(const double* L, const double* U, int n, const int* P, const double* r) {
    double* y = forward_substitution(L, n, r, P);
    double* d = backward_substitution(U, n, y);
    free_vector(y);
    return d;
}

double* update_solution(const double* x, const double* d, int n) {
    return vec_add(x, d, n);
}

void write_solution(const double* x, int size, const std::string& filename, const double* residual_history, int history_size, double final_ferr, double final_nbe, double final_cbe) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < size; ++i) {
        file << x[i] << "\n";
    }
    file << "\nResidual History\n";
    for (int i = 0; i < history_size; ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file << "\nFinal Errors\n";
    file << "Forward Error: " << final_ferr << "\n";
    file << "Normwise Backward Error: " << final_nbe << "\n";
    file << "Componentwise Backward Error: " << final_cbe << "\n";
    file.close();
}

double* iterative_refinement(const double* A, int n, const double* b, const double* x_true, double kappa, double tol, int max_iter, double*& residual_history, double*& ferr_history, double*& nbe_history, double*& cbe_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return nullptr;
    }

    history_size = 0;
    residual_history = new double[max_iter];
    ferr_history = new double[max_iter];
    nbe_history = new double[max_iter];
    cbe_history = new double[max_iter];

    double* L = create_dense_matrix(n, n);
    double* U = create_dense_matrix(n, n);
    int* P = new int[n];
    try {
        lu_factorization(A, n, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_dense_matrix(L);
        free_dense_matrix(U);
        return nullptr;
    }

    double* x = initial_solve(L, U, n, P, b);

    double u = std::numeric_limits<double>::epsilon(); // Machine epsilon
    double prev_dx_norm_inf = std::numeric_limits<double>::max(); // Previous correction norm

    std::cout << "u: " << u << std::endl;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual
        double* r = compute_residual(A, n, b, x);

        // Compute residual norm
        double norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = std::sqrt(norm_r);
        residual_history[history_size] = norm_r;

        // Compute forward error: max |x - x_true| / max |x_true|
        double ferr = 0.0;
        double x_true_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double err = std::abs(x[i] - x_true[i]);
            if (err > ferr) ferr = err;
            if (std::abs(x_true[i]) > x_true_norm) x_true_norm = std::abs(x_true[i]);
        }
        ferr_history[history_size] = x_true_norm > 0 ? ferr / x_true_norm : ferr;

        // Compute normwise backward error: ||r|| / (||A|| * ||x|| + ||b||)
        double x_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(x[i]) > x_norm) x_norm = std::abs(x[i]);
        }
        double A_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                row_sum += std::abs(A[i * n + j]);
            }
            if (row_sum > A_norm) A_norm = row_sum;
        }
        double b_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(b[i]) > b_norm) b_norm = std::abs(b[i]);
        }
        nbe_history[history_size] = norm_r / (A_norm * x_norm + b_norm);

        // Compute componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
        double* temp = new double[n];
        double cbe = 0.0;
        for (int i = 0; i < n; ++i) {
            double axb = 0.0;
            for (int j = 0; j < n; ++j) {
                axb += std::abs(A[i * n + j]) * std::abs(x[j]);
            }
            axb += std::abs(b[i]);
            temp[i] = axb > 0 ? std::abs(r[i]) / axb : 0.0;
            if (temp[i] > cbe) cbe = temp[i];
        }
        cbe_history[history_size] = cbe;
        delete[] temp;

        // Solve for correction
        double* d = solve_correction(L, U, n, P, r);
        free_vector(r);

        // Compute infinity norms for stopping criteria
        double dx_norm_inf = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(d[i]) > dx_norm_inf) dx_norm_inf = std::abs(d[i]);
        }
        double x_norm_inf = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(x[i]) > x_norm_inf) x_norm_inf = std::abs(x[i]);
        }

        // Check stopping criteria
        bool stop = false;
        if (x_norm_inf > 0 && dx_norm_inf / x_norm_inf <= u) {
            std::cout << "Converged after " << iter + 1 << " iterations: ||dx||_inf / ||x||_inf <= u\n";
            stop = true;
        } else if (prev_dx_norm_inf > 0 && dx_norm_inf / prev_dx_norm_inf >= tol) {
            std::cout << "Converged after " << iter + 1 << " iterations: ||dx^{(i+1)}||_inf / ||dx^{(i)}||_inf >= tol\n";
            stop = true;
        } else if (iter == max_iter - 1) {
            std::cout << "Stopped after " << max_iter << " iterations: maximum iterations reached\n";
            stop = true;
        }

        // Log iteration details
        std::cout << "Iteration " << iter << ": residual=" << residual_history[history_size]
                  << ", ferr=" << ferr_history[history_size]
                  << ", nbe=" << nbe_history[history_size]
                  << ", cbe=" << cbe_history[history_size]
                  << ", ||dx||_inf/||x||_inf=" << (x_norm_inf > 0 ? dx_norm_inf / x_norm_inf : 0.0)
                  << ", ||dx^{(i+1)}||_inf/||dx^{(i)}||_inf=" << (prev_dx_norm_inf > 0 ? dx_norm_inf / prev_dx_norm_inf : 0.0) << "\n";

        history_size++;
        prev_dx_norm_inf = dx_norm_inf;

        if (stop) {
            free_vector(d);
            break;
        }

        // Update solution
        double* x_new = update_solution(x, d, n);
        free_vector(d);
        free_vector(x);
        x = x_new;
    }

    free_dense_matrix(L);
    free_dense_matrix(U);
    delete[] P;
    return x;
}

// Function to read Matrix Market file and convert to dense matrix
double* read_matrix_market(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    // Skip header comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions and number of non-zeros
    int rows, cols, nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    if (rows != cols) {
        std::cerr << "Matrix must be square for this implementation\n";
        file.close();
        return nullptr;
    }
    n = rows;

    // Allocate dense matrix
    double* A = create_dense_matrix(n, n);

    // Read non-zero entries
    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading matrix entries\n";
            free_dense_matrix(A);
            file.close();
            return nullptr;
        }
        int row, col;
        double val;
        std::istringstream entry(line);
        entry >> row >> col >> val;
        // Matrix Market uses 1-based indexing, convert to 0-based
        A[(row - 1) * n + (col - 1)] = val;
    }

    file.close();
    return A;
}

int main() {
    int n;
    std::string matrix_file = "../data/suitesparse/1138_bus.mtx";
    double kappa = 1e6; // Condition number (approximate or estimated)
    double tol = 1e-8;   // Tolerance for convergence slowing criterion
    int max_iter = 2000; // Adjusted for larger matrix

    // Read matrix A from file
    double* A = read_matrix_market(matrix_file, n);
    if (!A) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    double* x_true = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    double* b = create_vector(n);
    matvec(A, n, x_true, b);

    double* residual_history = nullptr;
    double* ferr_history = nullptr;
    double* nbe_history = nullptr;
    double* cbe_history = nullptr;
    int history_size = 0;
    double* x = iterative_refinement(A, n, b, x_true, kappa, tol, max_iter, residual_history, ferr_history, nbe_history, cbe_history, history_size);

    if (x == nullptr) {
        std::cerr << "Failed to solve system\n";
        free_dense_matrix(A);
        free_vector(b);
        free_vector(x_true);
        delete[] residual_history;
        delete[] ferr_history;
        delete[] nbe_history;
        delete[] cbe_history;
        return 1;
    }

    double final_ferr = history_size > 0 ? ferr_history[history_size - 1] : 0.0;
    double final_nbe = history_size > 0 ? nbe_history[history_size - 1] : 0.0;
    double final_cbe = history_size > 0 ? cbe_history[history_size - 1] : 0.0;

    std::cout << "Final Forward Error: " << final_ferr << "\n";
    std::cout << "Final Normwise Backward Error: " << final_nbe << "\n";
    std::cout << "Final Componentwise Backward Error: " << final_cbe << "\n";

    std::string output_file = "solution.txt";
    write_solution(x, n, output_file, residual_history, history_size, final_ferr, final_nbe, final_cbe);

    free_dense_matrix(A);
    free_vector(b);
    free_vector(x_true);
    free_vector(x);
    delete[] residual_history;
    delete[] ferr_history;
    delete[] nbe_history;
    delete[] cbe_history;
    return 0;
}