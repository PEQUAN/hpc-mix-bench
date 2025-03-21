#include <cadna.h>
using namespace std;

#include <half_promise.hpp>
#include <floatx.hpp>
#include "./promise.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

// Dense matrix class
class Matrix {
private:
    std::vector<double_st> data;
    int n;

public:
    Matrix(int size) : n(size) {
        data.resize(n * n, 0.0);
    }

    double_st& operator()(int i, int j) {
        return data[i * n + j];
    }

    double_st operator()(int i, int j) const {
        return data[i * n + j];
    }

    int size() const { return n; }

    std::vector<double_st> matvec(const std::vector<double_st>& x) const {
        std::vector<double_st> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                y[i] += (*this)(i, j) * x[j];
            }
        }
        return y;
    }
};

// Vector operations
double_st dot(const std::vector<double_st>& a, const std::vector<double_st>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double_st norm(const std::vector<double_st>& v) {
    return std::sqrt(dot(v, v));
}

std::vector<double_st> axpy(double_st alpha, const std::vector<double_st>& x, const std::vector<double_st>& y) {
    std::vector<double_st> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

// Diagonal preconditioner
class DiagonalPreconditioner {
private:
    std::vector<double_st> diag_inv;
    int n;

public:
    DiagonalPreconditioner(const Matrix& A) : n(A.size()) {
        diag_inv.resize(n);
        for (int i = 0; i < n; ++i) {
            diag_inv[i] = (A(i, i) != 0.0) ? 1.0 / A(i, i) : 1.0;
        }
    }

    std::vector<double_st> apply(const std::vector<double_st>& r) const {
        std::vector<double_st> z(n);
        for (int i = 0; i < n; ++i) {
            z[i] = diag_inv[i] * r[i];
        }
        return z;
    }
};

// Preconditioned GMRES
struct GMRESResult {
    std::vector<double_st> x;
    double_st residual;
    int iterations;
};

GMRESResult gmres(const Matrix& A, const std::vector<double_st>& b, 
                  const DiagonalPreconditioner& M, int max_iter, double_st tol) {
    int n = A.size();
    std::vector<double_st> x(n, 0.0);  // Initial guess
    std::vector<double_st> r = axpy(-1.0, A.matvec(x), b);  // Initial residual
    double_st beta = norm(r);
    if (beta < tol) return {x, beta, 0};

    std::vector<std::vector<double_st>> V(max_iter + 1, std::vector<double_st>(n, 0.0));  // Krylov basis
    std::vector<std::vector<double_st>> H(max_iter + 1, std::vector<double_st>(max_iter, 0.0));  // Hessenberg matrix
    std::vector<double_st> c(max_iter), s(max_iter);  // Givens rotation coefficients
    std::vector<double_st> g(max_iter + 1);  // Residual vector

    V[0] = r;
    for (int i = 0; i < n; ++i) V[0][i] /= beta;
    g[0] = beta;

    int k;
    for (k = 0; k < max_iter && beta > tol; ++k) {
        // Arnoldi iteration with preconditioning
        std::vector<double_st> w = A.matvec(M.apply(V[k]));
        for (int j = 0; j <= k; ++j) {
            H[j][k] = dot(w, V[j]);
            for (int i = 0; i < n; ++i) {
                w[i] -= H[j][k] * V[j][i];
            }
        }
        H[k + 1][k] = norm(w);
        if (H[k + 1][k] < 1e-10) break;  // Lucky breakdown (exact solution found)
        V[k + 1] = w;
        for (int i = 0; i < n; ++i) V[k + 1][i] /= H[k + 1][k];

        // Apply previous Givens rotations
        for (int j = 0; j < k; ++j) {
            double_st temp = c[j] * H[j][k] + s[j] * H[j + 1][k];
            H[j + 1][k] = -s[j] * H[j][k] + c[j] * H[j + 1][k];
            H[j][k] = temp;
        }

        // Compute new Givens rotation
        double_st rho = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
        if (rho < 1e-10) break;
        c[k] = H[k][k] / rho;
        s[k] = H[k + 1][k] / rho;
        H[k][k] = rho;
        H[k + 1][k] = 0.0;

        // Update residual vector
        double_st g_temp = g[k];
        g[k] = c[k] * g[k] + s[k] * g[k + 1];
        g[k + 1] = -s[k] * g_temp + c[k] * g[k + 1];
        beta = std::abs(g[k + 1]);
    }

    // Solve upper triangular system H y = g
    std::vector<double_st> y(k);
    for (int i = k - 1; i >= 0; --i) {
        y[i] = g[i];
        for (int j = i + 1; j < k; ++j) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }

    // Update solution
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            x[i] += y[j] * M.apply(V[j])[i];
        }
    }

    r = axpy(-1.0, A.matvec(x), b);
    double_st residual = norm(r);
    return {x, residual, k};
}

// Generate a tridiagonal test matrix
Matrix generate_tridiagonal(int n) {
    Matrix A(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0);

    for (int i = 0; i < n; ++i) {
        A(i, i) = 4.0 + dis(gen);  // Stronger diagonal dominance
        if (i > 0) A(i, i - 1) = -1.0 + 0.1 * dis(gen);
        if (i < n - 1) A(i, i + 1) = -1.0 + 0.1 * dis(gen);
    }
    return A;
}

std::vector< double_st> generate_rhs(int n) {
    std::vector< double_st> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    cadna_init(0);

    int n = 1000;  // Matrix size
    Matrix A = generate_tridiagonal(n);
    std::vector<double_st> b = generate_rhs(n);
    DiagonalPreconditioner M(A);

    auto start = std::chrono::high_resolution_clock::now();
    GMRESResult result = gmres(A, b, M, n, 1e-6);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    // Verify solution
    std::vector<double_st> Ax = A.matvec(result.x);
    double_st error = norm(axpy(-1.0, Ax, b));
    PROMISE_CHECK_VAR(error);
    std::cout << "Verification residual: " << error << std::endl;

    return 0;

    cadna_end();
}