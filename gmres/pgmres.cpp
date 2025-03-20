#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

// Dense matrix class
class Matrix {
private:
    std::vector<double> data;
    int n;

public:
    Matrix(int size) : n(size) {
        data.resize(n * n, 0.0);
    }

    double& operator()(int i, int j) {
        return data[i * n + j];
    }

    double operator()(int i, int j) const {
        return data[i * n + j];
    }

    int size() const { return n; }

    std::vector<double> matvec(const std::vector<double>& x) const {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                y[i] += (*this)(i, j) * x[j];
            }
        }
        return y;
    }
};

// Vector operations
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

std::vector<double> axpy(double alpha, const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

// Diagonal preconditioner
class DiagonalPreconditioner {
private:
    std::vector<double> diag_inv;
    int n;

public:
    DiagonalPreconditioner(const Matrix& A) : n(A.size()) {
        diag_inv.resize(n);
        for (int i = 0; i < n; ++i) {
            diag_inv[i] = (A(i, i) != 0.0) ? 1.0 / A(i, i) : 1.0;
        }
    }

    std::vector<double> apply(const std::vector<double>& r) const {
        std::vector<double> z(n);
        for (int i = 0; i < n; ++i) {
            z[i] = diag_inv[i] * r[i];
        }
        return z;
    }
};

// Preconditioned GMRES
struct GMRESResult {
    std::vector<double> x;
    double residual;
    int iterations;
};

GMRESResult gmres(const Matrix& A, const std::vector<double>& b, 
                  const DiagonalPreconditioner& M, int max_iter = 100, double tol = 1e-6) {
    int n = A.size();
    std::vector<double> x(n, 0.0);  // Initial guess
    std::vector<double> r = axpy(-1.0, A.matvec(x), b);  // Initial residual
    double beta = norm(r);
    std::vector<std::vector<double>> V(n);  // Krylov basis
    std::vector<std::vector<double>> H(max_iter + 1, std::vector<double>(max_iter, 0.0));  // Hessenberg matrix
    std::vector<double> s(max_iter + 1, 0.0);  // Right-hand side for least squares
    s[0] = beta;

    if (beta < tol) {
        return {x, beta, 0};
    }

    V[0] = r;
    for (double& v : V[0]) v /= beta;

    int k;
    for (k = 0; k < max_iter && beta > tol; ++k) {
        // Arnoldi iteration
        std::vector<double> w = A.matvec(M.apply(V[k]));
        for (int j = 0; j <= k; ++j) {
            H[j][k] = dot(w, V[j]);
            w = axpy(-H[j][k], V[j], w);
        }
        H[k + 1][k] = norm(w);
        if (H[k + 1][k] < 1e-10) break;  // Breakdown
        V[k + 1] = w;
        for (double& v : V[k + 1]) v /= H[k + 1][k];

        // Apply Givens rotations to H and s
        for (int j = 0; j < k; ++j) {
            double temp = H[j][k];
            H[j][k] = H[j][k] * H[j][j] + H[j + 1][j] * H[j + 1][k];
            H[j + 1][k] = -H[j + 1][j] * temp + H[j][j] * H[j + 1][k];
        }
        double rho = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
        if (rho < 1e-10) break;
        H[k][k] = rho;
        H[k + 1][k] = 0.0;
        double c = H[k][k] / rho;
        double s_rot = H[k + 1][k] / rho;
        H[k][k] = c * H[k][k] + s_rot * H[k + 1][k];
        s[k + 1] = -s_rot * s[k];
        s[k] *= c;

        beta = std::abs(s[k + 1]);
    }

    // Solve upper triangular system Hy = s
    std::vector<double> y(k);
    for (int i = k - 1; i >= 0; --i) {
        y[i] = s[i];
        for (int j = i + 1; j < k; ++j) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }

    // Update solution
    for (int i = 0; i < k; ++i) {
        x = axpy(y[i], M.apply(V[i]), x);
    }

    r = axpy(-1.0, A.matvec(x), b);
    double residual = norm(r);

    return {x, residual, k};
}

// Generate a tridiagonal test matrix
Matrix generate_tridiagonal(int n) {
    Matrix A(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0);

    for (int i = 0; i < n; ++i) {
        A(i, i) = 2.0 + dis(gen);  // Diagonal
        if (i > 0) A(i, i - 1) = -1.0 + 0.1 * dis(gen);  // Lower diagonal
        if (i < n - 1) A(i, i + 1) = -1.0 + 0.1 * dis(gen);  // Upper diagonal
    }
    return A;
}

std::vector<double> generate_rhs(int n) {
    std::vector<double> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    int n = 1000;  // Matrix size
    Matrix A = generate_tridiagonal(n);
    std::vector<double> b = generate_rhs(n);
    DiagonalPreconditioner M(A);

    auto start = std::chrono::high_resolution_clock::now();
    GMRESResult result = gmres(A, b, M, n, 1e-12);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    // Verify solution
    std::vector<double> Ax = A.matvec(result.x);
    double error = norm(axpy(-1.0, Ax, b));
    std::cout << "Verification residual: " << error << std::endl;

    return 0;
}