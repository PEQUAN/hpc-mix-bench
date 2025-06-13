#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

class Matrix {
private:
    double* data;
    int n;

public:
    Matrix(int size) : n(size) {
        data = new double[n * n](); 
    }

    ~Matrix() {
        delete[] data;
    }

    double get(int i, int j) const {
        return data[i * n + j];
    }

    void set(int i, int j, double val) {
        data[i * n + j] = val;
    }

    int size() const { return n; }

    double* matvec(const double* x) const {
        double* y = new double[n]();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                y[i] += get(i, j) * x[j];
            }
        }
        return y;
    }
};

float dot(const double* a, const double* b, int n) {
    float result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float norm(const double* v, int n) {
    return sqrt(dot(v, v, n));
}

double* axpy(float alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

class DiagonalPreconditioner {
private:
    half_float::half* diag_inv;
    int n;

public:
    DiagonalPreconditioner(const Matrix& A) : n(A.size()) {
        diag_inv = new half_float::half[n];
        for (int i = 0; i < n; ++i) {
            if (A.get(i, i) != 0.0) {
                diag_inv[i] = 1.0 / A.get(i, i);
            } else {
                diag_inv[i] = 1.0;
            }
        }
    }

    ~DiagonalPreconditioner() {
        delete[] diag_inv;
    }

    double* apply(const double* r) const {
        double* z = new double[n];
        for (int i = 0; i < n; ++i) {
            z[i] = diag_inv[i] * r[i];
        }
        return z;
    }
};

struct GMRESResult {
    double* x;
    float residual;
    int iterations;
};

GMRESResult gmres(const Matrix& A, const double* b, 
                  const DiagonalPreconditioner& M, int max_iter, float tol) {
    int n = A.size();
    double* x = new double[n](); 
    double* Ax = A.matvec(x);
    double* r = axpy(-1.0, Ax, b, n); 
    delete[] Ax;
    float beta = norm(r, n);
    if (beta < tol) {
        delete[] r;
        return {x, beta, 0};
    }

    double** V = new double*[max_iter + 1];
    for (int i = 0; i <= max_iter; ++i) {
        V[i] = new double[n]();
    }
    double** H = new double*[max_iter + 1];
    for (int i = 0; i <= max_iter; ++i) {
        H[i] = new double[max_iter]();
    }
    double* c = new double[max_iter]();
    double* s = new double[max_iter]();
    double* g = new double[max_iter + 1]();

    for (int i = 0; i < n; ++i) {
        V[0][i] = r[i] / beta;
    }
    g[0] = beta;

    int k;
    for (k = 0; k < max_iter && beta > tol; ++k) {
        double* Vk = M.apply(V[k]);
        double* w = A.matvec(Vk);
        delete[] Vk;
        for (int j = 0; j <= k; ++j) {
            H[j][k] = dot(w, V[j], n);
            for (int i = 0; i < n; ++i) {
                w[i] -= H[j][k] * V[j][i];
            }
        }
        H[k + 1][k] = norm(w, n);
        if (H[k + 1][k] < 1e-10) {
            delete[] w;
            break; 
        }
        for (int i = 0; i < n; ++i) {
            V[k + 1][i] = w[i] / H[k + 1][k];
        }
        delete[] w;

        for (int j = 0; j < k; ++j) {
            float temp = c[j] * H[j][k] + s[j] * H[j + 1][k];
            H[j + 1][k] = -s[j] * H[j][k] + c[j] * H[j + 1][k];
            H[j][k] = temp;
        }

        // Compute new Givens rotation
        float rho = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
        if (rho < 1e-10) break;
        c[k] = H[k][k] / rho;
        s[k] = H[k + 1][k] / rho;
        H[k][k] = rho;
        H[k + 1][k] = 0.0;

        // Update residual vector
        float g_temp = g[k];
        g[k] = c[k] * g[k] + s[k] * g[k + 1];
        g[k + 1] = -s[k] * g_temp + c[k] * g[k + 1];
        beta = abs(g[k + 1]);
    }

    float* y = new float[k]();
    for (int i = k - 1; i >= 0; --i) {
        y[i] = g[i];
        for (int j = i + 1; j < k; ++j) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }

    for (int j = 0; j < k; ++j) {
        double* Mvj = M.apply(V[j]);
        for (int i = 0; i < n; ++i) {
            x[i] += y[j] * Mvj[i];
        }
        delete[] Mvj;
    }

    Ax = A.matvec(x);
    double* r_new = axpy(-1.0, Ax, b, n);
    float residual = norm(r_new, n);
    delete[] Ax;
    delete[] r_new;
    delete[] r;
    delete[] y;
    for (int i = 0; i <= max_iter; ++i) {
        delete[] V[i];
        delete[] H[i];
    }
    delete[] V;
    delete[] H;
    delete[] c;
    delete[] s;
    delete[] g;

    return {x, residual, k};
}

Matrix generate_tridiagonal(int n) {
    Matrix A(n);
    std::mt19937 gen(2025);
    std::uniform_real_distribution<> dis(0.1, 1.0);

    double val; 
    for (int i = 0; i < n; ++i) {
        val = 4.0 + dis(gen); 
        A.set(i, i, val);    
        val = -1.0 + 0.1 * dis(gen);               
        if (i > 0) A.set(i, i - 1, val);
        val = -1.0 + 0.1 * dis(gen); 
        if (i < n - 1) A.set(i, i + 1, val);
    }
    return A;
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
    int n = 1000; 
    Matrix A = generate_tridiagonal(n);
    double* b = generate_rhs(n);
    DiagonalPreconditioner M(A);

    GMRESResult result = gmres(A, b, M, n, 1e-6);
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = A.matvec(result.x);
    double* error_vec = axpy(-1.0, Ax, b, n);
    float error = norm(error_vec, n);
    std::cout << "Verification residual: " << error << std::endl;

    PROMISE_CHECK_ARRAY(Ax, n);
    delete[] Ax;
    delete[] error_vec;
    delete[] result.x;
    delete[] b;

    return 0;
}