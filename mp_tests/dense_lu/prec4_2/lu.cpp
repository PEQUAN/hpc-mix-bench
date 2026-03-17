#include <half.hpp>
#include <floatx.hpp>
/**
 * Dense Matrix LU Decomposition Benchmark
 * =========================================
 * Generates matrices with specific condition numbers (1e1, 1e4) using
 * the same SVD-based method as MATLAB's gallery('randsvd', n, kappa).
 *
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <cassert>
#include <functional>

// ─────────────────────────────────────────────────────────────
// Flat 2-D matrix helpers
//   Row-major: element (i,j) of an n-column matrix → data[i*n + j]
// ─────────────────────────────────────────────────────────────

double* alloc_matrix(int n) {
    double* A = new double[n * n]();   // () → zero-initialised
    return A;
}

void free_matrix(double*& A) {
    delete[] A;
    A = nullptr;
}

flx::floatx<4, 3>* alloc_vector(int n) {
    return new flx::floatx<4, 3>[n]();
}

void free_vector(flx::floatx<4, 3>*& v) {
    delete[] v;
    v = nullptr;
}

#define M(A, i, j, n)  (A)[(i)*(n) + (j)]

// ─────────────────────────────────────────────────────────────
// Basic matrix / vector utilities
// ─────────────────────────────────────────────────────────────

// Set A = identity
void make_identity(double* A, int n) {
    for (int i = 0; i < n * n; ++i) A[i] = 0.0;
    for (int i = 0; i < n; ++i) M(A, i, i, n) = 1.0;
}

// C = A * B  (all n×n, flat row-major)
void mat_mul(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = 0.0;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            flx::floatx<4, 3> aik = M(A, i, k, n);
            if (aik == 0.0) continue;
            for (int j = 0; j < n; ++j)
                M(C, i, j, n) += aik * M(B, k, j, n);
        }
}

// C = A - B
void mat_sub(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = A[i] - B[i];
}

// b = A * x
void mat_vec(const double* A, const double* x, double* b, int n) {
    for (int i = 0; i < n; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < n; ++j)
            b[i] += M(A, i, j, n) * x[j];
    }
}

// Copy src → dst (n elements)
void vec_copy(const flx::floatx<4, 3>* src, flx::floatx<4, 3>* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

// c = a - b (vectors of length n)
void vec_sub(const double* a, const double* b, double* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

flx::floatx<4, 3> norm2(const double* v, int n) {
    float s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return sqrt(s);
}

flx::floatx<4, 3> normF(const double* A, int n) {
    float s = 0.0;
    int nn = n * n;
    for (int i = 0; i < nn; ++i) s += A[i] * A[i];
    return sqrt(s);
}

// Copy n×n matrix src → dst
void mat_copy(const double* src, double* dst, int n) {
    int nn = n * n;
    for (int i = 0; i < nn; ++i) dst[i] = src[i];
}

// ─────────────────────────────────────────────────────────────
// Gram–Schmidt QR → random orthogonal matrix
//   Q is n×n flat row-major; columns are orthonormal.
// ─────────────────────────────────────────────────────────────
void random_orthogonal(double* Q, int n, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);

    // Fill with random values (stored column-major temporarily for Gram–Schmidt
    // on columns; but we keep row-major and treat Q[i][j] as the j-th column)
    for (int i = 0; i < n * n; ++i) Q[i] = nd(rng);

    // Classical Gram–Schmidt on columns (column j = Q[:,j] = Q[0..n-1][j])
    for (int j = 0; j < n; ++j) {
        // Subtract projections onto previous columns
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) dot += M(Q, i, k, n) * M(Q, i, j, n);
            for (int i = 0; i < n; ++i) M(Q, i, j, n) -= dot * M(Q, i, k, n);
        }
        // Normalize column j
        double len = 0.0;
        for (int i = 0; i < n; ++i) len += M(Q, i, j, n) * M(Q, i, j, n);
        len = sqrt(len);
        for (int i = 0; i < n; ++i) M(Q, i, j, n) /= len;
    }
}

// ─────────────────────────────────────────────────────────────
// Generate n×n matrix with prescribed condition number kappa.
//   A = U * diag(sigma) * V^T
//   sigma_i geometrically spaced in [1, kappa].
// ─────────────────────────────────────────────────────────────
double* make_matrix_with_cond(int n, double kappa, std::mt19937_64& rng) {
    double* U = alloc_matrix(n);
    double* V = alloc_matrix(n);
    random_orthogonal(U, n, rng);
    random_orthogonal(V, n, rng);

    // Singular values: geometrically spaced 1 … kappa
    double* sigma = new double[n];
    for (int i = 0; i < n; ++i)
        sigma[i] = std::pow(kappa, static_cast<double>(i) / (n - 1));

    // A = U * Sigma * V^T
    double* A = alloc_matrix(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                M(A, i, j, n) += M(U, i, k, n) * sigma[k] * M(V, j, k, n);

    delete[] sigma;
    free_matrix(U);
    free_matrix(V);
    return A;    // caller owns this allocation
}

// ─────────────────────────────────────────────────────────────
// LU decomposition with partial pivoting  (in-place, flat row-major)
//
//   On entry : A is n×n.
//   On exit  : A overwritten with L (strictly lower) and U (upper).
//              piv[i] = row swapped into position i.
//
//   Returns false if matrix is singular to working precision.
// ─────────────────────────────────────────────────────────────
bool lu_factorize_core(double* A, int* piv, int n) {
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int k = 0; k < n; ++k) {
        // Find pivot row
        int    p       = k;
        flx::floatx<4, 3> max_val = abs(M(A, k, k, n));
        for (int i = k + 1; i < n; ++i) {
            flx::floatx<4, 3> v = abs(M(A, i, k, n));
            if (v > max_val) { max_val = v; p = i; }
        }
        if (max_val < 1e-15) return false;   // singular

        // Swap rows k ↔ p  (entire rows in flat storage)
        if (p != k) {
            for (int j = 0; j < n; ++j)
                std::swap(M(A, k, j, n), M(A, p, j, n));
            std::swap(piv[k], piv[p]);
        }

        // Eliminate below diagonal
        float akk = M(A, k, k, n);
        for (int i = k + 1; i < n; ++i) {
            M(A, i, k, n) /= akk;                            // multiplier → L
            float lik = M(A, i, k, n);
            for (int j = k + 1; j < n; ++j)
                M(A, i, j, n) -= lik * M(A, k, j, n);       // Schur complement
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────
// Forward substitution: solve L y = P b
//   L has unit diagonal; sub-diagonal stored in LU.
//   Returns newly allocated vector y (caller must delete[]).
// ─────────────────────────────────────────────────────────────
double* forward_sub(const double* LU, const int* piv, const double* b, int n) {
    double* y = new double[n];
    // Apply permutation
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];
    // Forward sweep
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            y[i] -= M(LU, i, j, n) * y[j];
    return y;
}

// Back substitution: solve U x = y
//   Returns newly allocated vector x (caller must delete[]).
double* back_sub(const double* LU, const double* y, int n) {
    double* x = new double[n];
    for (int i = 0; i < n; ++i) x[i] = y[i];
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            x[i] -= M(LU, i, j, n) * x[j];
        x[i] /= M(LU, i, i, n);
    }
    return x;
}

// ─────────────────────────────────────────────────────────────
// Public solve: x = A^{-1} b  using pre-factored LU
//   Returns newly allocated solution vector (caller must delete[]).
// ─────────────────────────────────────────────────────────────
double* lu_solve(const double* LU, const int* piv, const double* b, int n) {
    double* y = forward_sub(LU, piv, b, n);
    double* x = back_sub(LU, y, n);
    delete[] y;
    return x;
}

// ─────────────────────────────────────────────────────────────
// Reconstruct P, L, U from packed LU and pivot vector
//   Caller must free_matrix() P, L, U after use.
// ─────────────────────────────────────────────────────────────
void extract_PLU(const double* LU_packed, const int* piv,
                 double*& P, double*& L, double*& U, int n) {
    P = alloc_matrix(n);
    L = alloc_matrix(n);
    U = alloc_matrix(n);

    // Permutation: piv[i] = original row at position i
    for (int i = 0; i < n; ++i) M(P, i, piv[i], n) = 1.0;

    // Split packed LU into L (unit lower) and U (upper)
    make_identity(L, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (j < i)  M(L, i, j, n) = M(LU_packed, i, j, n);
            else        M(U, i, j, n) = M(LU_packed, i, j, n);
        }
}

// ─────────────────────────────────────────────────────────────
// LU entry point A  (dedicated to low-condition matrix)
// ─────────────────────────────────────────────────────────────
bool lu_factorize(double* A, int* piv, int n) {
    return lu_factorize_core(A, piv, n);
}

// ─────────────────────────────────────────────────────────────
// Run a complete benchmark for one matrix
// ─────────────────────────────────────────────────────────────
void benchmark(const std::string& label,
               const double* A_orig,
               int n,
               std::function<bool(double*, int*, int)> lu_fn) {

    // ── Build known RHS from x_true = [1, 2, …, n] ──────────
    double* x_true = new double[n];
    for (int i = 0; i < n; ++i) x_true[i] = static_cast<flx::floatx<4, 3>>(i + 1);

    double* b = new double[n];
    mat_vec(A_orig, x_true, b, n);

    // ── Factorize (timed) ────────────────────────────────────
    double* LU  = alloc_matrix(n);
    mat_copy(A_orig, LU, n);
    int*    piv = new int[n];

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = lu_fn(LU, piv, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    flx::floatx<4, 3> elapsed_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (!ok) {
        std::cout << label << " : factorization FAILED (singular matrix)\n";
        delete[] x_true; delete[] b; free_matrix(LU); delete[] piv;
        return;
    }

    // ── Solve ────────────────────────────────────────────────
    double* x_computed = lu_solve(LU, piv, b, n);

    // ── Accuracy metrics ─────────────────────────────────────
    // 1) Relative residual  ||Ax - b||_2 / ||b||_2
    double* Ax  = new double[n];
    mat_vec(A_orig, x_computed, Ax, n);
    double* res = new double[n];
    vec_sub(Ax, b, res, n);
    flx::floatx<4, 3> rel_residual = norm2(res, n) / norm2(b, n);

    // 2) Relative error  ||x - x_true||_2 / ||x_true||_2
    double* err = new double[n];
    vec_sub(x_computed, x_true, err, n);
    flx::floatx<4, 3> rel_error = norm2(err, n) / norm2(x_true, n);

    // 3) Backward error  ||PA - LU||_F / ||A||_F
    double *P_mat = nullptr, *L_mat = nullptr, *U_mat = nullptr;
    extract_PLU(LU, piv, P_mat, L_mat, U_mat, n);

    double* PA   = alloc_matrix(n);
    double* LU_m = alloc_matrix(n);
    double* diff = alloc_matrix(n);
    mat_mul(P_mat, A_orig, PA,   n);
    mat_mul(L_mat, U_mat,  LU_m, n);
    mat_sub(PA, LU_m, diff, n);
    flx::floatx<4, 3> back_err = normF(diff, n) / normF(A_orig, n);

    // ── Print results ────────────────────────────────────────
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  " << label << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Matrix size          : " << n << " x " << n << "\n";
    std::cout << "  Factorization time   : " << elapsed_ms << " ms\n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Relative residual    : " << rel_residual << "\n";
    std::cout << "  Relative error       : " << rel_error    << "\n";
    std::cout << "  Factorization err    : " << back_err     << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    // ── Clean up ─────────────────────────────────────────────
    delete[] x_true;
    delete[] b;
    free_matrix(LU);
    delete[] piv;
    delete[] x_computed;
    delete[] Ax;
    delete[] res;
    delete[] err;
    free_matrix(P_mat);
    free_matrix(L_mat);
    free_matrix(U_mat);
    free_matrix(PA);
    free_matrix(LU_m);
    free_matrix(diff);
}

int main() {
    const int    N     = 500;
    const flx::floatx<8, 7> KAPPA = 1e4;

    std::mt19937_64 rng(42);     

    std::cout << "\n";
    std::cout << "  Dense LU with Partial Pivoting  –  Accuracy & Timing\n";
    std::cout << "  Condition numbers: 1e4\n";
    std::cout << "  Matrix size: " << N << "x" << N << "\n\n";

    std::cout << "Time scaling vs. matrix size (cond = 1e4):\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(18) << "Factorize (ms)"
              << "Rel. residual\n";
    std::cout << std::string(40, '-') << "\n";

    double* A = make_matrix_with_cond(N, KAPPA, rng);

    double* x_true = new double[N];
    for (int i = 0; i < N; ++i) x_true[i] = 1.0;

    double* b = new double[N];
    mat_vec(A, x_true, b, N);

    double* LU  = alloc_matrix(N);
    mat_copy(A, LU, N);
    int*    piv = new int[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    lu_factorize(LU, piv, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    flx::floatx<4, 3> ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    double* x_comp = lu_solve(LU, piv, b, N);

    PROMISE_CHECK_ARRAY(x_comp, N);
    double* Ax     = new  double[N];
    mat_vec(A, x_comp, Ax, N);
    double* res    = new double[N];
    vec_sub(Ax, b, res, N);
    double rel_res = norm2(res, N) / norm2(b, N);


    std::cout << std::fixed
                << std::setw(10) << N
                << std::setw(18) << std::setprecision(3) << ms
                << std::scientific << std::setprecision(3) << rel_res << "\n";

    free_matrix(A);
    delete[] x_true;
    delete[] b;
    free_matrix(LU);
    delete[] piv;
    delete[] x_comp;
    delete[] Ax;
    delete[] res;

    std::cout << std::string(40, '-') << "\n";
    return 0;
}