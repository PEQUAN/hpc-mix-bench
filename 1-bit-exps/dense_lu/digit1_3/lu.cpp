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
#include <limits>
#include <type_traits>

// ─────────────────────────────────────────────────────────────
// Flat 2-D matrix helpers
//   Row-major: element (i,j) of an n-column matrix → data[i*n + j]
// ─────────────────────────────────────────────────────────────

template <typename T>
void free_matrix(T*& A) {
    delete[] A;
    A = nullptr;
}

template <typename T>
void free_vector(T*& v) {
    delete[] v;
    v = nullptr;
}

#define M(A, i, j, n)  (A)[(i)*(n) + (j)]

double PIVOT_TOLERANCE_FACTOR = 100.0;

template <typename T>
T promise_sqrt(T x) {
    return static_cast<T>(std::sqrt(static_cast<double>(x)));
}


template <typename T>
T promise_pow(T x, T y) {
    return static_cast<T>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

// C++11-compatible SFINAE keeps this file buildable with the existing flags.
template <typename T>
typename std::enable_if<std::numeric_limits<T>::is_specialized, T>::type numeric_epsilon_for() {
    return std::numeric_limits<T>::epsilon();
}

template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_specialized, T>::type numeric_epsilon_for() {
    return static_cast<T>(std::numeric_limits<double>::epsilon());
}

// ─────────────────────────────────────────────────────────────
// Basic matrix / vector utilities
// ─────────────────────────────────────────────────────────────

// Set A = identity
template <typename T>
void make_identity(T* A, int n) {
    for (int i = 0; i < n * n; ++i) A[i] = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i) M(A, i, i, n) = static_cast<T>(1.0);
}

// C = A * B  (all n×n, flat row-major)
template <typename AType, typename BType, typename CType>
void mat_mul(const AType* A, const BType* B, CType* C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = static_cast<CType>(0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            AType aik = M(A, i, k, n);
            if (aik == static_cast<AType>(0.0)) continue;
            for (int j = 0; j < n; ++j)
                M(C, i, j, n) += aik * M(B, k, j, n);
        }
}

// C = A - B
template <typename AType, typename BType, typename CType>
void mat_sub(const AType* A, const BType* B, CType* C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = A[i] - B[i];
}

// b = A * x
template <typename AType, typename XType, typename BType>
void mat_vec(const AType* A, const XType* x, BType* b, int n) {
    for (int i = 0; i < n; ++i) {
        b[i] = static_cast<BType>(0.0);
        for (int j = 0; j < n; ++j)
            b[i] += M(A, i, j, n) * x[j];
    }
}

// Copy src → dst (n elements)
template <typename SrcType, typename DstType>
void vec_copy(const SrcType* src, DstType* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

// c = a - b (vectors of length n)
template <typename AType, typename BType, typename CType>
void vec_sub(const AType* a, const BType* b, CType* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

template <typename T>
T norm2(const T* v, int n) {
    T s = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return promise_sqrt(s);
}

template <typename T>
T normF(const T* A, int n) {
    T s = static_cast<T>(0.0);
    int nn = n * n;
    for (int i = 0; i < nn; ++i) s += A[i] * A[i];
    return promise_sqrt(s);
}

// Copy n×n matrix src → dst
template <typename SrcType, typename DstType>
void mat_copy(const SrcType* src, DstType* dst, int n) {
    int nn = n * n;
    for (int i = 0; i < nn; ++i) dst[i] = src[i];
}

// ─────────────────────────────────────────────────────────────
// Gram–Schmidt QR → random orthogonal matrix
//   Q is n×n flat row-major; columns are orthonormal.
// ─────────────────────────────────────────────────────────────
template <typename T>
void random_orthogonal(T* Q, int n, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);

    // Fill from a double distribution so every precision variant starts from
    // the same deterministic values before conversion to T.
    for (int i = 0; i < n * n; ++i) Q[i] = static_cast<T>(nd(rng));

    // Classical Gram–Schmidt on columns (column j = Q[:,j] = Q[0..n-1][j])
    for (int j = 0; j < n; ++j) {
        // Subtract projections onto previous columns
        for (int k = 0; k < j; ++k) {
            T dot = static_cast<T>(0.0);
            for (int i = 0; i < n; ++i) dot += M(Q, i, k, n) * M(Q, i, j, n);
            for (int i = 0; i < n; ++i) M(Q, i, j, n) -= dot * M(Q, i, k, n);
        }
        // Normalize column j
        T len = static_cast<T>(0.0);
        for (int i = 0; i < n; ++i) len += M(Q, i, j, n) * M(Q, i, j, n);
        len = promise_sqrt(len);
        for (int i = 0; i < n; ++i) M(Q, i, j, n) /= len;
    }
}

// ─────────────────────────────────────────────────────────────
// Generate n×n matrix with prescribed condition number kappa.
//   A = U * diag(sigma) * V^T
//   sigma_i geometrically spaced in [1, kappa].
// ─────────────────────────────────────────────────────────────
template <typename T>
void make_matrix_with_cond(T* A, int n, double kappa, std::mt19937_64& rng) {
    T* U = new T[n * n]();
    T* V = new T[n * n]();
    random_orthogonal(U, n, rng);
    random_orthogonal(V, n, rng);

    // Singular values: geometrically spaced 1 … kappa
    T* sigma = new T[n];
    for (int i = 0; i < n; ++i) {
        // For n == 1, use the lower endpoint kappa^0 to avoid 0/0.
        double exponent = n > 1 ? static_cast<double>(i) / (n - 1) : 0.0;
        sigma[i] = static_cast<T>(promise_pow(kappa, exponent));
    }

    // A = U * Sigma * V^T
    for (int i = 0; i < n * n; ++i) A[i] = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                M(A, i, j, n) += M(U, i, k, n) * sigma[k] * M(V, j, k, n);

    delete[] sigma;
    free_matrix(U);
    free_matrix(V);
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
template <typename T>
bool lu_factorize_core(T* A, int* piv, int n) {
    const T pivot_tol = static_cast<T>(PIVOT_TOLERANCE_FACTOR) * numeric_epsilon_for<T>();
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int k = 0; k < n; ++k) {
        // Find pivot row
        int    p       = k;
        T max_val = abs(M(A, k, k, n));
        for (int i = k + 1; i < n; ++i) {
            T v = abs(M(A, i, k, n));
            if (v > max_val) { max_val = v; p = i; }
        }
        if (max_val < pivot_tol) return false;   // singular

        // Swap rows k ↔ p  (entire rows in flat storage)
        if (p != k) {
            for (int j = 0; j < n; ++j)
                std::swap(M(A, k, j, n), M(A, p, j, n));
            std::swap(piv[k], piv[p]);
        }

        // Eliminate below diagonal
        T akk = M(A, k, k, n);
        for (int i = k + 1; i < n; ++i) {
            M(A, i, k, n) /= akk;                            // multiplier → L
            T lik = M(A, i, k, n);
            for (int j = k + 1; j < n; ++j)
                M(A, i, j, n) -= lik * M(A, k, j, n);       // Schur complement
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────
// Forward substitution: solve L y = P b
//   L has unit diagonal; sub-diagonal stored in LU.
//   Writes the solution into y.
// ─────────────────────────────────────────────────────────────
template <typename LUType, typename BType, typename YType>
void forward_sub(const LUType* LU, const int* piv, const BType* b, YType* y, int n) {
    // Apply permutation
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];
    // Forward sweep
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            y[i] -= M(LU, i, j, n) * y[j];
}

// Back substitution: solve U x = y
//   Writes the solution into x.
template <typename LUType, typename YType, typename XType>
void back_sub(const LUType* LU, const YType* y, XType* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = y[i];
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            x[i] -= M(LU, i, j, n) * x[j];
        x[i] /= M(LU, i, i, n);
    }
}

// ─────────────────────────────────────────────────────────────
// Public solve: x = A^{-1} b  using pre-factored LU
//   Writes the solution into x.
// ─────────────────────────────────────────────────────────────
template <typename LUType, typename BType, typename XType>
void lu_solve(const LUType* LU, const int* piv, const BType* b, XType* x, int n) {
    XType* y = new XType[n];
    forward_sub(LU, piv, b, y, n);
    back_sub(LU, y, x, n);
    delete[] y;
}

// ─────────────────────────────────────────────────────────────
// Reconstruct P, L, U from packed LU and pivot vector
//   Caller must free_matrix() P, L, U after use.
// ─────────────────────────────────────────────────────────────
template <typename LUType, typename PType, typename LType, typename UType>
void extract_PLU(const LUType* LU_packed, const int* piv,
                 PType*& P, LType*& L, UType*& U, int n) {
    P = new PType[n * n]();
    L = new LType[n * n]();
    U = new UType[n * n]();

    // Permutation: piv[i] = original row at position i
    for (int i = 0; i < n; ++i) M(P, i, piv[i], n) = static_cast<PType>(1.0);

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
template <typename T>
bool lu_factorize(T* A, int* piv, int n) {
    return lu_factorize_core(A, piv, n);
}

// ─────────────────────────────────────────────────────────────
// Run a complete benchmark for one matrix
// ─────────────────────────────────────────────────────────────
void benchmark(const std::string& label,
               const flx::floatx<5, 2>* A_orig,
               int n,
               std::function<bool(flx::floatx<5, 2>*, int*, int)> lu_fn) {

    // ── Build known RHS from x_true = [1, 2, …, n] ──────────
    flx::floatx<5, 2>* x_true = new flx::floatx<5, 2>[n];
    for (int i = 0; i < n; ++i) x_true[i] = static_cast<flx::floatx<5, 2>>(i + 1);

    flx::floatx<5, 2>* b = new flx::floatx<5, 2>[n];
    mat_vec(A_orig, x_true, b, n);

    // ── Factorize (timed) ────────────────────────────────────
    flx::floatx<5, 2>* LU  = new flx::floatx<5, 2>[n * n]();
    mat_copy(A_orig, LU, n);
    int*    piv = new int[n];

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = lu_fn(LU, piv, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        std::cout << label << " : factorization FAILED (singular matrix)\n";
        delete[] x_true; delete[] b; free_matrix(LU); delete[] piv;
        return;
    }

    // ── Solve ────────────────────────────────────────────────
    flx::floatx<5, 2>* x_computed = new flx::floatx<5, 2>[n];
    lu_solve(LU, piv, b, x_computed, n);

    // ── Accuracy metrics ─────────────────────────────────────
    // 1) Relative residual  ||Ax - b||_2 / ||b||_2
    flx::floatx<5, 2>* Ax  = new flx::floatx<5, 2>[n];
    mat_vec(A_orig, x_computed, Ax, n);
    flx::floatx<5, 2>* res = new flx::floatx<5, 2>[n];
    vec_sub(Ax, b, res, n);
    double rel_residual = static_cast<double>(norm2(res, n) / norm2(b, n));

    // 2) Relative error  ||x - x_true||_2 / ||x_true||_2
    flx::floatx<5, 2>* err = new flx::floatx<5, 2>[n];
    vec_sub(x_computed, x_true, err, n);
    double  rel_error = static_cast<double>(norm2(err, n) / norm2(x_true, n));

    // 3) Backward error  ||PA - LU||_F / ||A||_F
    flx::floatx<5, 2> *P_mat = nullptr, *L_mat = nullptr, *U_mat = nullptr;
    extract_PLU(LU, piv, P_mat, L_mat, U_mat, n);

    double* PA   = new double[n * n]();
    double* LU_m = new double[n * n]();
    double* diff = new double[n * n]();
    mat_mul(P_mat, A_orig, PA,   n);
    mat_mul(L_mat, U_mat,  LU_m, n);
    mat_sub(PA, LU_m, diff, n);
    double back_err = static_cast<double>(normF(diff, n) / normF(A_orig, n));

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
    const double KAPPA = 1e4;

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

    float* A = new float[N * N]();
    make_matrix_with_cond(A, N, KAPPA, rng);

    float* x_true = new float[N];
    for (int i = 0; i < N; ++i) x_true[i] = static_cast<flx::floatx<5, 2>>(1.0);

    float* b = new float[N];
    mat_vec(A, x_true, b, N);

    float* LU  = new float[N * N]();
    mat_copy(A, LU, N);
    int*    piv = new int[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    lu_factorize(LU, piv, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double* x_comp = new double[N];
    lu_solve(LU, piv, b, x_comp, N);

    PROMISE_CHECK_ARRAY(x_comp, N);
    flx::floatx<5, 2>* Ax     = new flx::floatx<5, 2>[N];
    mat_vec(A, x_comp, Ax, N);
    flx::floatx<5, 2>* res    = new flx::floatx<5, 2>[N];
    vec_sub(Ax, b, res, N);
    double rel_res = static_cast<double>(norm2(res, N) / norm2(b, N));


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