SORResult sor(const CSRMatrix& A, const double* b, double omega,
              int max_iter = 5000, double tol = 1e-6) {
    if (omega <= 0.0 || omega >= 2.0) {
        std::cerr << "Error: Omega must be in (0,2)\n";
        double* x = new double[A.n]();
        return {x, 0.0, 0, false};
    }

    const int n = A.n;
    double* x = new double[n]();                
    const double eps = std::numeric_limits<double>::epsilon();

    /* ---------- 1. extract diagonal ---------- */
    double* diag = get_diagonal(A);
    bool use_jacobi = false;
    std::vector<int> zero_rows;
    for (int i = 0; i < n; ++i)
        if (std::abs(diag[i]) < eps) zero_rows.push_back(i);

    if (!zero_rows.empty()) {
        use_jacobi = true;
        std::cout << "Warning: " << zero_rows.size()
                  << " rows have zero diagonal (first few: ";
        for (size_t k = 0; k < std::min<size_t>(5, zero_rows.size()); ++k)
            std::cout << zero_rows[k] << ' ';
        std::cout << "). Switching to Jacobi iteration.\n";
    }

    /* ---------- 2. tolerance ---------- */
    double b_norm = norm(b, n);
    double tol_abs = tol * (b_norm > 1e-12 ? b_norm : 1.0);

    /* ---------- 3. iteration ---------- */
    double* r = new double[n];                 // residual vector (re-used)
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {

        if (use_jacobi) {                     // ---- JACOBI ----
            // x_new[i] = (b[i] - sum_{j≠i} a_ij x_old[j]) / a_ii
            // when a_ii==0 we simply set x_new[i] = b[i] (least-harmful)
            for (int i = 0; i < n; ++i) {
                double sum_off = 0.0;
                double d = diag[i];
                for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
                    int col = A.col_indices[j];
                    if (col != i) sum_off += A.values[j] * x[col];
                }
                if (std::abs(d) > eps)
                    r[i] = (b[i] - sum_off) / d;
                else
                    r[i] = b[i];               // fallback when diagonal is zero
            }
            std::swap(x, r);                  // x ← r  (no extra allocation)
        }
        else {                                 // ---- SOR (Gauss-Seidel + omega) ----
            for (int i = 0; i < n; ++i) {
                double sum_off = 0.0;
                double d = 0.0;
                for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
                    int col = A.col_indices[j];
                    if (col == i) d = A.values[j];
                    else          sum_off += A.values[j] * x[col];
                }
                // d is guaranteed non-zero because we checked above
                x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sum_off) / d;
            }
        }

        /* ---------- residual check ---------- */
        double* Ax = matvec(A, x);
        double res = 0.0;
        for (int i = 0; i < n; ++i) {
            double ri = b[i] - Ax[i];
            res += ri * ri;
        }
        delete[] Ax;
        res = std::sqrt(res);

        if (res < tol_abs) {
            delete[] diag; delete[] r;
            std::cout << "Converged at iteration " << iter + 1 << '\n';
            return {x, res, iter + 1, true};
        }
    }

    /* ---------- max-iter reached ---------- */
    delete[] diag; delete[] r;
    std::cout << "Max iterations reached: " << iter << '\n';
    return {x, norm(matvec(A, x), n), iter, false};
}