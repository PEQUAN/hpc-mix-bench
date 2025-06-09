#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
} CSRMatrix;

double* allocate_vector(int n) {
    double* v = (double*)malloc(n * sizeof(double));
    if (!v) {
        fprintf(stderr, "Error: Memory allocation failed for vector\n");
        exit(1);
    }
    return v;
}

int* allocate_int_vector(int n) {
    int* v = (int*)malloc(n * sizeof(int));
    if (!v) {
        fprintf(stderr, "Error: Memory allocation failed for int vector\n");
        exit(1);
    }
    return v;
}

void deallocate_vector(double* v) {
    if (v) free(v);
}

void deallocate_int_vector(int* v) {
    if (v) free(v);
}

CSRMatrix allocate_csr_matrix(int n, int nnz) {
    CSRMatrix A = {0, NULL, NULL, NULL, 0};
    A.n = n;
    A.nnz = nnz;
    A.values = (double*)malloc(nnz * sizeof(double));
    A.col_indices = (int*)malloc(nnz * sizeof(int));
    A.row_ptr = (int*)malloc((n + 1) * sizeof(int));
    if (!A.values || !A.col_indices || !A.row_ptr) {
        fprintf(stderr, "Error: Memory allocation failed for CSR matrix\n");
        deallocate_vector(A.values);
        deallocate_int_vector(A.col_indices);
        deallocate_int_vector(A.row_ptr);
        exit(1);
    }
    return A;
}

void deallocate_csr_matrix(CSRMatrix* A) {
    if (A) {
        deallocate_vector(A->values);
        deallocate_int_vector(A->col_indices);
        deallocate_int_vector(A->row_ptr);
        A->n = 0;
        A->nnz = 0;
    }
}

CSRMatrix read_mtx_file(const char* filename) {
    CSRMatrix A = {0, NULL, NULL, NULL, 0};
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s\n", filename);
        return A;
    }

    char line[256];
    while (fgets(line, sizeof(line), file) && line[0] == '%') {}

    int n, m, nz;
    if (sscanf(line, "%d %d %d", &n, &m, &nz) != 3) {
        fprintf(stderr, "Error: Invalid matrix header\n");
        fclose(file);
        return A;
    }
    if (n != m) {
        fprintf(stderr, "Error: Matrix must be square\n");
        fclose(file);
        return A;
    }
    if (n <= 0 || nz < 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions or non-zeros\n");
        fclose(file);
        return A;
    }

    typedef struct { int row, col; double val; } Entry;
    Entry* temp = (Entry*)malloc(2 * nz * sizeof(Entry)); // Account for symmetric entries
    int* counts = (int*)calloc(n, sizeof(int));
    if (!temp || !counts) {
        fprintf(stderr, "Error: Memory allocation failed in read_mtx_file\n");
        deallocate_vector((double*)temp);
        deallocate_int_vector(counts);
        fclose(file);
        return A;
    }

    int entry_count = 0;
    for (int k = 0; k < nz; ++k) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Unexpected end of file at line %d\n", k + 2);
            deallocate_vector((double*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        int i, j;
        double val;
        if (sscanf(line, "%d %d %lf", &i, &j, &val) != 3) {
            fprintf(stderr, "Error: Invalid matrix entry at line %d\n", k + 2);
            deallocate_vector((double*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        i--; j--;
        if (i < 0 || i >= n || j < 0 || j >= n) {
            fprintf(stderr, "Error: Invalid indices at line %d: (%d,%d)\n", k + 2, i + 1, j + 1);
            deallocate_vector((double*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        if (entry_count + (i != j ? 2 : 1) > 2 * nz) {
            fprintf(stderr, "Error: Too many non-zero entries\n");
            deallocate_vector((double*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        temp[entry_count++] = (Entry){i, j, val};
        counts[i]++;
        if (i != j) {
            temp[entry_count++] = (Entry){j, i, val};
            counts[j]++;
        }
    }

    int total_nnz = entry_count;
    A = allocate_csr_matrix(n, total_nnz);
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + counts[i];
    }

    int* current = (int*)calloc(n, sizeof(int));
    if (!current) {
        fprintf(stderr, "Error: Memory allocation failed for current array\n");
        deallocate_csr_matrix(&A);
        deallocate_vector((double*)temp);
        deallocate_int_vector(counts);
        fclose(file);
        return A;
    }

    for (int k = 0; k < entry_count; ++k) {
        int i = temp[k].row;
        int idx = A.row_ptr[i] + current[i]++;
        if (idx >= A.nnz) {
            fprintf(stderr, "Error: Index out of bounds in CSR construction\n");
            deallocate_csr_matrix(&A);
            deallocate_vector((double*)temp);
            deallocate_int_vector(counts);
            deallocate_int_vector(current);
            fclose(file);
            return A;
        }
        A.values[idx] = temp[k].val;
        A.col_indices[idx] = temp[k].col;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            for (int k = j + 1; k < A.row_ptr[i + 1]; ++k) {
                if (A.col_indices[j] > A.col_indices[k]) {
                    int t = A.col_indices[j];
                    A.col_indices[j] = A.col_indices[k];
                    A.col_indices[k] = t;
                    double tv = A.values[j];
                    A.values[j] = A.values[k];
                    A.values[k] = tv;
                }
            }
        }
    }

    int non_positive_diagonal = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_indices[j];
            double val = A.values[j];
            if (col == i && val <= 0.0) {
                non_positive_diagonal++;
            }
            int found = 0;
            for (int k = A.row_ptr[col]; k < A.row_ptr[col + 1]; ++k) {
                if (A.col_indices[k] == i) {
                    if (fabs(A.values[k] - val) > 1e-10) {
                        fprintf(stderr, "Warning: Matrix may not be symmetric at (%d,%d) vs (%d,%d): %e vs %e\n",
                                i, col, col, i, val, A.values[k]);
                    }
                    found = 1;
                    break;
                }
            }
            if (!found && fabs(val) > 1e-10) {
                fprintf(stderr, "Warning: Matrix may not be symmetric: missing (%d,%d)\n", col, i);
            }
        }
    }
    if (non_positive_diagonal > 0) {
        fprintf(stderr, "Warning: Found %d non-positive diagonal elements\n", non_positive_diagonal);
    }

    deallocate_vector((double*)temp);
    deallocate_int_vector(counts);
    deallocate_int_vector(current);
    fclose(file);
    printf("Loaded matrix: %d x %d with %d non-zeros\n", n, n, A.nnz);
    return A;
}

void matvec(const CSRMatrix* A, const double* x, double* y) {
    if (!A || !x || !y) {
        fprintf(stderr, "Error: Null pointer in matvec\n");
        return;
    }
    for (int i = 0; i < A->n; ++i) {
        double sum = 0.0, c = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; ++j) {
            if (j >= A->nnz || A->col_indices[j] >= A->n) {
                fprintf(stderr, "Error: Invalid index in matvec at row %d, index %d\n", i, j);
                return;
            }
            double y_val = A->values[j] * x[A->col_indices[j]] - c;
            double t = sum + y_val;
            c = (t - sum) - y_val;
            sum = t;
        }
        y[i] = sum;
    }
}

void axpy(int n, double alpha, const double* x, const double* y, double* result) {
    if (!x || !y || !result) {
        fprintf(stderr, "Error: Null pointer in axpy\n");
        return;
    }
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

double dot(int n, const double* a, const double* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: Null pointer in dot\n");
        return 0.0;
    }
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double y = a[i] * b[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

double norm(int n, const double* v) {
    if (!v) {
        fprintf(stderr, "Error: Null pointer in norm\n");
        return 1e-16;
    }
    double nrm = sqrt(fabs(dot(n, v, v)));
    return (nrm < 1e-16) ? 1e-16 : nrm;
}

void compute_diagonal_preconditioner(const CSRMatrix* A, double* M) {
    if (!A || !M) {
        fprintf(stderr, "Error: Null pointer in compute_diagonal_preconditioner\n");
        return;
    }
    for (int i = 0; i < A->n; ++i) {
        M[i] = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; ++j) {
            if (j >= A->nnz || A->col_indices[j] >= A->n) {
                fprintf(stderr, "Error: Invalid index in preconditioner at row %d, index %d\n", i, j);
                return;
            }
            if (A->col_indices[j] == i) {
                M[i] = A->values[j];
                break;
            }
        }
        if (fabs(M[i]) < 1e-10) {
            M[i] = 1.0;
        } else {
            M[i] = 1.0 / M[i];
        }
    }
}

void apply_preconditioner(int n, const double* M, const double* r, double* z) {
    if (!M || !r || !z) {
        fprintf(stderr, "Error: Null pointer in apply_preconditioner\n");
        return;
    }
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

typedef struct {
    double* x;
    double residual;
    int iterations;
    int status; // 0: success, 1: max iterations, 2: stagnation, 3: divergence
} CGResult;

CGResult conjugate_gradient(const CSRMatrix* A, const double* b, int max_iter, double tol) {
    CGResult result = {NULL, 0.0, 0, 0};
    if (!A || !b || A->n <= 0 || max_iter <= 0 || tol <= 0.0) {
        fprintf(stderr, "Error: Invalid input to conjugate_gradient\n");
        return result;
    }

    int n = A->n;
    result.x = allocate_vector(n);
    double* r = allocate_vector(n);
    double* p = allocate_vector(n);
    double* z = allocate_vector(n);
    double* Ap = allocate_vector(n);
    double* M = allocate_vector(n);

    if (!result.x || !r || !p || !z || !Ap || !M) {
        fprintf(stderr, "Error: Memory allocation failed in conjugate_gradient\n");
        deallocate_vector(result.x);
        deallocate_vector(r);
        deallocate_vector(p);
        deallocate_vector(z);
        deallocate_vector(Ap);
        deallocate_vector(M);
        return result;
    }

    for (int i = 0; i < n; ++i) result.x[i] = 0.0;
    memcpy(r, b, n * sizeof(double));
    double b_norm = norm(n, b);
    if (b_norm < 1e-16) b_norm = 1e-16;

    compute_diagonal_preconditioner(A, M);
    apply_preconditioner(n, M, r, z);
    memcpy(p, z, n * sizeof(double));
    double rz_old = dot(n, r, z);
    double tol2 = tol * tol * b_norm * b_norm;

    double initial_rz = rz_old;
    double prev_rz = rz_old;
    int stagnant_count = 0;
    const double eps = 1e-16;

    printf("Initial residual norm: %.2e\n", norm(n, r) / b_norm);

    for (int k = 0; k < max_iter; ++k) {
        matvec(A, p, Ap);
        double pAp = dot(n, p, Ap);
        if (fabs(pAp) < eps) {
            result.status = 2;
            fprintf(stderr, "Warning: pAp too small at iteration %d: %e\n", k, pAp);
            break;
        }
        double alpha = rz_old / pAp;
        axpy(n, alpha, p, result.x, result.x);
        axpy(n, -alpha, Ap, r, r);
        apply_preconditioner(n, M, r, z);
        double rz_new = dot(n, r, z);
        double rel_residual = norm(n, r) / b_norm;

        printf("Iteration %d: residual norm = %.2e, alpha = %.2e, pAp = %.2e\n", k + 1, rel_residual, alpha, pAp);

        if (rz_new > 1e10 * initial_rz || isnan(rz_new) || isinf(rz_new)) {
            result.status = 3;
            fprintf(stderr, "Warning: Divergence detected at iteration %d: rz = %e\n", k + 1, rz_new);
            break;
        }

        if (fabs(rz_new - prev_rz) < eps * rz_new && k > 0) {
            stagnant_count++;
            if (stagnant_count > 5) {
                result.status = 2;
                fprintf(stderr, "Warning: Stagnation detected at iteration %d\n", k + 1);
                break;
            }
        } else {
            stagnant_count = 0;
        }

        if (rel_residual < tol) {
            result.status = 0;
            break;
        }

        double beta = rz_new / rz_old;
        axpy(n, beta, p, z, p);
        rz_old = rz_new;
        prev_rz = rz_new;
        result.iterations = k + 1;
    }

    if (result.iterations >= max_iter && result.status == 0) {
        result.status = 1;
    }

    result.residual = norm(n, r) / b_norm;
    deallocate_vector(r);
    deallocate_vector(p);
    deallocate_vector(z);
    deallocate_vector(Ap);
    deallocate_vector(M);
    return result;
}

double* generate_rhs(int n, unsigned int seed) {
    if (n <= 0) {
        fprintf(stderr, "Error: Invalid dimension in generate_rhs\n");
        return NULL;
    }
    double* b = allocate_vector(n);
    srand(seed);
    for (int i = 0; i < n; ++i) {
        b[i] = (double)rand() / RAND_MAX * 10.0;
    }
    return b;
}

void write_solution(const double* x, int n, const char* filename) {
    if (!x || n <= 0 || !filename) {
        fprintf(stderr, "Error: Invalid input to write_solution\n");
        return;
    }
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        return;
    }
    fprintf(file, "x\n");
    for (int i = 0; i < n; ++i) {
        fprintf(file, "%.16e\n", x[i]);
    }
    fclose(file);
}

int main() {
    const char* filename = "../data/suitesparse/1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        fprintf(stderr, "Error: Failed to load matrix\n");
        return 1;
    }

    double* b = generate_rhs(A.n, 32);
    if (!b) {
        fprintf(stderr, "Error: Failed to generate RHS\n");
        deallocate_csr_matrix(&A);
        return 1;
    }

    clock_t start = clock();
    CGResult result = conjugate_gradient(&A, b, A.n, 1e-6);
    clock_t end = clock();
    double duration = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    printf("Matrix size: %d x %d\n", A.n, A.n);
    printf("Training time: %.2f ms\n", duration);
    printf("Final residual: %.2e\n", result.residual);
    printf("Iterations to converge: %d\n", result.iterations);
    switch (result.status) {
        case 0: printf("Status: Converged\n"); break;
        case 1: printf("Status: Max iterations reached\n"); break;
        case 2: printf("Status: Stagnation detected\n"); break;
        case 3: printf("Status: Divergence detected\n"); break;
        default: printf("Status: Unknown\n"); break;
    }

    double* Ax = allocate_vector(A.n);
    double* temp = allocate_vector(A.n);
    if (Ax && temp) {
        matvec(&A, result.x, Ax);
        axpy(A.n, -1.0, Ax, b, temp);
        double verify_residual = norm(A.n, temp) / (norm(A.n, b) > 1e-16 ? norm(A.n, b) : 1.0);
        printf("Verification residual: %.2e\n", verify_residual);
    } else {
        fprintf(stderr, "Error: Memory allocation failed for verification\n");
    }

    write_solution(result.x, A.n, "../results/cg/cg_solution.csv");

    deallocate_csr_matrix(&A);
    deallocate_vector(b);
    deallocate_vector(result.x);
    deallocate_vector(Ax);
    deallocate_vector(temp);
    return 0;
}