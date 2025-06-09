#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct {
    int n;
    __PROMISE__* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
} CSRMatrix;

__PROMISE__* allocate_vector(int n) {
    __PROMISE__* v = (__PROMISE__*)malloc(n * sizeof(__PROMISE__));
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

void deallocate_vector(__PROMISE__* v) {
    if (v) free(v);
}

void deallocate_int_vector(int* v) {
    if (v) free(v);
}

CSRMatrix allocate_csr_matrix(int n, int nnz) {
    CSRMatrix A = {0, NULL, NULL, NULL, 0};
    A.n = n;
    A.nnz = nnz;
    A.values = (__PROMISE__*)malloc(nnz * sizeof(__PROMISE__));
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

    typedef struct { int row, col; __PROMISE__ val; } Entry;
    Entry* temp = (Entry*)malloc(2 * nz * sizeof(Entry)); // Account for symmetric entries
    int* counts = (int*)calloc(n, sizeof(int));
    if (!temp || !counts) {
        fprintf(stderr, "Error: Memory allocation failed in read_mtx_file\n");
        deallocate_vector((__PROMISE__*)temp);
        deallocate_int_vector(counts);
        fclose(file);
        return A;
    }

    int entry_count = 0;
    for (int k = 0; k < nz; ++k) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Unexpected end of file at line %d\n", k + 2);
            deallocate_vector((__PROMISE__*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        int i, j;
        __PROMISE__ val;
        if (sscanf(line, "%d %d %lf", &i, &j, &val) != 3) {
            fprintf(stderr, "Error: Invalid matrix entry at line %d\n", k + 2);
            deallocate_vector((__PROMISE__*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        i--; j--;
        if (i < 0 || i >= n || j < 0 || j >= n) {
            fprintf(stderr, "Error: Invalid indices at line %d: (%d,%d)\n", k + 2, i + 1, j + 1);
            deallocate_vector((__PROMISE__*)temp);
            deallocate_int_vector(counts);
            fclose(file);
            return A;
        }
        if (entry_count + (i != j ? 2 : 1) > 2 * nz) {
            fprintf(stderr, "Error: Too many non-zero entries\n");
            deallocate_vector((__PROMISE__*)temp);
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
        deallocate_vector((__PROMISE__*)temp);
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
            deallocate_vector((__PROMISE__*)temp);
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
                    __PROMISE__ tv = A.values[j];
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
            __PROMISE__ val = A.values[j];
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

    deallocate_vector((__PROMISE__*)temp);
    deallocate_int_vector(counts);
    deallocate_int_vector(current);
    fclose(file);
    printf("Loaded matrix: %d x %d with %d non-zeros\n", n, n, A.nnz);
    return A;
}

void matvec(const CSRMatrix* A, const __PROMISE__* x, __PROMISE__* y) {
    if (!A || !x || !y) {
        fprintf(stderr, "Error: Null pointer in matvec\n");
        return;
    }
    for (int i = 0; i < A->n; ++i) {
        __PROMISE__ sum = 0.0, c = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; ++j) {
            if (j >= A->nnz || A->col_indices[j] >= A->n) {
                fprintf(stderr, "Error: Invalid index in matvec at row %d, index %d\n", i, j);
                return;
            }
            __PROMISE__ y_val = A->values[j] * x[A->col_indices[j]] - c;
            __PROMISE__ t = sum + y_val;
            c = (t - sum) - y_val;
            sum = t;
        }
        y[i] = sum;
    }
}

void axpy(int n, __PROMISE__ alpha, const __PROMISE__* x, const __PROMISE__* y, __PROMISE__* result) {
    if (!x || !y || !result) {
        fprintf(stderr, "Error: Null pointer in axpy\n");
        return;
    }
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

__PROMISE__ dot(int n, const __PROMISE__* a, const __PROMISE__* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: Null pointer in dot\n");
        return 0.0;
    }
    __PROMISE__ sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ y = a[i] * b[i] - c;
        __PROMISE__ t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__PROMISE__ norm(int n, const __PROMISE__* v) {
    if (!v) {
        fprintf(stderr, "Error: Null pointer in norm\n");
        return 1e-16;
    }
    __PROMISE__ nrm = sqrt(fabs(dot(n, v, v)));
    double tempp = 1e-16;
    return (nrm < 1e-16) ? tempp : nrm;
}

void compute_diagonal_preconditioner(const CSRMatrix* A, __PROMISE__* M) {
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

void apply_preconditioner(int n, const __PROMISE__* M, const __PROMISE__* r, __PROMISE__* z) {
    if (!M || !r || !z) {
        fprintf(stderr, "Error: Null pointer in apply_preconditioner\n");
        return;
    }
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

typedef struct {
    __PROMISE__* x;
    __PROMISE__ residual;
    int iterations;
    int status; // 0: success, 1: max iterations, 2: stagnation, 3: divergence
} CGResult;

CGResult conjugate_gradient(const CSRMatrix* A, const __PROMISE__* b, int max_iter, __PROMISE__ tol) {
    CGResult result = {NULL, 0.0, 0, 0};
    if (!A || !b || A->n <= 0 || max_iter <= 0 || tol <= 0.0) {
        fprintf(stderr, "Error: Invalid input to conjugate_gradient\n");
        return result;
    }

    int n = A->n;
    result.x = allocate_vector(n);
    __PROMISE__* r = allocate_vector(n);
    __PROMISE__* p = allocate_vector(n);
    __PROMISE__* z = allocate_vector(n);
    __PROMISE__* Ap = allocate_vector(n);
    __PROMISE__* M = allocate_vector(n);

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
    memcpy(r, b, n * sizeof(__PROMISE__));
    __PROMISE__ b_norm = norm(n, b);
    if (b_norm < 1e-16) b_norm = 1e-16;

    compute_diagonal_preconditioner(A, M);
    apply_preconditioner(n, M, r, z);
    memcpy(p, z, n * sizeof(__PROMISE__));
    __PROMISE__ rz_old = dot(n, r, z);
    __PROMISE__ tol2 = tol * tol * b_norm * b_norm;

    __PROMISE__ initial_rz = rz_old;
    __PROMISE__ prev_rz = rz_old;
    int stagnant_count = 0;
    const __PROMISE__ eps = 1e-16;

    printf("Initial residual norm: %.2e\n", norm(n, r) / b_norm);

    for (int k = 0; k < max_iter; ++k) {
        matvec(A, p, Ap);
        __PROMISE__ pAp = dot(n, p, Ap);
        if (fabs(pAp) < eps) {
            result.status = 2;
            fprintf(stderr, "Warning: pAp too small at iteration %d: %e\n", k, pAp);
            break;
        }
        __PROMISE__ alpha = rz_old / pAp;
        axpy(n, alpha, p, result.x, result.x);
        axpy(n, -alpha, Ap, r, r);
        apply_preconditioner(n, M, r, z);
        __PROMISE__ rz_new = dot(n, r, z);
        __PROMISE__ rel_residual = norm(n, r) / b_norm;

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

        __PROMISE__ beta = rz_new / rz_old;
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

__PROMISE__* generate_rhs(int n, unsigned int seed) {
    if (n <= 0) {
        fprintf(stderr, "Error: Invalid dimension in generate_rhs\n");
        return NULL;
    }
    __PROMISE__* b = allocate_vector(n);
    srand(seed);
    for (int i = 0; i < n; ++i) {
        b[i] = (__PROMISE__)rand() / RAND_MAX * 10.0;
    }
    return b;
}



int main() {
    const char* filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        fprintf(stderr, "Error: Failed to load matrix\n");
        return 1;
    }

    __PROMISE__* b = generate_rhs(A.n, 32);
    if (!b) {
        fprintf(stderr, "Error: Failed to generate RHS\n");
        deallocate_csr_matrix(&A);
        return 1;
    }

    CGResult result = conjugate_gradient(&A, b, A.n, 1e-6);

    printf("Matrix size: %d x %d\n", A.n, A.n);
    printf("Final residual: %.2e\n", result.residual);
    printf("Iterations to converge: %d\n", result.iterations);
    switch (result.status) {
        case 0: printf("Status: Converged\n"); break;
        case 1: printf("Status: Max iterations reached\n"); break;
        case 2: printf("Status: Stagnation detected\n"); break;
        case 3: printf("Status: Divergence detected\n"); break;
        default: printf("Status: Unknown\n"); break;
    }

    __PROMISE__* Ax = allocate_vector(A.n);
    

    double* temp = allocate_vector(A.n);
    if (Ax && temp) {
        matvec(&A, result.x, Ax);
        axpy(A.n, -1.0, Ax, b, temp);
        double tempp = 1.0;
        double verify_residual = norm(A.n, temp) / (norm(A.n, b) > 1e-16 ? norm(A.n, b) : tempp);
        printf("Verification residual: %.2e\n", verify_residual);
    } else {
        fprintf(stderr, "Error: Memory allocation failed for verification\n");
    }

    __PROMISE__ solution[A.n];
    for (int i=0; i++; i<A.n){
        solution[i] = result.x[i];
    }
    PROMISE_CHECK_ARRAY(solution, A.n);

    deallocate_csr_matrix(&A);
    deallocate_vector(b);
    deallocate_vector(result.x);
    deallocate_vector(Ax);
    deallocate_vector(temp);
    return 0;
}