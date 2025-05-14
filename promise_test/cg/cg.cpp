
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <utility> 

struct CSRMatrix {
    int n;          
    int nnz;       
    __PROMISE__* values; 
    int* col_indices; 
    int* row_ptr;  

    CSRMatrix() : n(0), nnz(0), values(nullptr), col_indices(nullptr), row_ptr(nullptr) {}

    ~CSRMatrix() {
        delete[] values;
        delete[] col_indices;
        delete[] row_ptr;
    }
};

bool compare_by_column(const std::pair<int, __PROMISE__>& a, const std::pair<int, __PROMISE__>& b) {
    return a.first < b.first;
}

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return A;
    }

    std::string line;
    while (getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    // Temporary storage for entries
    struct RowEntry {
        std::pair<int, __PROMISE__>* entries;
        int size;
        int capacity;
    };
    RowEntry* temp = new RowEntry[n];
    for (int i = 0; i < n; ++i) {
        temp[i].size = 0;
        temp[i].capacity = 10; // Initial capacity
        temp[i].entries = new std::pair<int, __PROMISE__>[temp[i].capacity];
    }

    // Read non-zeros
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            for (int i = 0; i < n; ++i) delete[] temp[i].entries;
            delete[] temp;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        __PROMISE__ val;
        ss >> i >> j >> val;
        i--; j--; // Convert to 0-based indexing
        // Add to row i
        if (temp[i].size == temp[i].capacity) {
            temp[i].capacity *= 2;
            std::pair<int, __PROMISE__>* new_entries = new std::pair<int, __PROMISE__>[temp[i].capacity];
            for (int p = 0; p < temp[i].size; ++p) {
                new_entries[p] = temp[i].entries[p];
            }
            delete[] temp[i].entries;
            temp[i].entries = new_entries;
        }
        temp[i].entries[temp[i].size++] = {j, val};
        // Add to row j for symmetry (if off-diagonal)
        if (i != j) {
            if (temp[j].size == temp[j].capacity) {
                temp[j].capacity *= 2;
                std::pair<int, __PROMISE__>* new_entries = new std::pair<int, __PROMISE__>[temp[j].capacity];
                for (int p = 0; p < temp[j].size; ++p) {
                    new_entries[p] = temp[j].entries[p];
                }
                delete[] temp[j].entries;
                temp[j].entries = new_entries;
            }
            temp[j].entries[temp[j].size++] = {i, val};
        }
    }

    // Compute row_ptr and nnz
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp[i].size;
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size;
    }

    // Allocate values and col_indices
    A.values = new __PROMISE__[A.nnz];
    A.col_indices = new int[A.nnz];
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i].entries, temp[i].entries + temp[i].size, compare_by_column);
        for (int p = 0; p < temp[i].size; ++p) {
            A.col_indices[pos] = temp[i].entries[p].first;
            A.values[pos] = temp[i].entries[p].second;
            pos++;
        }
    }

    // Clean up temp
    for (int i = 0; i < n; ++i) {
        delete[] temp[i].entries;
    }
    delete[] temp;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

__PROMISE__* matvec(const CSRMatrix& A, const __PROMISE__* x, int n) {
    __PR_5__* y = new __PR_5__[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

__PROMISE__* axpy(__PROMISE__ alpha, const __PROMISE__* x, const __PROMISE__* y, int n) {
    __PR_4__* result = new __PR_4__[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

__PROMISE__ dot(const __PROMISE__* a, const __PROMISE__* b, int n) {
    __PROMISE__ sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__PROMISE__ norm(const __PROMISE__* v, int n) {
    return sqrt(dot(v, v, n));
}

struct CGResult {
    __PROMISE__* x;
    int n; // Size of x
    __PROMISE__ residual;
    int iterations;

    ~CGResult() {
        delete[] x;
    }
};

CGResult conjugate_gradient(const CSRMatrix& A, const __PROMISE__* b, int max_iter = 1000, __PROMISE__ tol = 1e-6) {
    int n = A.n;
    CGResult result;
    result.n = n;
    result.x = new __PR_3__[n]();
    __PR_1__* r = new __PR_1__[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    __PR_2__* p = new __PR_2__[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    __PROMISE__ rtr = dot(r, r, n);
    __PROMISE__ tol2 = tol * tol * dot(b, b, n);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        __PROMISE__* Ap = matvec(A, p, n);
        __PROMISE__ alpha = rtr / dot(p, Ap, n);
        __PROMISE__* x_new = axpy(alpha, p, result.x, n);
        delete[] result.x;
        result.x = x_new;
        __PROMISE__* r_new = axpy(-alpha, Ap, r, n);
        delete[] r;
        r = r_new;
        __PROMISE__ rtr_new = dot(r, r, n);
        __PROMISE__ beta = rtr_new / rtr;
        __PROMISE__* p_new = axpy(beta, p, r, n);
        delete[] p;
        p = p_new;
        delete[] Ap;
        rtr = rtr_new;
    }

    result.residual = norm(r, n);
    result.iterations = k;

    delete[] r;
    delete[] p;

    return result;
}

__PROMISE__* generate_rhs(int n) {
    __PROMISE__* b = new __PROMISE__[n];

    std::mt19937 gen(1000);
    std::uniform_real_distribution<> dis(0, 10);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}


int main() {
    std::string filename = "rdb5000.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    __PROMISE__* b = generate_rhs(A.n);

    CGResult result = conjugate_gradient(A, b, A.n);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    __PROMISE__* Ax = matvec(A, result.x, A.n);
    __PROMISE__* verify_vec = axpy(-1.0, Ax, b, A.n);
    __PROMISE__ verify_residual = norm(verify_vec, A.n);
    PROMISE_CHECK_ARRAY(verify_vec, A.n);

    delete[] b;
    delete[] Ax;
    delete[] verify_vec;

    return 0;
}