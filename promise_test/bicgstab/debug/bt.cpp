#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;
    std::vector<half_float::half> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

bool compare_by_column(const std::pair<int, half_float::half>& a, const std::pair<int, half_float::half>& b) {
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

    std::vector<std::vector<std::pair<int, half_float::half>>> temp(n);
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        half_float::half val;
        ss >> i >> j >> val;
        i--; j--;
        temp[i].push_back({j, val});
        if (i != j) temp[j].push_back({i, val});
    }

    A.row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i].begin(), temp[i].end(), compare_by_column);
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size();
        for (const auto& entry : temp[i]) {
            A.col_indices.push_back(entry.first);
            A.values.push_back(entry.second);
        }
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.values.size() << " non-zeros" << std::endl;
    return A;
}

std::vector<float> matvec(const CSRMatrix& A, const std::vector<float>& x) {
    std::vector<float> y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<float> axpy(float alpha, const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

float norm(const std::vector<float>& v) {
    float d = dot(v, v);
    // if (std::isnan(d) || std::isinf(d)) return -1.0;
    return sqrt(d);
}

struct BiCGSTABResult {
    std::vector<float> x;
    float residual;
    int iterations;
};

BiCGSTABResult bicgstab(const CSRMatrix& A, const std::vector<float>& b, int max_iter = 1000, float tol = 1e-6) {
    int n = A.n;
    std::vector<float> x(n, 0.0);
    std::vector<float> r = b;
    std::vector<float> r_hat = r;
    std::vector<float> p = r;
    std::vector<float> v(n, 0.0);
    float rho = 1.0, alpha = 1.0, omega = 1.0;
    float initial_norm = norm(b);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        return {x, -1.0, 0};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    float tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        float rho_new = dot(r_hat, r);
        if (abs(rho_new) < 1e-10) {
            std::cerr << "Breakdown: rho = " << rho_new << " at iteration " << k << std::endl;
            break;
        }
        float beta = (rho_new / rho) * (alpha / omega);
        p = axpy(beta, axpy(-omega, v, p), r);
        v = matvec(A, p);
        float rhat_v = dot(r_hat, v);
        if (abs(rhat_v) < 1e-10) {
            std::cerr << "Breakdown: r_hat^T v = " << rhat_v << " at iteration " << k << std::endl;
            break;
        }
        alpha = rho_new / rhat_v;
        std::vector<float> s = axpy(-alpha, v, r);
        float s_norm = norm(s);
        if (s_norm < tol_abs) {
            x = axpy(alpha, p, x);
            r = s;
            break;
        }
        std::vector<float> t = matvec(A, s);
        float t_t = dot(t, t);
        if (abs(t_t) < 1e-10) {
            std::cerr << "Breakdown: t^T t = " << t_t << " at iteration " << k << std::endl;
            break;
        }

        omega = dot(t, s) / t_t;
        x = axpy(alpha, p, axpy(omega, s, x));
        r = axpy(-omega, t, s);
        float r_norm = norm(r);
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k + 1 << std::endl;
            break;
        }
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Residual = " << r_norm << std::endl;
        }
        if (r_norm < tol_abs) break;
        rho = rho_new;
    }

    float residual = norm(r);
    return {x, residual, k + 1};
}

std::vector<float> generate_rhs(int n) {
    std::vector<float> b(n);
    std::mt19937 gen(2025);
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const std::vector<half_float::half>& x, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << ". Check permissions or path." << std::endl;
        return;
    }
    file << "x\n";
    for (half_float::half val : x) {
        file << val << "\n";
    }
    file.close();
}

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    std::vector<float> b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    BiCGSTABResult result = bicgstab(A, b, 2 * A.n); // Increase max_iter to 2n
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    std::vector<float> Ax = matvec(A, result.x);
    float verify_residual = norm(axpy(-1.0, Ax, b));
    std::cout << "Verification residual: " << verify_residual << std::endl;
    PROMISE_CHECK_ARRAY(result.x.data(), A.n);
    // write_solution(result.x, "results/bicgstab/bicgstab_solution.csv");

    return 0;
}
