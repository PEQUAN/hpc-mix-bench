#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>

struct CSRMatrix {
    int n;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

bool compare_by_column(const std::pair<int, double>& a, const std::pair<int, double>& b) {
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

    std::vector<std::vector<std::pair<int, double>>> temp(n);
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
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

std::vector<double> matvec(const CSRMatrix& A, const std::vector<double>& x) {
    if (A.n != static_cast<int>(x.size())) {
        throw std::runtime_error("Matrix-vector dimension mismatch");
    }
    std::vector<double> y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
    return y;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimension mismatch in dot product");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> axpy(double alpha, const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Vector dimension mismatch in axpy");
    }
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::fma(alpha, x[i], y[i]);
    }
    return result;
}

double norm(const std::vector<double>& v) {
    double d = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return std::sqrt(d);
}

std::vector<double> solve_hessenberg(const std::vector<std::vector<double>>& H, double beta, int m) {
    std::vector<double> y(m, 0.0);
    std::vector<double> rhs(m + 1, 0.0);
    rhs[0] = beta;
    std::vector<std::vector<double>> R = H;
    
    const double eps = std::numeric_limits<double>::epsilon();
    
    for (int i = 0; i < m; ++i) {
        if (i + 1 >= static_cast<int>(H.size())) {
            std::cerr << "Hessenberg matrix size mismatch at i=" << i << std::endl;
            break;
        }
        double hii = R[i][i];
        double hi1i = R[i + 1][i];
        double r = std::hypot(hii, hi1i);
        
        if (r < eps) {
            continue;
        }
        
        double c = hii / r;
        double s = -hi1i / r;
        
        for (int j = i; j < m; ++j) {
            double t1 = R[i][j];
            double t2 = R[i + 1][j];
            R[i][j] = std::fma(c, t1, -s * t2);
            R[i + 1][j] = std::fma(s, t1, c * t2);
        }
        
        double t1 = rhs[i];
        double t2 = rhs[i + 1];
        rhs[i] = std::fma(c, t1, -s * t2);
        rhs[i + 1] = std::fma(s, t1, c * t2);
    }
    
    for (int i = m - 1; i >= 0; --i) {
        if (std::abs(R[i][i]) < eps) {
            y[i] = 0.0;
            continue;
        }
        y[i] = rhs[i];
        for (int j = i + 1; j < m; ++j) {
            y[i] -= R[i][j] * y[j];
        }
        y[i] /= R[i][i];
    }
    return y;
}

// Define GMRESResult before gmres
struct GMRESResult {
    std::vector<double> x;
    double residual;
    int iterations;
    bool converged;
};

GMRESResult gmres(const CSRMatrix& A, const std::vector<double>& b, int max_iter = 1000, double tol = 1e-6) {
    if (A.n != static_cast<int>(b.size())) {
        throw std::runtime_error("Matrix-RHS dimension mismatch");
    }
    
    const double eps = std::numeric_limits<double>::epsilon();
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;
    double beta = norm(r);
    
    if (beta < eps) {
        return {x, beta, 0, true};
    }
    
    double tol_abs = tol * beta;
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> H;
    V.reserve(max_iter + 1);
    H.reserve(max_iter + 1);
    
    V.push_back(axpy(1.0 / beta, r, std::vector<double>(n, 0.0)));
    int k;
    
    for (k = 0; k < max_iter; ++k) {
        // std::cout << "Iteration " << k << ": V.size() = " << V.size() << std::endl;
        
        for (int i = 0; i <= k; ++i) {
            if (i >= static_cast<int>(H.size())) {
                H.push_back(std::vector<double>(k, 0.0));
            }
            while (static_cast<int>(H[i].size()) <= k) {
                H[i].push_back(0.0);
            }
        }
        
        std::vector<double> w = matvec(A, V[k]);
        for (int i = 0; i <= k; ++i) {
            double h_ik = dot(V[i], w);
            H[i][k] = h_ik;
            w = axpy(-h_ik, V[i], w);
        }
        
        double h_next = norm(w);
        H.push_back(std::vector<double>(k + 1, 0.0));
        H[k + 1][k] = h_next;
        
        if (h_next < eps * beta) {
            std::cout << "Breakdown at iteration " << k << std::endl;
            break;
        }
        
        V.push_back(axpy(1.0 / h_next, w, std::vector<double>(n, 0.0)));
        
        std::vector<double> y = solve_hessenberg(H, beta, k + 1);
        x = std::vector<double>(n, 0.0);
        for (int j = 0; j <= k; ++j) {
            if (j >= static_cast<int>(y.size()) || j >= static_cast<int>(V.size())) {
                // std::cerr << "Index error: j=" << j << ", y.size()=" << y.size() 
                //           << ", V.size()=" << V.size() << std::endl;
                break;
            }
            x = axpy(y[j], V[j], x);
        }
        
        r = axpy(-1.0, matvec(A, x), b);
        double r_norm = norm(r);
        
        if (r_norm < tol_abs) {
            std::cout << "Converged at iteration " << k + 1 << std::endl;
            return {x, r_norm, k + 1, true};
        }
    }
    
    std::cout << "Max iterations reached: " << k << std::endl;
    return {x, norm(r), k + 1, false};
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

void write_solution(const std::vector<double>& x, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << ". Check permissions or path." << std::endl;
        return;
    }
    file << "x\n";
    for (double val : x) {
        file << val << "\n";
    }
    file.close();
}

int main() {
    try {
        std::string filename = "../data/suitesparse/1138_bus.mtx";
        CSRMatrix A = read_mtx_file(filename);
        if (A.n == 0) return 1;

        std::vector<double> b = generate_rhs(A.n);

        auto start = std::chrono::high_resolution_clock::now();
        GMRESResult result = gmres(A, b, A.n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;

        write_solution(result.x, "../results/gmres2/gmres_solution.csv");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}