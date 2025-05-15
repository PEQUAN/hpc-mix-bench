#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

const bool USE_FIXED_SEED = true;
const unsigned int FIXED_SEED = 42;

// Read matrix from CSV file
double* read_csv(const std::string& filename, size_t& rows, size_t& cols) {
    std::ifstream file(filename);
    rows = 0;
    cols = 0;
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    // Count rows and columns
    std::string line;
    while (getline(file, line)) {
        ++rows;
        if (cols == 0) {
            std::stringstream ss(line);
            std::string value;
            while (getline(ss, value, ',')) {
                ++cols;
            }
        }
    }
    file.clear();
    file.seekg(0);

    // Allocate matrix
    double* matrix = new double[rows * cols];
    size_t row = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;
        while (getline(ss, value, ',') && col < cols) {
            matrix[row * cols + col] = std::stod(value);
            ++col;
        }
        ++row;
    }

    std::cout << "Loaded matrix with " << rows << " rows and " << cols << " columns" << std::endl;
    file.close();
    return matrix;
}

class RandomizedSVD {
private:
    size_t m; // Rows of input matrix
    size_t n; // Columns of input matrix
    size_t k; // Target rank
    double* A; // Input matrix: m x n
    double* U; // Left singular vectors: m x k
    double* S; // Singular values: k
    double* V; // Right singular vectors: n x k
    double runtime;
    unsigned int seed;
    bool useFixedSeed;

    // Matrix multiplication: C = A * B, where A is m x n, B is n x k, C is m x k
    void matrix_multiply(const double* A, const double* B, double* C, size_t m, size_t n, size_t k) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                C[i * k + j] = 0.0;
                for (size_t p = 0; p < n; ++p) {
                    C[i * k + j] += A[i * n + p] * B[p * k + j];
                }
            }
        }
    }

    // Transpose matrix multiplication: C = A^T * B, where A is m x k, B is m x n, C is k x n
    void transpose_multiply(const double* A, const double* B, double* C, size_t m, size_t k, size_t n) {
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = 0.0;
                for (size_t p = 0; p < m; ++p) {
                    C[i * n + j] += A[p * k + i] * B[p * n + j];
                }
            }
        }
    }

    // QR decomposition using Gram-Schmidt for Y (m x k), returns Q (m x k)
    void qr_decomposition(const double* Y, double* Q, size_t m, size_t k) {
        double* temp = new double[m];
        for (size_t j = 0; j < k; ++j) {
            // Copy column j of Y to temp
            for (size_t i = 0; i < m; ++i) {
                temp[i] = Y[i * k + j];
            }
            // Orthogonalize against previous columns
            for (size_t p = 0; p < j; ++p) {
                double dot = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    dot += temp[i] * Q[i * k + p];
                }
                for (size_t i = 0; i < m; ++i) {
                    temp[i] -= dot * Q[i * k + p];
                }
            }
            // Normalize
            double norm = 0.0;
            for (size_t i = 0; i < m; ++i) {
                norm += temp[i] * temp[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10) {
                for (size_t i = 0; i < m; ++i) {
                    Q[i * k + j] = temp[i] / norm;
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    Q[i * k + j] = 0.0;
                }
            }
        }
        delete[] temp;
    }

    // Simple SVD for small matrix B (k x n) using power method
    void small_svd(const double* B, double* U_B, double* S, double* V, size_t k, size_t n) {
        // Initialize V randomly
        std::mt19937 gen(useFixedSeed ? seed : std::random_device{}());
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < n * k; ++i) {
            V[i] = dist(gen);
        }

        // Power method for top k singular values/vectors
        double* temp = new double[n];
        double* temp2 = new double[k];
        for (size_t j = 0; j < k; ++j) {
            // Normalize current V column
            double norm = 0.0;
            for (size_t i = 0; i < n; ++i) {
                norm += V[i * k + j] * V[i * k + j];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10) {
                for (size_t i = 0; i < n; ++i) {
                    V[i * k + j] /= norm;
                }
            }

            // Power iterations
            for (int iter = 0; iter < 10; ++iter) {
                // temp2 = B * V(:,j)
                for (size_t i = 0; i < k; ++i) {
                    temp2[i] = 0.0;
                    for (size_t p = 0; p < n; ++p) {
                        temp2[i] += B[i * n + p] * V[p * k + j];
                    }
                }
                // temp = B^T * temp2
                for (size_t i = 0; i < n; ++i) {
                    temp[i] = 0.0;
                    for (size_t p = 0 riselect p = 0.0;
                    for (size_t p = 0; p < k; ++p) {
                        temp[i] += B[p * n + i] * temp2[p];
                    }
                    for (size_t i = 0; i < n; ++i) {
                        V[i * k + j] = temp[i];
                    }
                }
                // Estimate singular value
                norm = 0.0;
                for (size_t i = 0; i < k; ++i) {
                    norm += temp2[i] * temp2[i];
                }
                S[j] = std::sqrt(norm);

                // Compute U_B(:,j) = temp2 / S[j]
                if (S[j] > 1e-10) {
                    for (size_t i = 0; i < k; ++i) {
                        U_B[i * k + j] = temp2[i] / S[j];
                    }
                } else {
                    for (size_t i = 0; i < k; ++i) {
                        U_B[i * k + j] = 0.0;
                    }
                }

                // Orthogonalize V(:,j) against previous vectors
                for (size_t p = 0; p < j; ++p) {
                    double dot = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        dot += V[i * k + j] * V[i * k + p];
                    }
                    for (size_t i = 0; i < n; ++i) {
                        V[i * k + j] -= dot * V[i * k + p];
                    }
                }
            }
        }
        delete[] temp;
        delete[] temp2;
    }

public:
    RandomizedSVD(size_t k_, unsigned int seed_ = FIXED_SEED, bool useFixedSeed_ = USE_FIXED_SEED)
        : m(0), n(0), k(k_), A(nullptr), U(nullptr), S(nullptr), V(nullptr),
          runtime(0.0), seed(seed_), useFixedSeed(useFixedSeed_) {}

    ~RandomizedSVD() {
        delete[] A;
        delete[] U;
        delete[] S;
        delete[] V;
    }

    bool loadMatrix(const double* matrix, size_t rows, size_t cols) {
        if (rows == 0 || cols == 0) {
            std::cerr << "Invalid matrix dimensions" << std::endl;
            return false;
        }
        if (k > std::min(rows, cols)) {
            std::cerr << "Target rank " << k << " exceeds matrix dimensions" << std::endl;
            return false;
        }

        m = rows;
        n = cols;
        delete[] A;
        A = new double[m * n];
        for (size_t i = 0; i < m * n; ++i) {
            A[i] = matrix[i];
        }

        delete[] U;
        delete[] S;
        delete[] V;
        U = new double[m * k];
        S = new double[k];
        V = new double[n * k];
        return true;
    }

    void compute() {
        auto start = std::chrono::high_resolution_clock::now();

        // Step 1: Generate random Gaussian matrix Omega (n x k)
        double* Omega = new double[n * k];
        std::mt19937 gen(useFixedSeed ? seed : std::random_device{}());
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < n * k; ++i) {
            Omega[i] = dist(gen);
        }

        // Step 2: Compute Y = A * Omega (m x k)
        double* Y = new double[m * k];
        matrix_multiply(A, Omega, Y, m, n, k);

        // Step 3: Orthogonalize Y to get Q (m x k)
        double* Q = new double[m * k];
        qr_decomposition(Y, Q, m, k);

        // Step 4: Compute B = Q^T * A (k x n)
        double* B = new double[k * n];
        transpose_multiply(Q, A, B, m, k, n);

        // Step 5: SVD of B
        double* U_B = new double[k * k];
        small_svd(B, U_B, S, V, k, n);

        // Step 6: Compute U = Q * U_B
        matrix_multiply(Q, U_B, U, m, k, k);

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<double>(end - start).count();

        std::cout << "Randomized SVD completed in " << runtime << " seconds" << std::endl;

        // Cleanup
        delete[] Omega;
        delete[] Y;
        delete[] Q;
        delete[] B;
        delete[] U_B;
    }

    bool saveU(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening U file: " << filename << std::endl;
            return false;
        }
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                file << U[i * k + j];
                if (j < k - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
        std::cout << "U matrix saved to: " << filename << std::endl;
        return true;
    }

    bool saveS(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening S file: " << filename << std::endl;
            return false;
        }
        for (size_t i = 0; i < k; ++i) {
            file << S[i] << "\n";
        }
        file.close();
        std::cout << "Singular values saved to: " << filename << std::endl;
        return true;
    }

    bool saveV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening V file: " << filename << std::endl;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                file << V[i * k + j];
                if (j < k - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
        std::cout << "V matrix saved to: " << filename << std::endl;
        return true;
    }

    bool saveRuntime(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening runtime file: " << filename << std::endl;
            return false;
        }
        file << "runtime_seconds\n" << runtime << "\n";
        file.close();
        std::cout << "Runtime saved to: " << filename << std::endl;
        return true;
    }

    const double* getU() const { return U; }
    const double* getS() const { return S; }
    const double* getV() const { return V; }
    double getRuntime() const { return runtime; }
};

int main(int argc, char* argv[]) {
    size_t k = 2; // Default target rank

    if (argc >= 2) {
        k = atoi(argv[1]);
    }

    RandomizedSVD rsvd(k);

    // Read matrix from CSV
    size_t rows, cols;
    double* matrix = read_csv("../data/matrix.csv", rows, cols);
    if (!matrix) {
        std::cerr << "Failed to read matrix" << std::endl;
        return 1;
    }

    // Load matrix into RandomizedSVD
    if (!rsvd.loadMatrix(matrix, rows, cols)) {
        std::cerr << "Failed to load matrix" << std::endl;
        delete[] matrix;
        return 1;
    }

    // Compute randomized SVD
    rsvd.compute();

    // Save results
    rsvd.saveU("../results/svd/U.csv");
    rsvd.saveS("../results/svd/S.csv");
    rsvd.saveV("../results/svd/V.csv");
    rsvd.saveRuntime("../results/svd/runtime.csv");

    // Print singular values
    std::cout << "\nSingular values:\n";
    const double* S = rsvd.getS();
    for (size_t i = 0; i < k; ++i) {
        std::cout << S[i] << std::endl;
    }

    // Cleanup
    delete[] matrix;

    return 0;
}