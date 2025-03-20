#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

struct DataPoint {
    std::vector<double> features;
    double target;  // Optional, ignored for Nyström
};

class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }
    Matrix(const std::vector<DataPoint>& points) {
        rows = points.size();
        cols = points.empty() ? 0 : points[0].features.size();
        data.resize(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = points[i].features[j];
            }
        }
    }

    std::vector<double>& operator[](int i) { return data[i]; }
    const std::vector<double>& operator[](int i) const { return data[i]; }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            std::cerr << "Matrix multiplication error: incompatible dimensions (" 
                      << rows << "x" << cols << ") * (" << other.rows << "x" << other.cols << ")" << std::endl;
            return Matrix(0, 0);
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result[i][j] += data[i][k] * other[k][j];
                }
            }
        }
        return result;
    }
};

Matrix pseudo_inverse(const Matrix& A) {
    int n = A.rows, m = A.cols;
    Matrix AtA = A.transpose() * A;
    Matrix result(m, n);

    double lambda = 1e-6;
    for (int i = 0; i < m; ++i) {
        AtA[i][i] += lambda;
    }

    if (n == m && n <= 50) {
        Matrix inv(m, m);
        for (int i = 0; i < m; ++i) inv[i][i] = 1.0 / AtA[i][i];
        result = inv * A.transpose();
    }
    return result;
}

class Nystrom {
private:
    int n_components;
    Matrix C;       // n_samples x n_components
    Matrix W_inv;   // n_components x n_components

public:
    Nystrom(int k = 5) : n_components(k) {}

    void fit(const Matrix& X) {
        int n_samples = X.rows;
        int n_features = X.cols;
        if (n_components > n_features) n_components = n_features;

        std::vector<int> indices(n_features);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        std::vector<int> sampled_indices(indices.begin(), indices.begin() + n_components);

        C = Matrix(n_samples, n_components);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_components; ++j) {
                C[i][j] = X[i][sampled_indices[j]];
            }
        }

        Matrix W = C.transpose() * C;  // n_components x n_components (5x5)
        W_inv = pseudo_inverse(W);     // n_components x n_components (5x5)
    }

    Matrix transform(const Matrix& X) {
        // Project X onto the subspace: n_samples x n_components
        Matrix K = C.transpose() * X;  // (n_components x n_samples) * (n_samples x n_features) = n_components x n_features (5x20)
        return W_inv * K;              // (n_components x n_components) * (n_components x n_features) = n_components x n_features (5x20)
        // Note: This gives a feature-space projection; adjust reconstruction accordingly
    }

    Matrix reconstruct(const Matrix& X_reduced) {
        // Reconstruct to n_samples x n_features
        return C * X_reduced;  // (n_samples x n_components) * (n_components x n_features) = n_samples x n_features (9999x20)
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<double> means(n_features, 0.0);
    std::vector<double> stds(n_features, 0.0);
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            means[i] += point.features[i];
        }
    }
    for (int i = 0; i < n_features; ++i) {
        means[i] /= data.size();
    }
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            double diff = point.features[i] - means[i];
            stds[i] += diff * diff;
        }
    }
    for (int i = 0; i < n_features; ++i) {
        stds[i] = sqrt(stds[i] / data.size());
        if (stds[i] < 1e-9) stds[i] = 1e-9;
    }
    
    for (auto& point : scaled_data) {
        for (int i = 0; i < n_features; ++i) {
            point.features[i] = (point.features[i] - means[i]) / stds[i];
        }
    }
    return scaled_data;
}

std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return data;
    }
    std::string line;
    if (!getline(file, line)) return data;
    std::cout << "Header: " << line << std::endl;

    std::stringstream header_ss(line);
    std::string value;
    std::vector<std::string> header_cols;
    while (getline(header_ss, value, ',')) {
        header_cols.push_back(value);
    }
    int total_cols = header_cols.size();
    int n_features = total_cols - 1;  // Assume first is index, last is target or last feature
    bool has_index = (header_cols[0].empty() || header_cols[0] == "index" || header_cols[0][0] == ',');
    bool has_target = (total_cols > 1 && (header_cols.back() == "label" || header_cols.back() == "target"));
    if (has_index) n_features--;  // Exclude index
    if (!has_target && total_cols > 1) n_features++;  // No target, last column is a feature

    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        double target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',')) {
            if (has_index && column == 0) {  // Skip index
                column++;
                continue;
            }
            try {
                if (column < total_cols - (has_target ? 1 : 0)) {  // Features
                    features.push_back(std::stod(value));
                } else if (has_target && column == total_cols - 1) {  // Target
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return {};
            }
            column++;
        }
        
        if (features.size() != n_features) {
            std::cerr << "Error: Expected " << n_features << " features, got " << features.size() 
                      << " at line " << line_num << std::endl;
            return {};
        }
        data.push_back({features, target});
        line_num++;
    }
    std::cout << "Loaded " << data.size() << " data points with " << n_features << " features each" << std::endl;
    return data;
}

double compute_reconstruction_error(const Matrix& original, const Matrix& reconstructed) {
    if (original.rows != reconstructed.rows || original.cols != reconstructed.cols) {
        std::cerr << "Dimension mismatch in reconstruction error calculation: original ("
                  << original.rows << "x" << original.cols << "), reconstructed ("
                  << reconstructed.rows << "x" << reconstructed.cols << ")" << std::endl;
        return -1.0;
    }
    double error = 0.0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            double diff = original[i][j] - reconstructed[i][j];
            error += diff * diff;
        }
    }
    return std::sqrt(error);
}

void write_results_to_csv(const Matrix& original, const Matrix& reduced, const Matrix& reconstructed, 
                         const std::string& filename, int n_features, int n_components) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    if (reduced.rows == 0 || reconstructed.rows == 0) {
        std::cerr << "Error: Invalid reduced or reconstructed matrix dimensions" << std::endl;
        return;
    }

    file << "index";
    for (int i = 0; i < n_features; ++i) file << ",feature_" << i;
    for (int i = 0; i < n_components; ++i) file << ",component_" << i;
    for (int i = 0; i < n_features; ++i) file << ",reconstructed_" << i;
    file << "\n";

    for (int i = 0; i < original.rows; ++i) {
        file << i;
        for (int j = 0; j < original.cols; ++j) file << "," << original[i][j];
        for (int j = 0; j < reduced.cols; ++j) file << "," << reduced[i][j];
        for (int j = 0; j < reconstructed.cols; ++j) file << "," << reconstructed[i][j];
        file << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("../data/clustering/X_20d_10.csv");
    if (raw_data.empty()) {
        std::cerr << "Error: Failed to load data" << std::endl;
        return 1;
    }
    int n_features = raw_data.empty() ? 0 : raw_data[0].features.size();

    std::vector<DataPoint> data = scale_features(raw_data);
    if (data.empty()) {
        std::cerr << "Error: Feature scaling failed" << std::endl;
        return 1;
    }

    Matrix X(data);
    int n_components = std::min(20, n_features);
    Nystrom nystrom(n_components);

    auto start = std::chrono::high_resolution_clock::now();
    nystrom.fit(X);
    Matrix X_reduced = nystrom.transform(X);      // 5 x 20 (feature-space projection)
    Matrix X_reconstructed = nystrom.reconstruct(X_reduced);  // 9999 x 20
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Training time: " << duration.count() << " ms" << std::endl;

    double recon_error = compute_reconstruction_error(X, X_reconstructed);
    std::cout << "Reconstruction Error (Frobenius norm): " << recon_error << std::endl;

    // write_results_to_csv(X, X_reduced, X_reconstructed, "../results/nystrom/results.csv", n_features, n_components);

    return 0;
}