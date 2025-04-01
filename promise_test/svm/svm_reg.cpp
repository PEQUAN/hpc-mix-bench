#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>

struct DataPoint {
    std::vector<__PROMISE__> features;
    __PROMISE__ target;
};

class SVMRegressor {
private:
    std::vector<__PROMISE__> weights;
    __PROMISE__ bias;
    __PROMISE__ C;  // Regularization parameter
    __PROMISE__ epsilon;  // Epsilon-tube for regression
    int n_features;

    __PROMISE__ dot_product(const std::vector<__PROMISE__>& x1, const std::vector<__PROMISE__>& x2) {
        __PROMISE__ sum = 0.0;
        for (int i = 0; i < n_features; ++i) sum += x1[i] * x2[i];
        return sum;
    }

    __PROMISE__ predict_raw(const std::vector<__PROMISE__>& x) {
        return dot_product(weights, x) + bias;
    }

public:
    SVMRegressor(__PROMISE__ c = 1.0, __PROMISE__ eps = 0.1) : C(c), epsilon(eps) {}
    
    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) return;
        n_features = data[0].features.size();
        weights.resize(n_features, 0.0);
        bias = 0.0;

        // Simple gradient descent for linear SVR
        __PROMISE__ learning_rate = 0.001;
        int max_iter = 1000;
        for (int iter = 0; iter < max_iter; ++iter) {
            __PROMISE__ gradient_bias = 0.0;
            std::vector<__PROMISE__> gradient_weights(n_features, 0.0);
            __PROMISE__ total_error = 0.0;

            for (const auto& point : data) {
                __PROMISE__ pred = predict_raw(point.features);
                __PROMISE__ error = pred - point.target;

                if (std::abs(error) > epsilon) {
                    __PROMISE__ grad = (error > epsilon) ? 1.0 : -1.0;
                    gradient_bias += grad;
                    for (int i = 0; i < n_features; ++i) {
                        gradient_weights[i] += grad * point.features[i];
                    }
                    total_error += std::abs(error) - epsilon;
                }
            }

            // Update weights and bias
            for (int i = 0; i < n_features; ++i) {
                weights[i] -= learning_rate * (gradient_weights[i] + 2 * C * weights[i]);
            }
            bias -= learning_rate * gradient_bias;

            if (total_error < 1e-5) break;  // Early stopping
        }
    }
    
    __PROMISE__ predict(const std::vector<__PROMISE__>& features) {
        return predict_raw(features);
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<__PROMISE__> means(n_features, 0.0);
    std::vector<__PROMISE__> stds(n_features, 0.0);
    
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
            __PROMISE__ diff = point.features[i] - means[i];
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
    
    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<__PROMISE__> features;
        __PROMISE__ target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',')) {
            if (column == 0) {  // Skip index
                column++;
                continue;
            }
            try {
                if (column < 11) {  // Features (age to s6)
                    features.push_back(std::stod(value));
                } else if (column == 11) {  // Target (label)
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return {};
            }
            column++;
        }
        
        if (features.size() != 10) {
            std::cerr << "Error: Expected 10 features, got " << features.size() 
                      << " at line " << line_num << std::endl;
            return {};
        }
        data.push_back({features, target});
        line_num++;
    }
    std::cout << "Loaded " << data.size() << " data points with " 
              << (data.empty() ? 0 : data[0].features.size()) << " features each" << std::endl;
    return data;
}

void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<__PROMISE__>& predictions, 
                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "age,sex,bmi,bp,s1,s2,s3,s4,s5,s6,target,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j] << (j < data[i].features.size() - 1 ? "," : "");
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("../data/regression/diabetes.csv");
    if (raw_data.empty()) return 1;
    
    std::vector<DataPoint> data = scale_features(raw_data);
    if (data.empty()) return 1;
    
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    SVMRegressor svr(1.0, 0.1);
    auto start = std::chrono::high_resolution_clock::now();
    svr.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<__PROMISE__> predictions;
    __PROMISE__ mse = 0.0;
    for (const auto& point : test_data) {
        __PROMISE__ pred = svr.predict(point.features);
        predictions.push_back(pred);
        __PROMISE__ diff = pred - point.target;
        mse += diff * diff;
    }
    mse /= test_data.size();
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    PROMISE_CHECK_ARRAY(predictions.data(), predictions.size());
    write_predictions(test_data, predictions, "../svm/pred_reg.csv");
    
    return 0;
}