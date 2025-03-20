#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

struct DataPoint {
    std::vector<double> features;
    double target;
};

struct DecisionStump {
    int feature_index;
    double split_value;
    double left_value;
    double right_value;

    double predict(const std::vector<double>& features) {
        return features[feature_index] < split_value ? left_value : right_value;
    }

    void fit(const std::vector<DataPoint>& data, const std::vector<double>& weights) {
        int n_features = data[0].features.size();
        double best_error = std::numeric_limits<double>::infinity();
        
        for (int f = 0; f < n_features; ++f) {
            std::vector<double> values;
            for (const auto& point : data) values.push_back(point.features[f]);
            std::sort(values.begin(), values.end());
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                double split = (values[i] + values[i + 1]) / 2;
                double left_sum = 0.0, right_sum = 0.0;
                double left_weight = 0.0, right_weight = 0.0;
                
                for (size_t j = 0; j < data.size(); ++j) {
                    if (data[j].features[f] < split) {
                        left_sum += weights[j] * data[j].target;
                        left_weight += weights[j];
                    } else {
                        right_sum += weights[j] * data[j].target;
                        right_weight += weights[j];
                    }
                }
                
                double left_val = left_weight > 0 ? left_sum / left_weight : 0.0;
                double right_val = right_weight > 0 ? right_sum / right_weight : 0.0;
                double error = 0.0;
                
                for (size_t j = 0; j < data.size(); ++j) {
                    double pred = data[j].features[f] < split ? left_val : right_val;
                    error += weights[j] * std::abs(data[j].target - pred);
                }
                
                if (error < best_error) {
                    best_error = error;
                    feature_index = f;
                    split_value = split;
                    left_value = left_val;
                    right_value = right_val;
                }
            }
        }
    }
};

class AdaBoostRegressor {
private:
    std::vector<DecisionStump> stumps;
    std::vector<double> stump_weights;
    int n_estimators;

public:
    AdaBoostRegressor(int n_est = 50) : n_estimators(n_est) {}
    
    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) return;
        int n_samples = data.size();
        std::vector<double> weights(n_samples, 1.0 / n_samples);
        double total_weight = 1.0;

        for (int t = 0; t < n_estimators; ++t) {
            DecisionStump stump;
            stump.fit(data, weights);
            double error = 0.0;
            std::vector<double> residuals(n_samples);
            
            for (int i = 0; i < n_samples; ++i) {
                residuals[i] = std::abs(data[i].target - stump.predict(data[i].features));
                error += weights[i] * residuals[i];
            }
            error /= total_weight;
            if (error >= 0.5) break;  // Stop if error too large
            
            double alpha = 0.5 * std::log((1.0 - error) / (error + 1e-10));
            stumps.push_back(stump);
            stump_weights.push_back(alpha);
            
            total_weight = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                weights[i] *= std::exp(alpha * residuals[i]);
                total_weight += weights[i];
            }
            for (int i = 0; i < n_samples; ++i) {
                weights[i] /= total_weight;
            }
        }
    }
    
    double predict(const std::vector<double>& features) {
        double sum = 0.0;
        double weight_sum = 0.0;
        for (size_t i = 0; i < stumps.size(); ++i) {
            sum += stump_weights[i] * stumps[i].predict(features);
            weight_sum += stump_weights[i];
        }
        return sum / (weight_sum + 1e-10);
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
    
    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        double target = 0.0;
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
                      const std::vector<double>& predictions, 
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
    
    AdaBoostRegressor ada(50);
    auto start = std::chrono::high_resolution_clock::now();
    ada.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<double> predictions;
    double mse = 0.0;
    for (const auto& point : test_data) {
        double pred = ada.predict(point.features);
        predictions.push_back(pred);
        double diff = pred - point.target;
        mse += diff * diff;
    }
    mse /= test_data.size();
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    
    write_predictions(test_data, predictions, "results/adaboost/preds_reg.csv");
    
    return 0;
}