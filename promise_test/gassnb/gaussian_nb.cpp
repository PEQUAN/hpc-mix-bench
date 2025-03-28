#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>

struct DataPoint {
    std::vector<__PROMISE__> features;
    int label;
};

class GaussianNB {
private:
    std::map<int, __PROMISE__> class_priors;                    // P(class)
    std::map<int, std::vector<__PROMISE__>> class_means;        // Mean of each feature per class
    std::map<int, std::vector<__PROMISE__>> class_variances;    // Variance of each feature per class

    // Gaussian probability density function
    __PROMISE__ gaussian_pdf(__PROMISE__ x, __PROMISE__ mean, __PROMISE__ variance) {
        __PROMISE__ exponent = -((x - mean) * (x - mean)) / (2 * variance);
        return (1.0 / sqrt(2 * M_PI * variance)) * exp(exponent);
    }

public:
    void fit(const std::vector<DataPoint>& data) {
        std::map<int, std::vector<std::vector<__PROMISE__>>> class_features;
        int n_features = data[0].features.size();
        
        // Separate features by class and count occurrences
        for (const auto& point : data) {
            class_features[point.label].push_back(point.features);
            class_priors[point.label]++;
        }
        
        // Calculate priors
        for (auto& prior : class_priors) {
            prior.second /= data.size();
        }
        
        // Calculate means and variances for each feature per class
        for (const auto& class_data : class_features) {
            int label = class_data.first;
            const auto& feature_vectors = class_data.second;
            std::vector<__PROMISE__> means(n_features, 0.0);
            std::vector<__PROMISE__> variances(n_features, 0.0);
            
            // Calculate means
            for (const auto& features : feature_vectors) {
                for (size_t i = 0; i < n_features; ++i) {
                    means[i] += features[i];
                }
            }
            for (size_t i = 0; i < n_features; ++i) {
                means[i] /= feature_vectors.size();
            }
            
            // Calculate variances
            for (const auto& features : feature_vectors) {
                for (size_t i = 0; i < n_features; ++i) {
                    __PROMISE__ diff = features[i] - means[i];
                    variances[i] += diff * diff;
                }
            }
            for (size_t i = 0; i < n_features; ++i) {
                variances[i] /= feature_vectors.size();
                // Add small epsilon to prevent division by zero
                if (variances[i] < 1e-9) variances[i] = 1e-9;
            }
            
            class_means[label] = means;
            class_variances[label] = variances;
        }
    }
    
    int predict(const std::vector<__PROMISE__>& features) {
        __PROMISE__ max_log_prob = -std::numeric_limits<__PROMISE__>::infinity();
        int best_class = -1;
        
        // Calculate log probability for each class
        for (const auto& prior : class_priors) {
            int label = prior.first;
            __PROMISE__ log_prob = log(prior.second);  // log(P(class))
            
            for (size_t i = 0; i < features.size(); ++i) {
                __PROMISE__ prob = gaussian_pdf(features[i], 
                                        class_means[label][i], 
                                        class_variances[label][i]);
                log_prob += log(prob);  // Use log to prevent underflow
            }
            
            if (log_prob > max_log_prob) {
                max_log_prob = log_prob;
                best_class = label;
            }
        }

        
        return best_class;
    }
};

// CSV writing function (unchanged)
void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<int>& predictions, 
                      const std::string& filename) {
    std::ofstream file(filename);
    file << "feature1,feature2,...,label,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j];
            if (j < data[i].features.size() - 1) file << ",";
        }
        file << "," << data[i].label << "," << predictions[i] << "\n";
    }
}

// CSV reading function (unchanged)
std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    getline(file, line);  // Skip header: ,feature1,feature2,...,label
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        // Skip the index column
        getline(ss, value, ',');  // Ignore the first value (index)
        
        // Read features
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }

        // Last value is the true label
        int true_label = (int)features.back();
        features.pop_back();
        data.push_back({features, true_label});
    }
    std::cout << "Loaded " << data.size() << " data points with "  
              << (data.empty() ? 0 : data[0].features.size()) << " features each" << std::endl;
    
    file.close();
    return data;
}

int main() {
    // Read data
    std::vector<DataPoint> data = read_csv("iris.csv");
    
    // Train-test split (80-20)
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    // Train and measure time
    GaussianNB classifier;
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    // Predict and calculate accuracy
    std::vector<int> predictions;
    int correct = 0;
    for (const auto& point : test_data) {
        int pred = classifier.predict(point.features);
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test_data.size() * 100;
    
    PROMISE_CHECK_VAR(accuracy);
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    return 0;
}