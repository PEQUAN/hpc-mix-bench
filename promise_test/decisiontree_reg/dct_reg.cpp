#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

const int MAX_FEATURES = 100;
const int MAX_DATA_POINTS = 10000;
const int MAX_FEATURE_INDICES = 100;

struct DataPoint {
    double* features; 
    int num_features; 
    double target;

    DataPoint() : features(nullptr), num_features(0), target(0.0) {}

    ~DataPoint() { delete[] features; }

    DataPoint(const DataPoint& other) : num_features(other.num_features), target(other.target) {
        features = new double[num_features];
        for (int i = 0; i < num_features; ++i) {
            features[i] = other.features[i];
        }
    }

    DataPoint& operator=(const DataPoint& other) {
        if (this != &other) {
            delete[] features;
            num_features = other.num_features;
            target = other.target;
            features = new __PROMISE__[num_features];
            for (int i = 0; i < num_features; ++i) {
                features[i] = other.features[i];
            }
        }
        return *this;
    }
};

struct DecisionTreeRegressor {
    struct Node {
        bool is_leaf = false;
        __PROMISE__ value = 0.0;
        __PROMISE__ split_value = 0.0;
        int feature_index = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> root;
    int max_depth;

    __PROMISE__ calculate_variance(const DataPoint* data, int size) {
        if (size == 0) return 0.0;
        __PROMISE__ mean = 0.0;
        for (int i = 0; i < size; ++i) mean += data[i].target;
        mean /= size;

        __PROMISE__ variance = 0.0;
        for (int i = 0; i < size; ++i) {
            __PROMISE__ diff = data[i].target - mean;
            variance += diff * diff;
        }
        return variance / size;
    }

    std::pair<int, __PROMISE__> find_best_split(const DataPoint* data, int data_size,
                                          const int* feature_indices, int num_features) {
        if (data_size == 0 || num_features == 0) return {-1, 0.0};

        __PROMISE__ best_reduction = -std::numeric_limits<__PROMISE__>::infinity();
        int best_feature = -1;
        __PROMISE__ best_value = 0.0;

        __PROMISE__ total_variance = calculate_variance(data, data_size);

        for (int f_idx = 0; f_idx < num_features; ++f_idx) {
            int f = feature_indices[f_idx];
            __PROMISE__* values = new __PROMISE__[data_size];
            int valid_values = 0;
            for (int i = 0; i < data_size; ++i) {
                if (f >= data[i].num_features) {
                    std::cerr << "Error: Feature index " << f << " out of bounds" << std::endl;
                    delete[] values;
                    return {-1, 0.0};
                }
                values[valid_values++] = data[i].features[f];
            }
            std::sort(values, values + valid_values);
            if (valid_values < 2) {
                delete[] values;
                continue;
            }

            for (int i = 0; i < valid_values - 1; ++i) {
                __PROMISE__ split_val = (values[i] + values[i + 1]) / 2;

                DataPoint* left = new DataPoint[data_size];
                DataPoint* right = new DataPoint[data_size];
                int left_size = 0, right_size = 0;

                for (int j = 0; j < data_size; ++j) {
                    if (data[j].features[f] < split_val) {
                        left[left_size++] = data[j];
                    } else {
                        right[right_size++] = data[j];
                    }
                }

                if (left_size == 0 || right_size == 0) {
                    delete[] left;
                    delete[] right;
                    continue;
                }

                __PROMISE__ left_var = calculate_variance(left, left_size);
                __PROMISE__ right_var = calculate_variance(right, right_size);
                __PROMISE__ reduction = total_variance -
                                  (left_size * left_var + right_size * right_var) / data_size;

                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_feature = f;
                    best_value = split_val;
                }

                delete[] left;
                delete[] right;
            }
            delete[] values;
        }
        return {best_feature, best_value};
    }

    std::unique_ptr<Node> build_tree(const DataPoint* data, int data_size,
                                    const int* feature_indices, int num_features, int depth) {
        auto node = std::make_unique<Node>();

        if (data_size == 0) {
            std::cerr << "Error: Empty data in build_tree at depth " << depth << std::endl;
            node->is_leaf = true;
            node->value = 0.0;
            return node;
        }

        if (depth >= max_depth || data_size < 2) {
            node->is_leaf = true;
            __PROMISE__ sum = 0.0;
            for (int i = 0; i < data_size; ++i) sum += data[i].target;
            node->value = sum / data_size;
            return node;
        }

        auto [feature, value] = find_best_split(data, data_size, feature_indices, num_features);
        if (feature == -1) {
            node->is_leaf = true;
            __PROMISE__ sum = 0.0;
            for (int i = 0; i < data_size; ++i) sum += data[i].target;
            node->value = sum / data_size;
            return node;
        }

        DataPoint* left_data = new DataPoint[data_size];
        DataPoint* right_data = new DataPoint[data_size];
        int left_size = 0, right_size = 0;

        for (int i = 0; i < data_size; ++i) {
            if (data[i].features[feature] < value) {
                left_data[left_size++] = data[i];
            } else {
                right_data[right_size++] = data[i];
            }
        }

        node->feature_index = feature;
        node->split_value = value;
        node->left = build_tree(left_data, left_size, feature_indices, num_features, depth + 1);
        node->right = build_tree(right_data, right_size, feature_indices, num_features, depth + 1);

        delete[] left_data;
        delete[] right_data;

        return node;
    }

public:
    DecisionTreeRegressor(int max_d = 10) : max_depth(max_d) {}

    void fit(const DataPoint* data, int data_size, const int* feature_indices, int num_features) {
        if (data_size == 0) {
            std::cerr << "Error: Empty dataset in DecisionTreeRegressor::fit" << std::endl;
            return;
        }
        root = build_tree(data, data_size, feature_indices, num_features, 0);
    }

    __PROMISE__ predict(const __PROMISE__* features, int num_features) {
        if (!root) {
            std::cerr << "Error: Tree not initialized in predict" << std::endl;
            return 0.0;
        }
        Node* current = root.get();
        while (!current->is_leaf) {
            if (current->feature_index >= num_features) {
                std::cerr << "Error: Feature index " << current->feature_index
                          << " exceeds feature size " << num_features << std::endl;
                return 0.0;
            }
            current = (features[current->feature_index] < current->split_value) ?
                      current->left.get() : current->right.get();
            if (!current) {
                std::cerr << "Error: Null node in predict" << std::endl;
                return 0.0;
            }
        }
        return current->value;
    }
};

DataPoint* scale_features(const DataPoint* data, int data_size, int& out_size) {
    if (data_size == 0) {
        std::cerr << "Error: Empty data in scale_features" << std::endl;
        out_size = 0;
        return nullptr;
    }

    int n_features = data[0].num_features;
    DataPoint* scaled_data = new DataPoint[data_size];
    for (int i = 0; i < data_size; ++i) {
        scaled_data[i] = data[i];
    }

    __PR_2__* means = new __PR_2__[n_features]();
    __PR_3__* stds = new __PR_3__[n_features]();

    for (int i = 0; i < data_size; ++i) {
        if (data[i].num_features != n_features) {
            std::cerr << "Error: Inconsistent feature count" << std::endl;
            delete[] scaled_data;
            delete[] means;
            delete[] stds;
            out_size = 0;
            return nullptr;
        }
        for (int j = 0; j < n_features; ++j) {
            means[j] += data[i].features[j];
        }
    }
    for (int j = 0; j < n_features; ++j) {
        means[j] /= data_size;
    }

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            __PROMISE__ diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < n_features; ++j) {
        stds[j] = sqrt(stds[j] / data_size);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            scaled_data[i].features[j] = (scaled_data[i].features[j] - means[j]) / stds[j];
        }
    }

    delete[] means;
    delete[] stds;
    out_size = data_size;
    return scaled_data;
}

DataPoint* read_csv(const std::string& filename, int& out_size) {
    DataPoint* data = new DataPoint[MAX_DATA_POINTS];
    int data_size = 0;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        delete[] data;
        out_size = 0;
        return nullptr;
    }

    std::string line;
    getline(file, line); 

    while (getline(file, line) && data_size < MAX_DATA_POINTS) {
        std::stringstream ss(line);
        std::string value;
        __PROMISE__* features = new __PROMISE__[MAX_FEATURES];
        int num_features = 0;

        getline(ss, value, ',');

        // Read features
        while (getline(ss, value, ',')) {
            if (num_features >= MAX_FEATURES) {
                std::cerr << "Error: Too many features in CSV" << std::endl;
                delete[] features;
                delete[] data;
                out_size = 0;
                return nullptr;
            }
            features[num_features++] = std::stod(value);
        }

        if (num_features < 1) {
            delete[] features;
            continue;
        }

        __PROMISE__ true_label = features[num_features - 1];
        --num_features; 

        DataPoint point;
        point.features = new __PROMISE__[num_features];
        point.num_features = num_features;
        point.target = true_label;
        for (int i = 0; i < num_features; ++i) {
            point.features[i] = features[i];
        }

        data[data_size++] = point;
        delete[] features;
    }

    file.close();
    std::cout << "Loaded " << data_size << " data points with "
              << (data_size > 0 ? data[0].num_features : 0) << " features each" << std::endl;

    out_size = data_size;
    return data;
}


int main() {
    int raw_data_size;
    DataPoint* raw_data = read_csv("diabetes.csv", raw_data_size);


    int data_size;
    DataPoint* data = scale_features(raw_data, raw_data_size, data_size);
    delete[] raw_data;

    int train_size = static_cast<int>(0.7 * data_size);


    DataPoint* train_data = new DataPoint[train_size];
    DataPoint* test_data = new DataPoint[data_size - train_size];
    int test_size = data_size - train_size;

    for (int i = 0; i < train_size; ++i) {
        train_data[i] = data[i];
    }
    for (int i = 0; i < test_size; ++i) {
        test_data[i] = data[train_size + i];
    }
    delete[] data;

    int num_features = test_data[0].num_features;
    int* all_features = new int[num_features];
    for (int i = 0; i < num_features; ++i) {
        all_features[i] = i;
    }

    DecisionTreeRegressor dt(5);
    dt.fit(train_data, train_size, all_features, num_features);

    __PR_1__* predictions = new __PR_1__[test_size];
    __PROMISE__ mse = 0.0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = dt.predict(test_data[i].features, test_data[i].num_features);
        __PROMISE__ diff = predictions[i] - test_data[i].target;
        mse += diff * diff;
    }

    PROMISE_CHECK_ARRAY(predictions, test_size);
    mse /= test_size;
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;


    delete[] train_data;
    delete[] test_data;
    delete[] all_features;
    delete[] predictions;

    return 0;
}