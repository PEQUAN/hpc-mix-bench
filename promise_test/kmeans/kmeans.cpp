#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_map>

const bool USE_FIXED_SEED = true;

struct DataPoint {
    std::vector<__PROMISE__> features;
    int label;
};

std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    getline(file, line); 
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<__PROMISE__> features;
        
        // Skip the index column
        getline(ss, value, ',');  // Ignore the first value (index)
        
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

class KMeans {
private:
    int numPoints;
    int numFeatures;
    int k;
    std::vector<__PROMISE__> data;
    std::vector<int> labels;
    std::vector<int> groundTruth;
    
    __PROMISE__ runtime;
    unsigned int seed;
    bool useFixedSeed;

    std::vector<__PROMISE__> getPoint(int idx) const {
        if (idx < 0 || idx >= numPoints) {
            std::cerr << "Error: Invalid point index " << idx << std::endl;
            return std::vector<__PROMISE__>(numFeatures, 0.0);
        }
        std::vector<__PROMISE__> point(numFeatures);
        for (int j = 0; j < numFeatures; ++j) {
            point[j] = data[idx * numFeatures + j];
        }
        return point;
    }

    __PROMISE__ euclideanDistance(const std::vector<__PROMISE__>& p1, const std::vector<__PROMISE__>& p2) const {
        if (p1.size() != p2.size()) {
            std::cerr << "Error: Mismatched dimensions in euclideanDistance" << std::endl;
            return 0.0;
        }
        __PROMISE__ sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return sqrt(sum);
    }

    void initializeCentroids() {
        std::mt19937 gen;
        if (useFixedSeed) {
            gen = std::mt19937(seed);
            std::cout << "Using fixed seed: " << seed << std::endl;
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
            std::cout << "Using random_device for seed" << std::endl;
        }

        std::uniform_int_distribution<> dis(0, numPoints - 1);
        int firstCentroid = dis(gen);
        std::vector<__PROMISE__> centroid = getPoint(firstCentroid);
        for (int j = 0; j < numFeatures; ++j) {
            centroids[j] = static_cast<__PROMISE__>(centroid[j]);
        }

        for (int c = 1; c < k; ++c) {
            std::vector<__PROMISE__> distances(numPoints, std::numeric_limits<__PROMISE__>::max());
            for (int i = 0; i < numPoints; ++i) {
                auto point = getPoint(i);
                __PROMISE__ minDist = std::numeric_limits<__PROMISE__>::max();
                for (int j = 0; j < c; ++j) {
                    std::vector<__PROMISE__> cent(numFeatures);
                    for (int f = 0; f < numFeatures; ++f) {
                        cent[f] = centroids[j * numFeatures + f];
                    }
                    __PROMISE__ dist = static_cast<__PROMISE__>(euclideanDistance(point, cent));
                    minDist = min(minDist, dist);
                }
                distances[i] = minDist * minDist;
            }

            std::discrete_distribution<> dist(distances.begin(), distances.end());
            int nextCentroid = dist(gen);
            auto newCentroid = getPoint(nextCentroid);
            for (int j = 0; j < numFeatures; ++j) {
                centroids[c * numFeatures + j] = newCentroid[j];
            }
        }
    }

public:
    std::vector<__PROMISE__> centroids;
    KMeans(int k_, int numFeatures_, unsigned int seed_ = 0, bool useFixedSeed_ = false)
        : k(k_), numFeatures(numFeatures_), numPoints(0), runtime(0.0),
          seed(seed_), useFixedSeed(useFixedSeed_) {}

    bool loadFromDataPoints(const std::vector<DataPoint>& dataPoints) {
        if (dataPoints.empty()) {
            std::cerr << "No data points provided" << std::endl;
            return false;
        }

        numPoints = dataPoints.size();
        if (dataPoints[0].features.size() != static_cast<size_t>(numFeatures)) {
            std::cerr << "Feature count mismatch: expected " << numFeatures 
                      << ", got " << dataPoints[0].features.size() << std::endl;
            numPoints = 0;
            return false;
        }

        data.resize(numPoints * numFeatures);
        groundTruth.resize(numPoints);
        labels.resize(numPoints, 0);
        centroids.resize(k * numFeatures, 0.0);

        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                data[i * numFeatures + j] = static_cast<__PROMISE__>(dataPoints[i].features[j]);
            }
            groundTruth[i] = dataPoints[i].label;
        }
        return true;
    }

    void fit(int maxIterations = 100) {
        if (numPoints < k) {
            std::cerr << "Number of points (" << numPoints << ") less than k (" << k << ")" << std::endl;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        initializeCentroids();
        bool changed = true;
        int iterations = 0;

        while (changed && iterations < maxIterations) {
            changed = false;

            for (int i = 0; i < numPoints; ++i) {
                auto point = getPoint(i);
                __PROMISE__ minDist = std::numeric_limits<__PROMISE__>::max();
                int newLabel = 0;

                for (int c = 0; c < k; ++c) {
                    std::vector<__PROMISE__> centroid(numFeatures);
                    for (int j = 0; j < numFeatures; ++j) {
                        centroid[j] = centroids[c * numFeatures + j];
                    }
                    __PROMISE__ dist = static_cast<__PROMISE__>(euclideanDistance(point, centroid));
                    if (dist < minDist) {
                        minDist = dist;
                        newLabel = c;
                    }
                }

                if (labels[i] != newLabel) {
                    labels[i] = newLabel;
                    changed = true;
                }
            }

            std::vector<int> counts(k, 0);
            std::fill(centroids.begin(), centroids.end(), 0.0);

            for (int i = 0; i < numPoints; ++i) {
                int cluster = labels[i];
                counts[cluster]++;
                for (int j = 0; j < numFeatures; ++j) {
                    centroids[cluster * numFeatures + j] += data[i * numFeatures + j];
                }
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    for (int j = 0; j < numFeatures; ++j) {
                        centroids[c * numFeatures + j] /= counts[c];
                    }
                }
            }

            iterations++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<__PROMISE__>(end - start).count();

        std::cout << "Converged after " << iterations << " iterations" << std::endl;
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    }

    __PROMISE__ calculateSSE() const {
        __PROMISE__ sse = 0.0;
        for (int i = 0; i < numPoints; ++i) {
            auto point = getPoint(i);
            int cluster = labels[i];
            std::vector<__PROMISE__> centroid(numFeatures);
            for (int j = 0; j < numFeatures; ++j) {
                centroid[j] = centroids[cluster * numFeatures + j];
            }
            __PROMISE__ dist = euclideanDistance(point, centroid);
            sse += dist * dist;
        }
        return sse;
    }
    __PROMISE__ calculateARI() const {
        if (groundTruth.empty()) {
            std::cerr << "No ground truth labels available for ARI calculation" << std::endl;
            return 0.0;
        }

        int maxLabel = *std::max_element(groundTruth.begin(), groundTruth.end()) + 1;
        int maxCluster = k;
        std::vector<std::vector<int>> contingency(maxCluster, std::vector<int>(maxLabel, 0));
        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        std::vector<int> a(maxCluster, 0), b(maxLabel, 0);
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }

        __PROMISE__ sum_nij = 0.0, sum_a = 0.0, sum_b = 0.0;
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                sum_nij += (static_cast<__PROMISE__>(contingency[i][j]) * (contingency[i][j] - 1)) / 2.0;
            }
            sum_a += (static_cast<__PROMISE__>(a[i]) * (a[i] - 1)) / 2.0;
        }
        for (int j = 0; j < maxLabel; ++j) {
            sum_b += (static_cast<__PROMISE__>(b[j]) * (b[j] - 1)) / 2.0;
        }

        __PROMISE__ n = numPoints;
        __PROMISE__ expected = (sum_a * sum_b) / (n * (n - 1) / 2.0);
        __PROMISE__ max_index = (sum_a + sum_b) / 2.0;
        __PROMISE__ index = sum_nij;

        if (max_index == expected) return 0.0;
        return (index - expected) / (max_index - expected);
    }


    const std::vector<int>& getLabels() const { return labels; }
    const std::vector<__PROMISE__>& getCentroids() const { return centroids; }
    __PROMISE__ getRuntime() const { return runtime; }
};

int main(int argc, char *argv[]) {
    size_t K(2), NUM_FEATURES(2);
    size_t SEED(42);

    if(argc == 2){
        K = atoi(argv[1]);
    } else if(argc == 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]);
    } else if(argc > 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]); 
        SEED = atoi(argv[3]); 
    }
    
    KMeans kmeans(K, NUM_FEATURES, SEED, USE_FIXED_SEED);

    std::vector<DataPoint> dataPoints = read_csv("blobs_2d_10_include_y.csv");
    if (dataPoints.empty()) {
        std::cerr << "Failed to read CSV data" << std::endl;
        return 1;
    }

    if (!kmeans.loadFromDataPoints(dataPoints)) {
        std::cerr << "Failed to load data points into KMeans" << std::endl;
        return 1;
    }

    kmeans.fit();
    //__PROMISE__ SSE = kmeans.calculateSSE(); 
    std::cout << "\nEvaluation Metrics:" << std::endl;
    //std::cout << "SSE: " << SSE << std::endl;
    //std::cout << "AMI: " << kmeans.calculateAMI() << std::endl;
    //std::cout << "ARI: " << kmeans.calculateARI() << std::endl;

    PROMISE_CHECK_ARRAY(kmeans.centroids.data(), K*NUM_FEATURES);
    // const auto& centroids = kmeans.getCentroids();
    // std::cout << "\nCentroids:\n";
    // for (int c = 0; c < K; ++c) {
    //     std::cout << "Centroid " << c << ": ";
    //     for (int j = 0; j < NUM_FEATURES; ++j) {
    //         std::cout << centroids[c * NUM_FEATURES + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}