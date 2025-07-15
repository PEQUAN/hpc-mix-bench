#include <iostream>
#include <vector>
#include <random>

std::vector<double> vectorSubtraction() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    std::vector<half_float::half> vec1(5);
    std::vector<half_float::half> vec2(5);
    
    for(size_t i = 0; i < 5; ++i) {
        vec1[i] = dist(rng);
        vec2[i] = dist(rng);
    }
    
    std::vector<double> result(5);
    for(size_t i = 0; i < 5; ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    
    return result;
}

int main() {
    std::vector<double> result = vectorSubtraction();
    
    std::cout << "Result of vector subtraction:\n";
    for(float val : result) {
        std::cout << val << " ";
    }


    // half_float::half result2[5];
    // double* result2 = new double[5]; works!
    double* result2 = (double*) calloc(5, sizeof(double));



    for (int i=0; i<5; i++){
        result2[i] = result[i];
    }

    PROMISE_CHECK_ARRAY(result2, 5);
    std::cout << "\n";
    
    return 0;
}

