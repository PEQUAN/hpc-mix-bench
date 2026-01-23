#include <iostream>

void sumArrays(double* arr1, double* arr2, double* result, double *sum, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
        *sum += (double)result[i];
    }
}

int main() {
    int size = 5;
    double* arr1 = new double[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
    double* arr2 = new double[size] {6.63, 7.717, 8.82, 9.9, 10.141};
    double* result = new double[size];
    double result_sum; 

    sumArrays(arr1, arr2, result, &result_sum, size);
    std::cout << "result_sum:" << result_sum << std::endl;
    delete[] arr1; delete[] arr2; delete[] result;

    return 0;
}