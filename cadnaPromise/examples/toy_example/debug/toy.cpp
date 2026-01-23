#include <half.hpp>
#include <floatx.hpp>
#include <iostream>

void sumArrays(flx::floatx<5, 5>* arr1, flx::floatx<8, 7>* arr2, flx::floatx<5, 5>* result, flx::floatx<8, 7> *sum, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
        *sum += (double)result[i];
    }
}

int main() {
    int size = 5;
    flx::floatx<5, 5>* arr1 = new flx::floatx<5, 5>[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
    flx::floatx<8, 7>* arr2 = new flx::floatx<8, 7>[size] {6.63, 7.717, 8.82, 9.9, 10.141};
    flx::floatx<5, 5>* result = new flx::floatx<5, 5>[size];
    flx::floatx<8, 7> result_sum; 

    sumArrays(arr1, arr2, result, &result_sum, size);
    
    std::cout << "result_sum:" << result_sum << std::endl;
    delete[] arr1; delete[] arr2; delete[] result;

    return 0;
}