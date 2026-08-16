// Generated H100 CUDA port for digit3_1.
// Pivot comparison keeps the FloatX E5M2 choice used in the transformed CPU program.
#define DLU_PIVOT_E 5
#define DLU_PIVOT_T 2
#define DLU_AKK_E 5
#define DLU_AKK_T 10

#include "../cuda_dense_lu_h100.cuh"
