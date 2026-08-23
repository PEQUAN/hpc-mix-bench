// Generated H100 CUDA port for digit3_2.
// The macro values mirror the FloatX types selected in the transformed CPU program.
#define BP_SQUASH_E 5
#define BP_SQUASH_T 10
#define BP_OUTPUT_O_E 5
#define BP_OUTPUT_O_T 10
#define BP_OUTPUT_T_E 5
#define BP_OUTPUT_T_T 10
#define BP_HIDDEN_H_E 5
#define BP_HIDDEN_H_T 2
#define BP_HIDDEN_SUM_E 5
#define BP_HIDDEN_SUM_T 2
#define BP_ADJUST_E 5
#define BP_ADJUST_T 2

#include "../cuda_backprop_h100.cuh"
