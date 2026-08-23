// H100 CUDA double-precision baseline for Back Propagation.
#define BP_RUN_LABEL "H100 CUDA backprop double-precision baseline"
#define BP_SQUASH_E 11
#define BP_SQUASH_T 52
#define BP_OUTPUT_O_E 11
#define BP_OUTPUT_O_T 52
#define BP_OUTPUT_T_E 11
#define BP_OUTPUT_T_T 52
#define BP_HIDDEN_H_E 11
#define BP_HIDDEN_H_T 52
#define BP_HIDDEN_SUM_E 11
#define BP_HIDDEN_SUM_T 52
#define BP_ADJUST_E 11
#define BP_ADJUST_T 52

#include "cuda_backprop_h100.cuh"
