/*
 * Rodinia LUD (LU Decomposition) - Single File Version (OpenMP)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define GET_RAND_FP ((double)rand() / ((double)(RAND_MAX) + (double)(1)))
#define MIN(i, j)   ((i) < (j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE { RET_SUCCESS, RET_FAILURE } func_ret_t;

typedef struct __stopwatch_t {
    struct timeval begin;
    struct timeval end;
} stopwatch;

static int do_verify = 0;
static int omp_num_threads = 1;   // default 1 thread

/* ====================== Stopwatch ====================== */
void stopwatch_start(stopwatch *sw) {
    if (sw == NULL) return;
    bzero(&sw->begin, sizeof(struct timeval));
    bzero(&sw->end, sizeof(struct timeval));
    gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw) {
    if (sw == NULL) return;
    gettimeofday(&sw->end, NULL);
}

double get_interval_by_sec(stopwatch *sw) {
    if (sw == NULL) return 0.0;
    return ((double)(sw->end.tv_sec - sw->begin.tv_sec) +
            (double)(sw->end.tv_usec - sw->begin.tv_usec) / 1000000.0);
}

/* ====================== Matrix Creation ====================== */
func_ret_t create_matrix_from_file(double **mp, const char *filename, int *size_p) {
    int i, j, size;
    double *m;
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return RET_FAILURE;
    }

    // 修复警告：检查返回值
    if (fscanf(fp, "%d\n", &size) != 1) {
        fprintf(stderr, "Error: Failed to read matrix size\n");
        fclose(fp);
        return RET_FAILURE;
    }

    m = (double *)malloc(sizeof(double) * size * size);
    if (m == NULL) {
        fclose(fp);
        return RET_FAILURE;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (fscanf(fp, "%lf ", m + i * size + j) != 1) {
                free(m);
                fclose(fp);
                return RET_FAILURE;
            }
        }
    }

    fclose(fp);
    *size_p = size;
    *mp = m;
    return RET_SUCCESS;
}

func_ret_t create_matrix(double **mp, int size) {
    double *m = (double *)malloc(sizeof(double) * size * size);
    if (m == NULL) return RET_FAILURE;

    double lamda = -0.001;
    double coe[2 * size - 1];
    for (int i = 0; i < size; i++) {
        double coe_i = 10.0 * exp(lamda * i);
        coe[size - 1 + i] = coe_i;
        coe[size - 1 - i] = coe_i;
    }

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            m[i * size + j] = coe[size - 1 - i + j];

    *mp = m;
    return RET_SUCCESS;
}

/* ====================== Verification ====================== */
void lud_verify(double *m, double *lu, int matrix_dim) {
    int i, j, k;
    double *tmp = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));

    for (i = 0; i < matrix_dim; i++) {
        for (j = 0; j < matrix_dim; j++) {
            double sum = 0.0;
            double temp = 1.0;
            for (k = 0; k <= MIN(i, j); k++) {
                double l = (i == k) ? temp : lu[i * matrix_dim + k];
                double u = lu[k * matrix_dim + j];
                sum += l * u;
            }
            tmp[i * matrix_dim + j] = sum;
        }
    }

    for (i = 0; i < matrix_dim; i++) {
        for (j = 0; j < matrix_dim; j++) {
            if (fabs(m[i * matrix_dim + j] - tmp[i * matrix_dim + j]) > 0.0001) {
                printf("dismatch at (%d, %d): (o)%f (n)%f\n",
                       i, j, m[i * matrix_dim + j], tmp[i * matrix_dim + j]);
            }
        }
    }
    free(tmp);
}

void matrix_duplicate(double *src, double **dst, int matrix_dim) {
    int s = matrix_dim * matrix_dim * sizeof(double);
    double *p = (double *)malloc(s);
    memcpy(p, src, s);
    *dst = p;
}

/* ====================== Core LUD Kernel (OpenMP) ====================== */
void lud_omp(__PROMISE__ *a, int size) {
    int i, j, k;
    __PROMISE__ sum;

    // 强制设置线程数并禁用动态调整
    omp_set_dynamic(0);
    omp_set_num_threads(omp_num_threads);

    printf("Using %d OpenMP threads\n", omp_num_threads);

    for (i = 0; i < size; i++) {
#pragma omp parallel for private(j, k, sum) shared(size, i, a)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }

#pragma omp parallel for private(j, k, sum) shared(size, i, a)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

/* ====================== Main ====================== */
int main(int argc, char *argv[]) {
    int matrix_dim = 32;
    int opt;
    const char *input_file = NULL;
    double *m = NULL, *mm = NULL;
    stopwatch sw;

    static struct option long_options[] = {
        {"input",   1, NULL, 'i'},
        {"size",    1, NULL, 's'},
        {"verify",  0, NULL, 'v'},
        {"threads", 1, NULL, 'n'},   // 新增：指定线程数
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "::vs:i:n:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'v':
                do_verify = 1;
                break;
            case 's':
                matrix_dim = atoi(optarg);
                printf("Generate input matrix internally, size = %d\n", matrix_dim);
                break;
            case 'n':
                omp_num_threads = atoi(optarg);
                if (omp_num_threads < 1) omp_num_threads = 1;
                printf("Set number of threads = %d\n", omp_num_threads);
                break;
            default:
                fprintf(stderr, "Usage: %s [-v] [-s matrix_size] [-i input_file] [-n threads]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (input_file) {
        printf("Reading matrix from file %s\n", input_file);
        if (create_matrix_from_file(&m, input_file, &matrix_dim) != RET_SUCCESS) {
            fprintf(stderr, "error reading matrix from file\n");
            exit(EXIT_FAILURE);
        }
    } else {
        printf("Creating matrix internally size=%d\n", matrix_dim);
        if (create_matrix(&m, matrix_dim) != RET_SUCCESS) {
            fprintf(stderr, "error creating matrix internally\n");
            exit(EXIT_FAILURE);
        }
    }

    if (do_verify) {
        matrix_duplicate(m, &mm, matrix_dim);
    }

    stopwatch_start(&sw);
    lud_omp(m, matrix_dim);
    PROMISE_CHECK_ARRAY(m, matrix_dim * matrix_dim);  
    stopwatch_stop(&sw);

    printf("Time consumed(ms): %.3f\n", 1000.0 * get_interval_by_sec(&sw));

    if (do_verify) {
        printf(">>>Verify<<<<\n");
        lud_verify(mm, m, matrix_dim);
        free(mm);
    }

    free(m);
    return EXIT_SUCCESS;
}