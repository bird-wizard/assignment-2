#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }
    
    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *output_file = argv[3];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, output;

    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    output.shape[0] = 1;
    output.shape[1] = 1;
    output.data = (float*)malloc(sizeof(float) * rows * cols);

    // Sum all elements of the array
    //@@ Modify the below code in the remaining demos

    // Initialize 4 lanes of floats to zero
    float sum = 0.0;
    float four_sum;

    // Increment by 4 instead of by 1
    for (int i = 0; i < rows * cols; i += 4)
    {
        // Load the current 4 elements into a quadword
        float32x4_t vec = vld1q_f32(&host_a.data[i]);

        // Sum all four elements of the quadword
        float32x2_t low_half = vget_low_f32(vec);
        float32x2_t high_half = vget_high_f32(vec);
        float32x2_t pair_sum = vpadd_f32(low_half, high_half);

        // Extract the value from the quad word
        four_sum = vget_lane_f32(pair_sum, 0) + vget_lane_f32(pair_sum, 1);

        sum += four_sum;
    }

    printf("sum: %f == %f\n", sum, host_b.data[0]);
    //printf("Error (My Sum - Answer Sum): %f\n",sum - host_b.data[0]);

    output.data[0] = sum;
    err = CheckMatrix(&host_b, &output);
    CHECK_ERR(err, "CheckMatrix");
    SaveMatrix(output_file, &output);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(output.data);

    return 0;
}