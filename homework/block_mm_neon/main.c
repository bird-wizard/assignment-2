#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void BlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    //@@ Insert code to implement naive matrix multiply here
    int rows1, cols1, rows2, cols2;
    rows1 = input0->shape[0];
    cols1 = input0->shape[1];
    rows2 = input1->shape[0];
    cols2 = input1->shape[1];

    // Initialize result matrix
    for (int i = 0; i < rows1*cols2; i++){
        result->data[i] = 0.0f;
    }

    int blockSize = 5;
    float input[4] = {0.0f};
    float four_sum = 0;
    
    for (int r1 = 0; r1 < rows1; r1+=blockSize){        
        for (int c2 = 0; c2 < cols2; c2+=blockSize){

            // Remainder
            int real_blockSize_R1 = blockSize;
            int real_blockSize_C2 = blockSize;
            
            if (r1 + blockSize > rows1){
                real_blockSize_R1 = rows1 - r1;
            }
            
            if (c2 + blockSize > cols2){
                real_blockSize_C2 = cols2 - c2;
            }
            // C1 Iterator
            for(int c1 = 0; c1 < cols1; c1 += 4){
                // R1 Block
                for(int block_r1 = r1; block_r1 < r1 + real_blockSize_R1; block_r1++){
                    // C2 Block
                    for(int block_c2 = c2; block_c2 < c2 + real_blockSize_C2; block_c2++){
                        
                            input[0] = input0->data[block_r1*cols1 + c1] * input1->data[c1*cols2 + block_c2];
                            input[1] = input0->data[block_r1*cols1 + c1 + 1] * input1->data[(c1 + 1)*cols2 + block_c2];
                            input[2] = input0->data[block_r1*cols1 + c1 + 2] * input1->data[(c1 + 2)*cols2 + block_c2];
                            input[3] = input0->data[block_r1*cols1 + c1 + 3] * input1->data[(c1 + 3)*cols2 + block_c2];

                            
                            // Load the current 4 elements into a quadword
                            float32x4_t vec = vld1q_f32(&input);

                            // Sum all four elements of the quadword
                            float32x2_t low_half = vget_low_f32(vec);
                            float32x2_t high_half = vget_high_f32(vec);
                            float32x2_t pair_sum = vpadd_f32(low_half, high_half);

                            // Extract the value from the quad word
                            four_sum = vget_lane_f32(pair_sum, 0) + vget_lane_f32(pair_sum, 1);

                            result->data[block_r1*cols2 + block_c2] += four_sum;
                        }
                    }
            }        
        }
    }
}
    

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1]; // Dataset/1/input0.raw   
    const char *input_file_b = argv[2]; // Dataset/1/input1.raw
    const char *input_file_c = argv[3]; // Dataset/1/output.raw
    const char *input_file_d = argv[4]; // output.raw
 
    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    BlockMatrixMultiply(&host_a, &host_b, &host_c);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}