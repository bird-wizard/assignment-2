#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void NaiveMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
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

    int blockSize = 2;

    // R1 holds the row of Matrix A
    // Increment by blockSize
    for (int r1 = 0; r1 < rows1; r1+=blockSize){
        
        // C2 holds the column of Matrix B
        // Increment by blockSize
        for (int c2 = 0; c2 < cols2; c2+=blockSize){
            
            // Catch remainders prior to iterating up to blockSize   
            int real_blockSize = blockSize;
            if (r1 + blockSize > rows1){
                real_blockSize = rows1 - r1;
            }

            // e.g. rows1 = 4; blockSize = 5
            // real_blockSize = 4 - 0

            // Iterate over blockRows at index R1 to index R1+BlockSize
            for(int blockRow = r1; blockRow < r1 + real_blockSize; blockRow++){
                //Iterate over blockCols at index C2 to index C2+BlockSize
                for(int blockCol = c2; blockCol < c2 + real_blockSize; blockCol++){
                    // C1 iterates over the element of R1 (R1xCols1+C1) (or in this case, blockRow)
                    for(int c1 = 0; c1 < cols1; c1++){
                        // Row1 of Results is filled first, each element iterated by columns of Matrix B
                        // R1 = 0; Cols2 = 2; C2 = 0, 1, 2, 3...;
                        // element = R1xCols2 + C2

                        // Matrix A index
                        // R1 = 0; Cols1 = 2; C1 = 0, 1, 2, 3...;
                        // element = R1xCols1+C1

                        // Matrix B index
                        // (Column of Matrix A moves pointer of Matrix B down its columns)
                        // C1 = 0, 1, 2, 3...; Cols2 = 2; C2 = 0
                        // element = C1xCols2+C2 
                        result->data[blockRow*cols2 + blockCol] += input0->data[blockRow*cols1 + c1] * input1->data[c1*cols2 + blockCol];
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
    NaiveMatrixMultiply(&host_a, &host_b, &host_c);

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