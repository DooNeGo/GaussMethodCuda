
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define SYSTEM_PAUSE system("pause");
#define GO_TO_NEXT_LINE printf("\n");

enum
{
    SUCCESSFUL_COMPLETION = 0,
    UNSUCCESSFUL_COMPLETION = 1,
};

float* getMatrix(int rows, int columns)
{
    srand(time(0));
    float* matrix = (float*)malloc(sizeof(int) * rows * columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            *(matrix + i * columns + j) = -3.0 + 8.0 * (rand() / (float)RAND_MAX);
            //scanf("%f", matrix + i * columns + j);
        }
    }
    return matrix;
}

void showMatrix(float* matrix, int rows, int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            fprintf(stdout, "%7.2f", matrix[i * columns + j]);
        }
        GO_TO_NEXT_LINE
    }
}

__global__ void divideRowGPU(float* cudaMatrix, int counter, int columns, int elemsPerThread, int threadNum, float divider)
{
    int i = blockIdx.x * blockDim.x * elemsPerThread + threadIdx.x * elemsPerThread;
    if (i >= columns)
        return;
    for (int j = 0; i < columns && j < elemsPerThread; j++)
    {
        cudaMatrix[counter * columns + i + j] /= divider;
    }

}

__global__ void doGaussGPU(float* cudaMatrix, int columns, int i, int rows, int elemsPerThread, int threadNum)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = i + 1 + y;
    if (k >= rows)
        return;
    float temp = cudaMatrix[k * columns + i];
    int l = blockIdx.x * blockDim.x * elemsPerThread + threadIdx.x * elemsPerThread;
    for (int j = 0; l < columns && j < elemsPerThread; j++, l++)
        cudaMatrix[k * columns + l] -= cudaMatrix[i * columns + l] * temp;
}

float *doGaussHost(float* matrix, int rows, int columns)
{
    float* cudaMatrix = 0;
    float* newMatrix = (float*)malloc(sizeof(float) * rows * columns);
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((float**)&cudaMatrix, sizeof(float) * rows * columns);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return NULL;
    }
    cudaStatus = cudaMemcpy(cudaMatrix, matrix, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return NULL;
    }
    int threadNum = 128;
    const int elemsPerThread = 16;
    dim3 blockSize = dim3(threadNum, 1, 1);
    dim3 gridSize = dim3(columns / (threadNum * elemsPerThread) + 1, rows, 1);
    dim3 gridSize1 = dim3(columns / (threadNum * elemsPerThread) + 1, 1, 1);
    for (int i = 0; i < rows; i++)
    {
        float divider = matrix[i * columns + i];
        divideRowGPU << < gridSize1, blockSize >> > (cudaMatrix, i, columns, elemsPerThread, threadNum, divider);
        cudaDeviceSynchronize();
        //doGaussGPU << < gridSize, blockSize >> > (cudaMatrix, columns, i, rows, elemsPerThread, threadNum);
        cudaDeviceSynchronize();
        //printf("%d\n", i + 1);
    }

    cudaStatus = cudaMemcpy(newMatrix, cudaMatrix, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return NULL;
    }

    cudaFree(cudaMatrix);
    return newMatrix;
}

float* getRoots(float* matrix, int rows, int columns)
{
    float* roots = (float*)malloc(sizeof(float) * rows);
    for (int i = rows - 1; i >= 0; i--)
    {
        float temp = 0;
        for (int j = rows - (rows - 1 - i); j < rows; j++)
        {
            temp += *(matrix + i * columns + j) * roots[j];
        }
        roots[i] = *(matrix + i * columns + rows) - temp;
    }
    return roots;
}

void showRoots(float* roots, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        printf("x%d: %3.3f  ", i + 1, roots[i]);
    }
    GO_TO_NEXT_LINE
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return UNSUCCESSFUL_COMPLETION;
    }

    int rows;
    int columns;
    clock_t timeBefore;
    clock_t timeAfter;

    printf("Enter rows and columns: ");
    scanf("%d%d", &rows, &columns);

    float* matrix = getMatrix(rows, columns);
    timeBefore = clock();
    float* newMatrix = doGaussHost(matrix, rows, columns);
    timeAfter = clock();
    float* roots = getRoots(newMatrix, rows, columns);
    double delay = (double)(timeAfter - timeBefore) / CLOCKS_PER_SEC;
    
    printf("\nEntered matrix:\n");
    showMatrix(matrix, rows, columns);
    printf("\nNew matrix:\n");
    showMatrix(newMatrix, rows, columns);
    printf("\nRoots: ");
    showRoots(roots, rows);
    printf("\n\nDelay : %.30lf seconds\n", delay);
    
    free(matrix);
    free(roots);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return UNSUCCESSFUL_COMPLETION;
    }
    system("pause");
    return SUCCESSFUL_COMPLETION;
}
