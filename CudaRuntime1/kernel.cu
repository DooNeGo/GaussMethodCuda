#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

__global__ void divideRowGPU(float* cudaMatrix, const int counter, const int columns, const int elemsPerThread, const float divider)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * elemsPerThread;
    if (x >= columns)
        return;
    for (int j = 0; x + j < columns && j < elemsPerThread; j++)
    {
        cudaMatrix[counter * columns + x + j] /= divider;
    }
}

__global__ void doGaussGPU(float* cudaMatrix, const int columns, const int i, const int rows, const int elemsPerThread, const float *temp)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = i + 1 + y;
    if (k >= rows)
        return;
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * elemsPerThread;
    for (int j = 0; x + j < columns && j < elemsPerThread; j++)
    {
        cudaMatrix[k * columns + x + j] -= cudaMatrix[i * columns + x + j] * temp[k - 1];
    }
}

__global__ void copyMatrixColumn(float *temp, const float *cudaMatrix, const int rows, const int cols, const int counter, const int elemsPerThread)
{
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * elemsPerThread;
    int i = counter + y;
    for (int j = 0; i + j < rows - 1 && j < elemsPerThread; j++)
    {
        temp[i + j] = cudaMatrix[(i + 1 + j) * cols + counter];
    }
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
    int threadNum = 1;
    const int elemsPerThread = 16;
    dim3 blockSize = dim3(threadNum, 1, 1);
    dim3 blockSize1 = dim3(1, threadNum, 1);
    dim3 gridSize = dim3(columns / (threadNum * elemsPerThread) + 1, rows, 1);
    dim3 gridSizeOnlyColumn = dim3(columns / (threadNum * elemsPerThread) + 1, 1, 1);
    dim3 gridSizeOnlyRow = dim3(1, rows / (threadNum * elemsPerThread) + 1, 1);
    float* gpuTemp = 0;
    cudaMalloc((float**)&gpuTemp, sizeof(float) * (rows - 1));
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpy(newMatrix, cudaMatrix, sizeof(float) * rows * columns, cudaMemcpyDeviceToHost);
        float divider = newMatrix[i * columns + i];
        divideRowGPU << < gridSizeOnlyColumn, blockSize >> > (cudaMatrix, i, columns, elemsPerThread, divider);
        cudaDeviceSynchronize();
        copyMatrixColumn<<< gridSizeOnlyRow, blockSize1>>>(gpuTemp, cudaMatrix, rows, columns, i, elemsPerThread);
        cudaDeviceSynchronize();
        doGaussGPU << < gridSize, blockSize >> > (cudaMatrix, columns, i, rows, elemsPerThread, gpuTemp);
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
    cudaFree(gpuTemp);

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
    
    //printf("\nEntered matrix:\n");
    //showMatrix(matrix, rows, columns);
    //printf("\nNew matrix:\n");
    //showMatrix(newMatrix, rows, columns);
    //printf("\nRoots: ");
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
