#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define HALO 1
#define BLOCK_WITH_HALO (BLOCK_SIZE + 2 * HALO)
#define GRID_DIM 3

#define IDX2D(i, j) ((i) * (nx + 2) + (j))

__device__ int SH_ID(int i, int j) 
{
    return i * (blockDim.x + 2) + j;
}
__device__ int ID_2D(int i, int j) 
{
    return i * (nx + 2) + j;
}

__device__ void haloExchange(
    float* sh_h, const float* h, int global_i, int global_j, int local_i, int local_j, int nx, int ny)
{
    // === LEFT halo ===
    if (local_j == 1) 
    {
        int global_id_j = global_j - 1;
        int local_id = SH_ID(local_i, local_j - 1);

        if (global_id_j >= 0) 
        {
            int global_id = ID_2D(global_i, global_id_j);
            sh_h[local_id] = h[global_id];
        }
    }

    // === RIGHT halo ===
    if (local_j == blockDim.x) 
    {
        int global_id_j = global_j + 1;
        int local_id = SH_ID(local_i, local_j + 1);

        if (global_id_j < nx + 2) 
        {
            int global_id = ID_2D(global_i, global_id_j);
            sh_h[local_id] = h[global_id];
        }
    }

    // === BOTTOM halo ===
    if (local_i == 1) 
    {
        int global_id_i = global_i - 1;
        int local_id = SH_ID(local_i - 1, local_j);

        if (global_id_i >= 0) 
        {
            int global_id = ID_2D(global_id_i, global_j);
            sh_h[local_id] = h[global_id];
        }
    }

    // === TOP halo ===
    if (local_i == blockDim.y) 
    {
        int global_id_i = global_i + 1;
        int local_id = SH_ID(local_i + 1, local_j);

        if (global_id_i < ny + 2) 
        {
            int global_id = ID_2D(global_id_i, global_j);
            sh_h[local_id] = h[global_id];
        }
    }
}

__global__ void testHaloKernel(float *h, float *h_result, int nx, int ny) 
{
    int global_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    int local_i = threadIdx.y + 1;
    int local_j = threadIdx.x + 1;

    extern __shared__ float sh_h[];

    int global_id = ID_2D(global_i, global_j);
    int local_id = SH_ID(local_i, local_j);

    sh_h[local_id] = h[global_id];
    __syncthreads();

    haloExchange(sh_h, h, global_i, global_j, local_i, local_j, nx, ny);
    __syncthreads();

    // Write interior
    h_result[global_id] = sh_h[local_id];

    // Write halo if thread is on boundary
    if (threadIdx.x == 0) {
        int global_id_left = ID_2D(global_i, global_j - 1);
        int local_id_left = SH_ID(local_i, local_j - 1);
        if (global_j - 1 >= 0) h_result[global_id_left] = sh_h[local_id_left];
    }
    if (threadIdx.x == blockDim.x - 1) {
        int global_id_right = ID_2D(global_i, global_j + 1);
        int local_id_right = SH_ID(local_i, local_j + 1);
        if (global_j + 1 < nx + 2) h_result[global_id_right] = sh_h[local_id_right];
    }
    if (threadIdx.y == 0) {
        int global_id_bottom = ID_2D(global_i - 1, global_j);
        int local_id_bottom = SH_ID(local_i - 1, local_j);
        if (global_i - 1 >= 0) h_result[global_id_bottom] = sh_h[local_id_bottom];
    }
    if (threadIdx.y == blockDim.y - 1) {
        int global_id_top = ID_2D(global_i + 1, global_j);
        int local_id_top = SH_ID(local_i + 1, local_j);
        if (global_i + 1 < ny + 2) h_result[global_id_top] = sh_h[local_id_top];
    }
}

int main() {
    const int nx = BLOCK_SIZE * GRID_DIM;
    const int ny = BLOCK_SIZE * GRID_DIM;
    const int total_size = (nx + 2) * (ny + 2);

    float *h_h = (float*)malloc(total_size * sizeof(float));
    float *h_result = (float*)malloc(total_size * sizeof(float));

    for (int i = 0; i < ny + 2; ++i) {
        for (int j = 0; j < nx + 2; ++j) {
            int block_id = (i - 1) / BLOCK_SIZE * GRID_DIM + (j - 1) / BLOCK_SIZE;
            h_h[IDX2D(i, j)] = (i == 0 || j == 0 || i == ny + 1 || j == nx + 1) ? -1.0f : block_id;
        }
    }

    float *d_h, *d_result;
    cudaMalloc(&d_h, total_size * sizeof(float));
    cudaMalloc(&d_result, total_size * sizeof(float));

    cudaMemcpy(d_h, h_h, total_size * sizeof(float), cudaMemcpyHostToDevice);

    printf("Before halo exchange:\n");
    for (int i = 0; i < ny + 2; ++i) {
        for (int j = 0; j < nx + 2; ++j) {
            printf("%5.1f ", h_h[IDX2D(i, j)]);
        }
        printf("\n");
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(GRID_DIM, GRID_DIM);
    size_t shmem_size = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float);

    testHaloKernel<<<gridDim, blockDim, shmem_size>>>(d_h, d_result, nx, ny);
    cudaMemcpy(h_result, d_result, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nAfter halo exchange (global view):\n");
    for (int i = 0; i < ny + 2; ++i) {
        for (int j = 0; j < nx + 2; ++j) {
            printf("%5.1f ", h_result[IDX2D(i, j)]);
        }
        printf("\n");
    }

    printf("\nPer-block structured views:\n");
    for (int by = 0; by < GRID_DIM; ++by) {
        for (int bx = 0; bx < GRID_DIM; ++bx) {
            printf("\nBlock (%d, %d):\n", by, bx);
            for (int ty = 0; ty < BLOCK_WITH_HALO; ++ty) {
                for (int tx = 0; tx < BLOCK_WITH_HALO; ++tx) {
                    int gi = by * BLOCK_SIZE + ty;
                    int gj = bx * BLOCK_SIZE + tx;
                    printf("%5.1f ", h_result[IDX2D(gi, gj)]);
                }
                printf("\n");
            }
        }
    }

    cudaFree(d_h);
    cudaFree(d_result);
    free(h_h);
    free(h_result);
    return 0;
}