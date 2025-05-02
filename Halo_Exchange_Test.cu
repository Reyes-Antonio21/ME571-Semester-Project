#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define HALO 1
#define BLOCK_WITH_HALO (BLOCK_SIZE + 2 * HALO)
#define GRID_DIM 3

#define IDX2D(i, j, stride) ((i) * (stride) + (j))

__device__ int SH_ID(int i, int j, int stride) {
    return i * stride + j;
}
__device__ int ID_2D(int i, int j, int stride) {
    return i * stride + j;
}

__device__ void haloExchange(
    float* sh_h, float* sh_uh, float* sh_vh,
    const float* h, const float* uh, const float* vh,
    int i, int j, int local_i, int local_j,
    int nx, int ny, int blockDim_x, int blockDim_y)
{
    int global_stride = nx + 2;
    int sh_stride = blockDim_x + 2;

    // === LEFT ===
    if (local_j == 1) {
        int gid = ID_2D(i, j - 1, global_stride);
        int lid = SH_ID(local_i, local_j - 1, sh_stride);
        if (j - 1 >= 0) {
            sh_h[lid]  = h[gid];
            sh_uh[lid] = uh[gid];
            sh_vh[lid] = vh[gid];
        }
    }

    // === RIGHT ===
    if (local_j == blockDim_x) {
        int gid = ID_2D(i, j + 1, global_stride);
        int lid = SH_ID(local_i, local_j + 1, sh_stride);
        if (j + 1 < nx + 2) {
            sh_h[lid]  = h[gid];
            sh_uh[lid] = uh[gid];
            sh_vh[lid] = vh[gid];
        }
    }

    // === BOTTOM ===
    if (local_i == 1) {
        int gid = ID_2D(i - 1, j, global_stride);
        int lid = SH_ID(local_i - 1, local_j, sh_stride);
        if (i - 1 >= 0) {
            sh_h[lid]  = h[gid];
            sh_uh[lid] = uh[gid];
            sh_vh[lid] = vh[gid];
        }
    }

    // === TOP ===
    if (local_i == blockDim_y) {
        int gid = ID_2D(i + 1, j, global_stride);
        int lid = SH_ID(local_i + 1, local_j, sh_stride);
        if (i + 1 < ny + 2) {
            sh_h[lid]  = h[gid];
            sh_uh[lid] = uh[gid];
            sh_vh[lid] = vh[gid];
        }
    }
}

__global__ void testHaloKernel(float *h, float *h_result, int nx, int ny) {
    int global_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    int local_i = threadIdx.y + 1;
    int local_j = threadIdx.x + 1;

    extern __shared__ float sh_h[];

    int global_stride = nx + 2;
    int sh_stride = blockDim.x + 2;

    int gid = ID_2D(global_i, global_j, global_stride);
    int lid = SH_ID(local_i, local_j, sh_stride);

    sh_h[lid] = h[gid];
    __syncthreads();

    haloExchange(sh_h, h, global_i, global_j, local_i, local_j, nx, ny, blockDim.x, blockDim.y);
    __syncthreads();

    // Write interior
    h_result[gid] = sh_h[lid];

    // Write halo if thread is on boundary
    if (threadIdx.x == 0) {
        int gid_left = ID_2D(global_i, global_j - 1, global_stride);
        int lid_left = SH_ID(local_i, local_j - 1, sh_stride);
        if (global_j - 1 >= 0) h_result[gid_left] = sh_h[lid_left];
    }
    if (threadIdx.x == blockDim.x - 1) {
        int gid_right = ID_2D(global_i, global_j + 1, global_stride);
        int lid_right = SH_ID(local_i, local_j + 1, sh_stride);
        if (global_j + 1 < nx + 2) h_result[gid_right] = sh_h[lid_right];
    }
    if (threadIdx.y == 0) {
        int gid_bottom = ID_2D(global_i - 1, global_j, global_stride);
        int lid_bottom = SH_ID(local_i - 1, local_j, sh_stride);
        if (global_i - 1 >= 0) h_result[gid_bottom] = sh_h[lid_bottom];
    }
    if (threadIdx.y == blockDim.y - 1) {
        int gid_top = ID_2D(global_i + 1, global_j, global_stride);
        int lid_top = SH_ID(local_i + 1, local_j, sh_stride);
        if (global_i + 1 < ny + 2) h_result[gid_top] = sh_h[lid_top];
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
            h_h[IDX2D(i, j, nx + 2)] = (i == 0 || j == 0 || i == ny + 1 || j == nx + 1) ? -1.0f : block_id;
        }
    }

    float *d_h, *d_result;
    cudaMalloc(&d_h, total_size * sizeof(float));
    cudaMalloc(&d_result, total_size * sizeof(float));

    cudaMemcpy(d_h, h_h, total_size * sizeof(float), cudaMemcpyHostToDevice);

    printf("Before halo exchange:\n");
    for (int i = 0; i < ny + 2; ++i) {
        for (int j = 0; j < nx + 2; ++j) {
            printf("%5.1f ", h_h[IDX2D(i, j, nx + 2)]);
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
            printf("%5.1f ", h_result[IDX2D(i, j, nx + 2)]);
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
                    printf("%5.1f ", h_result[IDX2D(gi, gj, nx + 2)]);
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