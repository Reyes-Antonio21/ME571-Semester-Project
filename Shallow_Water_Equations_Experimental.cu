# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <chrono>
# include <cuda_runtime.h>

// ************************************************ UTILITIES ************************************************ //

void getArgs(int *nx, float *dt, float *x_length, float *finalRuntime, int argc, char *argv[])
{
  // Get the quadrature file root name:

  if ( argc <= 1 ){
    *nx = 400;
  }else{
    *nx = atoi ( argv[1] );
  }
  
  if ( argc <= 2 ){
    *dt = 0.002;
  }else{
    *dt = atof ( argv[2] );
  }
  
  if ( argc <= 3 ){
    *x_length = 10.0;
  }else{
    *x_length = atof ( argv[3] );
    }
  
  if ( argc <= 4 ){
    *finalRuntime = 0.5;
  }else{
    *finalRuntime = atof ( argv[4] );
  }
}
// ****************************************************************************** //

void writeResults(float h[], float uh[], float vh[], float x[], float y[], float time, int nx, int ny)
{
  char filename[50];

  int i, j, id;

  //Create the filename based on the time step.
  sprintf(filename, "tc2d_%08.6f.dat", time);

  //Open the file.
  FILE *file = fopen (filename, "wt" );
    
  if (!file)
  {
    fprintf (stderr, "\n" );

    fprintf (stderr, "WRITE_RESULTS - Fatal error!\n");

    fprintf (stderr, "  Could not open the output file.\n");

    exit (1);
  }

  else
  {  
    //Write the data.
    for ( i = 0; i < ny; i++ ) 
      for ( j = 0; j < nx; j++ )
      {
        id = ((i + 1)*(nx + 2)+(j + 1));
        fprintf ( file, "%24.16g\t%24.16g\t%24.16g\t %24.16g\t %24.16g\n", x[j], y[i], h[id], uh[id], vh[id]);
      }
    
    //Close the file.
    fclose (file);
  }

  return;
}
// ****************************************************************************** //

__global__ void initializeInterior(float *x, float *y, float *h, float *uh, float *vh, int nx, int ny, float dx, float dy, float x_length)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (i < ny + 1 && j < nx + 1)
  {
    int id = i * (nx + 2) + j;

    float xx = -x_length / 2.0f + dx / 2.0f + (j - 1) * dx;
    float yy = -x_length / 2.0f + dy / 2.0f + (i - 1) * dy;

    x[j - 1] = xx;
    y[i - 1] = yy;

    h[id] += 1.0f + 0.40f * expf(-5.0f * (xx * xx + yy * yy));
  }
}
// ****************************************************************************** //

__global__ void applyLeftBoundary(float *h, float *uh, float *vh, int nx, int ny)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0 && i < ny + 1)
  {
    int nx_ext = nx + 2;
    int id = i * nx_ext;
    int id_interior = i * nx_ext + 1;

    float h_val = h[id_interior];
    float uh_val = uh[id_interior];
    float vh_val = vh[id_interior];

    h[id]  = h_val;
    uh[id] = -uh_val;
    vh[id] =  vh_val;
  }
}
// ****************************************************************************** //

__global__ void applyRightBoundary(float *h, float *uh, float *vh, int nx, int ny)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0 && i < ny + 1)
  {
    int nx_ext = nx + 2;
    int id = i * nx_ext + (nx + 1);
    int id_interior = i * nx_ext + nx;

    float h_val = h[id_interior];
    float uh_val = uh[id_interior];
    float vh_val = vh[id_interior];

    h[id]  = h_val;
    uh[id] = -uh_val;
    vh[id] =  vh_val;
  }
}
// ****************************************************************************** //

__global__ void applyBottomBoundary(float *h, float *uh, float *vh, int nx, int ny)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j > 0 && j < nx + 1)
  {
    int nx_ext = nx + 2;
    int id = j;
    int id_interior = 1 * nx_ext + j;

    float h_val = h[id_interior];
    float uh_val = uh[id_interior];
    float vh_val = vh[id_interior];

    h[id]  = h_val;
    uh[id] =  uh_val;
    vh[id] = -vh_val;
  }
}
// ****************************************************************************** //

__global__ void applyTopBoundary(float *h, float *uh, float *vh, int nx, int ny)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j > 0 && j < nx + 1)
  {
    int nx_ext = nx + 2;
    int id = (ny + 1) * nx_ext + j;
    int id_interior = ny * nx_ext + j;

    float h_val = h[id_interior];
    float uh_val = uh[id_interior];
    float vh_val = vh[id_interior];

    h[id]  = h_val;
    uh[id] =  uh_val;
    vh[id] = -vh_val;
  }
}
// ****************************************************************************** //

__device__ void applyDomainBoundaryConditionsShared(float *__restrict__ sh_h, float *__restrict__ sh_uh, float *__restrict__ sh_vh, int local_i, int local_j, int i, int j, int nx, int ny, int blockDim_x, int blockDim_y)
{
  #define SH_ID(i,j) ((i)*(blockDim_x+2)+(j))

  unsigned int local_id;

  local_id = SH_ID(local_i, local_j);

  if (i == 0 && j > 0 && j < nx+1) {  // Bottom wall
  sh_h [local_id] = sh_h [SH_ID(local_i+1, local_j)];
  sh_uh[local_id] = -sh_uh[SH_ID(local_i+1, local_j)];
  sh_vh[local_id] =  sh_vh[SH_ID(local_i+1, local_j)];
  }
  if (i == ny+1 && j > 0 && j < nx+1) { // Top wall
  sh_h [local_id] = sh_h [SH_ID(local_i-1, local_j)];
  sh_uh[local_id] = -sh_uh[SH_ID(local_i-1, local_j)];
  sh_vh[local_id] =  sh_vh[SH_ID(local_i-1, local_j)];
  }
  if (j == 0 && i > 0 && i < ny+1) { // Left wall
  sh_h [local_id] = sh_h [SH_ID(local_i, local_j+1)];
  sh_uh[local_id] =  sh_uh[SH_ID(local_i, local_j+1)];
  sh_vh[local_id] = -sh_vh[SH_ID(local_i, local_j+1)];
  }
  if (j == nx+1 && i > 0 && i < ny+1) { // Right wall
  sh_h [local_id] = sh_h [SH_ID(local_i, local_j-1)];
  sh_uh[local_id] =  sh_uh[SH_ID(local_i, local_j-1)];
  sh_vh[local_id] = -sh_vh[SH_ID(local_i, local_j-1)];
  }

  #undef SH_ID
}
// ****************************************************************************** //

__device__ void refreshInternalHalosShared(float *__restrict__ sh_h, float *__restrict__ sh_uh, float *__restrict__ sh_vh, int local_i, int local_j, int i, int j, int nx, int ny, int blockDim_x, int blockDim_y)
{
  #define SH_ID(i,j) ((i) * (blockDim_x + 2) + (j))

  // Left halo (copy first interior column)
  if (threadIdx.x == 0 && j > 0) {
  sh_h [SH_ID(local_i, 0)] = sh_h [SH_ID(local_i, 1)];
  sh_uh[SH_ID(local_i, 0)] = sh_uh[SH_ID(local_i, 1)];
  sh_vh[SH_ID(local_i, 0)] = sh_vh[SH_ID(local_i, 1)];
  }

  // Right halo (copy last interior column)
  if (threadIdx.x == blockDim_x-1 && j < nx+1) {
  sh_h [SH_ID(local_i, blockDim_x+1)] = sh_h [SH_ID(local_i, blockDim_x)];
  sh_uh[SH_ID(local_i, blockDim_x+1)] = sh_uh[SH_ID(local_i, blockDim_x)];
  sh_vh[SH_ID(local_i, blockDim_x+1)] = sh_vh[SH_ID(local_i, blockDim_x)];
  }

  // Bottom halo (copy first interior row)
  if (threadIdx.y == 0 && i > 0) {
  sh_h [SH_ID(0, local_j)] = sh_h [SH_ID(1, local_j)];
  sh_uh[SH_ID(0, local_j)] = sh_uh[SH_ID(1, local_j)];
  sh_vh[SH_ID(0, local_j)] = sh_vh[SH_ID(1, local_j)];
  }

  // Top halo (copy last interior row)
  if (threadIdx.y == blockDim_y-1 && i < ny+1) {
  sh_h [SH_ID(blockDim_y+1, local_j)] = sh_h [SH_ID(blockDim_y, local_j)];
  sh_uh[SH_ID(blockDim_y+1, local_j)] = sh_uh[SH_ID(blockDim_y, local_j)];
  sh_vh[SH_ID(blockDim_y+1, local_j)] = sh_vh[SH_ID(blockDim_y, local_j)];
  }

  #undef SH_ID
}
// ****************************************************************************** //

__global__ void persistentFusedKernel(float *__restrict__ h, float *__restrict__ uh, float *__restrict__ vh, float lambda_x, float lambda_y, int nx, int ny, float dt, float finalRuntime)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int local_i = threadIdx.y + 1;
  unsigned int local_j = threadIdx.x + 1;

  unsigned int id, local_id;
  unsigned int local_id_left, local_id_right, local_id_bottom, local_id_top;

  extern __shared__ float sharedmemory[];

  float *sh_h   = sharedmemory;
  float *sh_uh  = sh_h   + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_vh  = sh_uh  + (blockDim.y + 2) * (blockDim.x + 2);

  float *sh_fh  = sh_vh  + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_gh  = sh_fh  + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_fuh = sh_gh  + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_guh = sh_fuh + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_fvh = sh_guh + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_gvh = sh_fvh + (blockDim.y + 2) * (blockDim.x + 2);

  float *sh_hm  = sh_gvh + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_uhm = sh_hm  + (blockDim.y + 2) * (blockDim.x + 2);
  float *sh_vhm = sh_uhm + (blockDim.y + 2) * (blockDim.x + 2);

  #define SH_ID(i,j) ((i)*(blockDim.x+2)+(j))
  #define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

  // === Load from global into shared (initial)
  if (i < ny+2 && j < nx+2)
  {
    id = ID_2D(i, j, nx);
    local_id = SH_ID(local_i, local_j);

    sh_h[local_id] = h[id];
    sh_uh[local_id] = uh[id];
    sh_vh[local_id] = vh[id];
  }

  __syncthreads();

  applyDomainBoundaryConditionsShared(sh_h, sh_uh, sh_vh, local_i, local_j, i, j, nx, ny, blockDim.x, blockDim.y);

  __syncthreads();

  float programRuntime = 0.0f;
  float g = 9.81f;
  float g_half = 0.5f * g;

  while (programRuntime < finalRuntime)
  {
    // === Refresh halos first
    refreshInternalHalosShared(sh_h, sh_uh, sh_vh, local_i, local_j, i, j, nx, ny);
    
    __syncthreads();

    // === Compute fluxes
    if (i > 0 && i < ny+1 && j > 0 && j < nx+1) 
    {
      local_id = SH_ID(local_i,local_j);

      float h_val  = sh_h [local_id];
      float uh_val = sh_uh[local_id];
      float vh_val = sh_vh[local_id];

      float inv_h = (h_val > 1e-6f) ? 1.0f / h_val : 0.0f;
      float h2 = h_val * h_val;

      sh_fh [local_id] = uh_val;
      sh_gh [local_id] = vh_val;

      sh_fuh[local_id] = uh_val * uh_val * inv_h + g_half * h2;
      sh_fvh[local_id] = uh_val * vh_val * inv_h;

      sh_guh[local_id] = uh_val * vh_val * inv_h;
      sh_gvh[local_id] = vh_val * vh_val * inv_h + g_half * h2;
    }

    __syncthreads();

    // === Compute updated variables
    if (i > 0 && i < ny+1 && j > 0 && j < nx+1) 
    {
      local_id = SH_ID(local_i,local_j);
      local_id_left = SH_ID(local_i, local_j-1);
      local_id_right = SH_ID(local_i, local_j+1);
      local_id_bottom = SH_ID(local_i+1, local_j);
      local_id_top = SH_ID(local_i-1, local_j);

      float fh_left   = sh_fh [local_id_left];
      float fh_right  = sh_fh [local_id_right];
      float gh_bottom = sh_gh [local_id_bottom];
      float gh_top    = sh_gh [local_id_top];

      float fuh_left   = sh_fuh[local_id_left];
      float fuh_right  = sh_fuh[local_id_right];
      float guh_bottom = sh_guh[local_id_bottom];
      float guh_top    = sh_guh[local_id_top];

      float fvh_left   = sh_fvh[local_id_left];
      float fvh_right  = sh_fvh[local_id_right];
      float gvh_bottom = sh_gvh[local_id_bottom];
      float gvh_top    = sh_gvh[local_id_top];

      sh_hm [local_id] = 0.25f * (sh_h[local_id_left] + sh_h[local_id_right] + sh_h[local_id_top] + sh_h[local_id_bottom])
                      - lambda_x * (fh_right - fh_left)
                      - lambda_y * (gh_top - gh_bottom);

      sh_uhm[local_id] = 0.25f * (sh_uh[local_id_left] + sh_uh[local_id_right] + sh_uh[local_id_top] + sh_uh[local_id_bottom])
                      - lambda_x * (fuh_right - fuh_left)
                      - lambda_y * (guh_top - guh_bottom);

      sh_vhm[local_id] = 0.25f * (sh_vh[local_id_left] + sh_vh[local_id_right] + sh_vh[local_id_top] + sh_vh[local_id_bottom])
                      - lambda_x * (fvh_right - fvh_left)
                      - lambda_y * (gvh_top - gvh_bottom);
    }

    __syncthreads();

    // === Swap updated values
    if (i > 0 && i < ny+1 && j > 0 && j < nx+1) 
    {
      local_id = SH_ID(local_i, local_j);

      sh_h [local_id] = sh_hm [local_id];
      sh_uh[local_id] = sh_uhm[local_id];
      sh_vh[local_id] = sh_vhm[local_id];
    }

    __syncthreads();

    // === Write updated values back to global memory
    if (i > 0 && i < ny+1 && j > 0 && j < nx+1)
    {
      id = ID_2D(i, j, nx);
      h[id]  = sh_h[SH_ID(local_i, local_j)];
      uh[id] = sh_uh[SH_ID(local_i, local_j)];
      vh[id] = sh_vh[SH_ID(local_i, local_j)];
    }

    __syncthreads();

    // === Reload from global memory into shared memory
    if (i < ny+2 && j < nx+2)
    {
      id = ID_2D(i, j, nx);
      local_id = SH_ID(local_i, local_j);

      sh_h[local_id] = h[id];
      sh_uh[local_id] = uh[id];
      sh_vh[local_id] = vh[id];
    }

    __syncthreads();

    programRuntime += dt;
  }

  #undef SH_ID
  #undef ID_2D
}

// ****************************************************** MAIN ****************************************************** //
int main ( int argc, char *argv[] )
{ 
  // ************************************************** INSTANTIATION ************************************************* //
  int k;

  int nx; 
  int ny; 

  float dx;
  float dy;
  
  float x_length;

  float dt;
  float finalRuntime;
  
  // pointers to host, device memory 
  float *h, *d_h;
  float *uh, *d_uh;
  float *vh, *d_vh;

  float *fh, *d_fh;
  float *fuh, *d_fuh;
  float *fvh, *d_fvh;

  float *gh, *d_gh;
  float *guh, *d_guh;
  float *gvh, *d_gvh;

  float *hm, *d_hm; 
  float *uhm, *d_uhm;
  float *vhm, *d_vhm;

  float *x, *d_x;
  float *y, *d_y;

  // get command line arguments
  getArgs(&nx, &dt, &x_length, &finalRuntime, argc, argv);
  ny = nx; // we assume the grid is square

  // Define the locations of the nodes and time steps and the spacing.
  dx = x_length / ( float ) ( nx );
  dy = x_length / ( float ) ( nx );

  float lambda_x = 0.5f * dt / dx;
  float lambda_y = 0.5f * dt / dy;

  // Define the block and grid sizes
  int dimx = 32;
  int dimy = 32;
  dim3 blockSize(dimx, dimy);
  dim3 gridSize((nx + 2 + blockSize.x - 1) / blockSize.x, (ny + 2 + blockSize.y - 1) / blockSize.y);

  // Calculate shared memory size
  size_t sharedMemSize = ((12 * (blockSize.x+2) * (blockSize.y+2) * sizeof(float)) + 127) & ~127;

  cudaFuncSetAttribute(persistentFusedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

  int boundaryBlockSize = 1024;
  int gridSizeY = (ny + boundaryBlockSize - 1) / boundaryBlockSize; 
  int gridSizeX = (nx + boundaryBlockSize - 1) / boundaryBlockSize;  

  // ************************************************ MEMORY ALLOCATIONS ************************************************ //

  // **** Allocate memory on host ****
  // Allocate space (nx+2)((nx+2) long, to account for ghosts
  // height array
  h  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  hm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  // x momentum array
  uh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  uhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fuh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  guh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  // y momentum array
  vh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  vhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  // location arrays
  x = ( float * ) malloc ( nx * sizeof ( float ) );
  y = ( float * ) malloc ( ny * sizeof ( float ) );

  // **** Allocate memory on device ****

  // Allocate space (nx+2)((nx+2) long, to account for ghosts
  cudaMalloc((void **)&d_h, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_uh, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_vh, (nx+2) * (ny+2) * sizeof ( float ));

  cudaMalloc((void **)&d_fh, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_fuh, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_fvh, (nx+2) * (ny+2) * sizeof ( float ));

  cudaMalloc((void **)&d_gh, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_guh, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_gvh, (nx+2) * (ny+2) * sizeof ( float ));

  cudaMalloc((void **)&d_hm, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_uhm, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMalloc((void **)&d_vhm, (nx+2) * (ny+2) * sizeof ( float ));

  cudaMalloc((void **)&d_x, nx * sizeof ( float ));
  cudaMalloc((void **)&d_y, ny * sizeof ( float ));

  cudaMemset(d_h, 0, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMemset(d_uh, 0, (nx+2) * (ny+2) * sizeof ( float ));
  cudaMemset(d_vh, 0, (nx+2) * (ny+2) * sizeof ( float ));
  
  // *********************************************************************** INITIAL CONDITIONS ********************************************************************** //

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D\n" );
  printf ( "\n" );
  
  for(k = 1; k < 6; k++)
  {
    // Apply the initial conditions.
    initializeInterior<<<gridSize, blockSize>>>(d_x, d_y, d_h, d_uh, d_vh, nx, ny, dx, dy, x_length);

    applyLeftBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyRightBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyBottomBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyTopBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    if(k == 1 && nx == 200)
    {
      cudaMemcpy(h, d_h, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);
      cudaMemcpy(uh, d_uh, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);
      cudaMemcpy(vh, d_vh, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);

      cudaMemcpy(x, d_x, nx * sizeof ( float ), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, ny * sizeof ( float ), cudaMemcpyDeviceToHost);

      // Write initial condition to a file
      writeResults(h, uh, vh, x, y, 0.000000, nx, ny);
    }

    // ******************************************************************** COMPUTATION SECTION ******************************************************************** //

    // start program timer
    auto start_time = std::chrono::steady_clock::now();

    persistentFusedKernel<<<gridSize, blockSize, sharedMemSize>>>(d_h, d_uh, d_vh, lambda_x, lambda_y, nx, ny, dt, finalRuntime);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
      printf("CUDA Error launching persistentFusedKernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();  // Wait for kernel to finish

    // stop timer
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = end_time - start_time;

    // Print out the results
    printf("Problem size: %d, Iteration: %d, Elapsed time: %f s\n", nx, k, time_elapsed);

    if(k == 1 && nx == 200)
    {
      cudaMemcpy(h, d_h, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);
      cudaMemcpy(uh, d_uh, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);
      cudaMemcpy(vh, d_vh, (nx+2) * (ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost);

      // Write initial condition to a file
      writeResults(h, uh, vh, x, y, 0.500000, nx, ny);
    }
  }

  // ******************************************************************** DEALLOCATE MEMORY ******************************************************************** //

  //Free device memory.
  cudaFree(d_h);
  cudaFree(d_uh);
  cudaFree(d_vh);

  cudaFree(d_fh);
  cudaFree(d_fuh);
  cudaFree(d_fvh);

  cudaFree(d_gh);
  cudaFree(d_guh);
  cudaFree(d_gvh);

  cudaFree(d_hm);
  cudaFree(d_uhm);
  cudaFree(d_vhm);

  cudaFree(d_x);
  cudaFree(d_y);

  // Free host memory.
  free ( h );
  free ( uh );
  free ( vh ); 

  free ( fh );
  free ( fuh );
  free ( fvh );

  free ( gh );
  free ( guh );
  free ( gvh ); 

  free ( x );
  free ( y );

  // Terminate.
  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D:\n" );
  printf ( "Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
// ******************************************************************************************************************************************** //