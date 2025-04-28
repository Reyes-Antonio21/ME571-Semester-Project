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

__global__ void shallowWaterSolverHaloExchange(float *h, float *uh, float *vh,
  float lambda_x, float lambda_y,
  int nx, int ny,
  float dt, float finalRuntime)
{
  extern __shared__ float sharedmemory[];

  float *sh_h   = sharedmemory;
  float *sh_uh  = sh_h   + (blockDim.y+2)*(blockDim.x+2);
  float *sh_vh  = sh_uh  + (blockDim.y+2)*(blockDim.x+2);

  float *sh_hm  = sh_vh  + (blockDim.y+2)*(blockDim.x+2);
  float *sh_uhm = sh_hm  + (blockDim.y+2)*(blockDim.x+2);
  float *sh_vhm = sh_uhm + (blockDim.y+2)*(blockDim.x+2);

  #define SH_ID(i,j) ((i)*(blockDim.x+2)+(j))
  #define GID(i,j) ((i)*(nx+2)+(j)) // assume global h,uh,vh have ghost layer

  int global_i = blockIdx.y * blockDim.y + threadIdx.y;
  int global_j = blockIdx.x * blockDim.x + threadIdx.x;

  int local_i = threadIdx.y + 1;
  int local_j = threadIdx.x + 1;

  bool has_left   = (blockIdx.x > 0);
  bool has_right  = (blockIdx.x < gridDim.x - 1);
  bool has_bottom = (blockIdx.y < gridDim.y - 1);
  bool has_top    = (blockIdx.y > 0);

  float programRuntime = 0.0f;
  float g = 9.81f;
  float g_half = 0.5f * g;

  while (programRuntime < finalRuntime)
  {
  // === Load interior ===
  if (global_i < ny && global_j < nx)
  {
  int gid = GID(global_i+1, global_j+1);
  int lid = SH_ID(local_i, local_j);

  sh_h[lid]  = h[gid];
  sh_uh[lid] = uh[gid];
  sh_vh[lid] = vh[gid];
  }

  // === Load halos ===
  if (threadIdx.x == 0 && has_left && global_i < ny)
  {
  int gid = GID(global_i+1, global_j);
  int lid = SH_ID(local_i, 0);

  sh_h[lid]  = h[gid];
  sh_uh[lid] = uh[gid];
  sh_vh[lid] = vh[gid];
  }
  if (threadIdx.x == blockDim.x-1 && has_right && global_i < ny)
  {
  int gid = GID(global_i+1, global_j+2);
  int lid = SH_ID(local_i, blockDim.x+1);

  sh_h[lid]  = h[gid];
  sh_uh[lid] = uh[gid];
  sh_vh[lid] = vh[gid];
  }
  if (threadIdx.y == 0 && has_bottom && global_j < nx)
  {
  int gid = GID(global_i, global_j+1);
  int lid = SH_ID(0, local_j);

  sh_h[lid]  = h[gid];
  sh_uh[lid] = uh[gid];
  sh_vh[lid] = vh[gid];
  }
  if (threadIdx.y == blockDim.y-1 && has_top && global_j < nx)
  {
  int gid = GID(global_i+2, global_j+1);
  int lid = SH_ID(blockDim.y+1, local_j);

  sh_h[lid]  = h[gid];
  sh_uh[lid] = uh[gid];
  sh_vh[lid] = vh[gid];
  }

  __syncthreads();

  // === Compute fluxes and update ===
  if (global_i < ny && global_j < nx)
  {
  int lid = SH_ID(local_i, local_j);
  int lid_l = SH_ID(local_i, local_j-1);
  int lid_r = SH_ID(local_i, local_j+1);
  int lid_b = SH_ID(local_i-1, local_j);
  int lid_t = SH_ID(local_i+1, local_j);

  // Lax-Friedrichs style update
  sh_hm[lid]  = 0.25f * (sh_h[lid_l] + sh_h[lid_r] + sh_h[lid_b] + sh_h[lid_t])
  - lambda_x * (sh_uh[lid_r] - sh_uh[lid_l])
  - lambda_y * (sh_vh[lid_t] - sh_vh[lid_b]);

  sh_uhm[lid] = 0.25f * (sh_uh[lid_l] + sh_uh[lid_r] + sh_uh[lid_b] + sh_uh[lid_t])
  - lambda_x * ((sh_uh[lid_r]*sh_uh[lid_r]/sh_h[lid_r] + 0.5f*g*sh_h[lid_r]*sh_h[lid_r]) -
  (sh_uh[lid_l]*sh_uh[lid_l]/sh_h[lid_l] + 0.5f*g*sh_h[lid_l]*sh_h[lid_l]))
  - lambda_y * ((sh_uh[lid_t]*sh_vh[lid_t]/sh_h[lid_t]) -
  (sh_uh[lid_b]*sh_vh[lid_b]/sh_h[lid_b]));

  sh_vhm[lid] = 0.25f * (sh_vh[lid_l] + sh_vh[lid_r] + sh_vh[lid_b] + sh_vh[lid_t])
  - lambda_x * ((sh_uh[lid_r]*sh_vh[lid_r]/sh_h[lid_r]) -
  (sh_uh[lid_l]*sh_vh[lid_l]/sh_h[lid_l]))
  - lambda_y * ((sh_vh[lid_t]*sh_vh[lid_t]/sh_h[lid_t] + 0.5f*g*sh_h[lid_t]*sh_h[lid_t]) -
  (sh_vh[lid_b]*sh_vh[lid_b]/sh_h[lid_b]));
  }

  __syncthreads();

  // === Swap updated values ===
  if (global_i < ny && global_j < nx)
  {
  int lid = SH_ID(local_i, local_j);

  sh_h[lid]  = sh_hm[lid];
  sh_uh[lid] = sh_uhm[lid];
  sh_vh[lid] = sh_vhm[lid];
  }

  __syncthreads();

  programRuntime += dt;
  }

  // === Store back final values ===
  if (global_i < ny && global_j < nx)
  {
  int gid = GID(global_i+1, global_j+1);
  int lid = SH_ID(local_i, local_j);

  h[gid]  = sh_h[lid];
  uh[gid] = sh_uh[lid];
  vh[gid] = sh_vh[lid];
  }

  #undef SH_ID
  #undef GID
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