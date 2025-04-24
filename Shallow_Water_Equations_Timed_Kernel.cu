# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <chrono>
# include <cuda_runtime.h>

# define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

// ************************************************ UTILITIES ************************************************ //

void getArgs(int *nx, double *dt, float *x_length, double *finalRuntime, int argc, char *argv[])
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

__global__ void initializeInterior(float *x, float *y, float *h, float *uh, float *vh, int nx, int ny, float dx, float dy, float x_length)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip ghost
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

__global__ void computeFluxes(float *h, float *uh, float *vh, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, int nx, int ny) 
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  
  unsigned int id = ((i) * (nx + 2) + (j));

  float g = 9.81f; // Gravitational acceleration
  float h_safe = fmaxf(h[id], 1e-6f); // Prevent division by zero
  
  if (i < ny + 2 && j < nx + 2)
  {
    // Compute fluxes safely
    fh[id] = uh[id];

    fuh[id] = uh[id] * uh[id] / h_safe + 0.5f * g * h_safe * h_safe;

    fvh[id] = uh[id] * vh[id] / h_safe;

    gh[id] = vh[id];

    guh[id] = uh[id] * vh[id] / h_safe;

    gvh[id] = vh[id] * vh[id] / h_safe + 0.5f * g * h_safe * h_safe;
  }
}
// ****************************************************************************** //

__global__ void computeVariables(float *hm, float *uhm, float *vhm, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, float *h, float *uh, float *vh, float lambda_x, float lambda_y, int nx, int ny)
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int id, id_left, id_right, id_bottom, id_top;

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
    id = ((i) * (nx + 2) + (j));

    id_left   = ((i) * (nx + 2) + (j - 1));
    id_right  = ((i) * (nx + 2) + (j + 1));
    id_bottom = ((i - 1) * (nx + 2) + (j));
    id_top    = ((i + 1) * (nx + 2) + (j));

    hm[id] = 0.25 * (h[id_left] + h[id_right] + h[id_bottom] + h[id_top])
          - lambda_x * (fh[id_right] - fh[id_left])
          - lambda_y * (gh[id_top] - gh[id_bottom]);

    uhm[id] = 0.25 * (uh[id_left] + uh[id_right] + uh[id_bottom] + uh[id_top])
            - lambda_x * (fuh[id_right] - fuh[id_left])
            - lambda_y * (guh[id_top] - guh[id_bottom]);

    vhm[id] = 0.25 * (vh[id_left] + vh[id_right] + vh[id_bottom] + vh[id_top])
            - lambda_x * (fvh[id_right] - fvh[id_left])
            - lambda_y * (gvh[id_top] - gvh[id_bottom]);
  }
}
// ****************************************************************************** //

__global__ void updateVariables(float *h, float *uh, float *vh, float *hm, float *uhm, float *vhm, int nx, int ny)
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int id;

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
    id = ((i) * (nx + 2) + (j));

    h[id] = hm[id];
    uh[id] = uhm[id];
    vh[id] = vhm[id];
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
    h[id]  = h[id_interior];
    uh[id] = -uh[id_interior];
    vh[id] =  vh[id_interior];
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
    h[id]  = h[id_interior];
    uh[id] = -uh[id_interior];
    vh[id] =  vh[id_interior];
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
    h[id]  = h[id_interior];
    uh[id] =  uh[id_interior];
    vh[id] = -vh[id_interior];
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
    h[id]  = h[id_interior];
    uh[id] =  uh[id_interior];
    vh[id] = -vh[id_interior];
  }
}
// ****************************************************************************** //

// ****************************************************** MAIN ****************************************************** //
int main ( int argc, char *argv[] )
{ 
  // ************************************************** INSTANTIATION ************************************************* //
  int k, l;

  int nx; 
  int ny; 

  float dx;
  float dy;
  
  float x_length;

  double dt;
  double programRuntime; 
  double finalRuntime;
  
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

  float lambda_x = 0.5  * (float) dt / dx;
  float lambda_y = 0.5 * (float) dt / dy;

  // Define the block and grid sizes
  int dimx = 32;
  int dimy = 32;
  dim3 blockSize(dimx, dimy);
  dim3 gridSize((nx + 2 + blockSize.x - 1) / blockSize.x, (ny + 2 + blockSize.y - 1) / blockSize.y);

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
  
  for(k = 1; k < 11; k++)
  {
    // set initial time & step counter
    programRuntime = 0.0f;
    l = 0;
    
    // instantiate section timing variables
    double time_elapsed_cf = 0.0;
    double time_elapsed_cv = 0.0;
    double time_elapsed_uv = 0.0;
    double time_elapsed_bc = 0.0;

    // Apply the initial conditions.
    initializeInterior<<<gridSize, blockSize>>>(d_x, d_y, d_h, d_uh, d_vh, nx, ny, dx, dy, x_length);

    applyLeftBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyRightBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyBottomBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    applyTopBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

    // ******************************************************************** COMPUTATION SECTION ******************************************************************** //

    // start program timer
    auto start_time = std::chrono::steady_clock::now();

    while (programRuntime < finalRuntime) // time loop begins
    {
      // Take a time step and increase step counter
      programRuntime += dt;
      l++;

      // *********************************************************************************************************************************************************** //

      // Start timing compute fluxes calculations
      auto start_time_cf = std::chrono::steady_clock::now();

      // **** COMPUTE FLUXES ****
      computeFluxes<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, nx, ny);

      // Stop timing compute fluxes calculations
      auto end_time_cf = std::chrono::steady_clock::now();

      // calculate time elapsed for compute fluxes
      time_elapsed_cf = time_elapsed_cf + std::chrono::duration<double>(end_time_cf - start_time_cf).count();

      // *********************************************************************************************************************************************************** //

      // Start timing compute variable calculations
      auto start_time_cv = std::chrono::steady_clock::now();
      
      // **** COMPUTE VARIABLES ****
      computeVariables<<<gridSize, blockSize>>>(d_hm, d_uhm, d_vhm, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, d_h, d_uh, d_vh, lambda_x, lambda_y, nx, ny);
    
      // Stop timing compute variable calculations
      auto end_time_cv = std::chrono::steady_clock::now();

      // calculate time elapsed for compute variables
      time_elapsed_cv = time_elapsed_cv + std::chrono::duration<double>(end_time_cv - start_time_cv).count();

      // *********************************************************************************************************************************************************** //

      // Start timing update variables calculations
      auto start_time_uv = std::chrono::steady_clock::now();

      // **** UPDATE VARIABLES ****
      updateVariables<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_hm, d_uhm, d_vhm, nx, ny);

      // Stop timing update variables calculations
      auto end_time_uv = std::chrono::steady_clock::now();

      // calculate time elapsed for update variables
      time_elapsed_uv = time_elapsed_uv + std::chrono::duration<double>(end_time_uv - start_time_uv).count();

      // *********************************************************************************************************************************************************** //

      // Start timing apply boundary condition calculations
      auto start_time_bc = std::chrono::steady_clock::now();

      // **** APPLY BOUNDARY CONDITIONS ****
      applyLeftBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

      applyRightBoundary<<<gridSizeY, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

      applyBottomBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny);

      applyTopBoundary<<<gridSizeX, boundaryBlockSize>>>(d_h, d_uh, d_vh, nx, ny); 

      // Stop timing apply boundary condition calculations
      auto end_time_bc = std::chrono::steady_clock::now();

      // calculate time elapsed for apply boundary conditions
      time_elapsed_bc = time_elapsed_bc + std::chrono::duration<double>(end_time_bc - start_time_bc).count();
    } 

    // stop timer
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = end_time - start_time;
    
    double avg_time_elapsed_cf = time_elapsed_cf / (double) l;
    double avg_time_elapsed_cv = time_elapsed_cv / (double) l;
    double avg_time_elapsed_uv = time_elapsed_uv / (double) l;
    double avg_time_elapsed_bc = time_elapsed_bc / (double) l;

    // Print out the results
    printf("Problem size: %d, Time steps: %d, Iteration: %d, Elapsed time: %f s, Average elapsed time for compute fluxes: %f s, Average elapsed time for compute variables: %f s, Average elapsed time for update variables: %f s, Average elapsed time for apply boundary conditions: %f s\n", nx, l, k, time_elapsed, avg_time_elapsed_cf, avg_time_elapsed_cv, avg_time_elapsed_uv, avg_time_elapsed_bc);
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