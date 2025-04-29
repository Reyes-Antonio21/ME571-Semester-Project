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
    int id = i * (nx + 2);
    int id_interior = i * (nx + 2) + 1;

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

__device__ void haloExchange(float* sh_h, float* sh_uh, float* sh_vh, const float* h, const float* uh, const float* vh, int i, int j, int local_i, int local_j, int nx, int ny, int blockDim_x)
{
  #define SH_ID(i, j, blockDim_x) ((i) * (blockDim_x + 2) + (j))
  #define ID_2D(i, j, nx) ((i) * (nx + 2) + (j))

  int global_id, halo_global_id;
  int local_halo_id;

  // === LEFT Halo ===
  if (threadIdx.x == 0 && j > 0)
  {
    halo_global_id = ID_2D(i, j - 1, nx);
    local_halo_id  = SH_ID(local_i, local_j - 1, blockDim_x);

    sh_h[local_halo_id]  = h[halo_global_id];
    sh_uh[local_halo_id] = uh[halo_global_id];
    sh_vh[local_halo_id] = vh[halo_global_id];
  }
  else if (threadIdx.x == 0 && j == 0)
  {
    global_id = ID_2D(i, j, nx);
    local_halo_id = SH_ID(local_i, local_j - 1, blockDim_x);

    sh_h[local_halo_id]  = h[global_id];
    sh_uh[local_halo_id] = -uh[global_id];
    sh_vh[local_halo_id] =  vh[global_id];
  }

  // === RIGHT Halo ===
  if (threadIdx.x == blockDim.x - 1 && j < nx + 1)
  {
    halo_global_id = ID_2D(i, j + 1, nx);
    local_halo_id  = SH_ID(local_i, local_j + 1, blockDim_x);

    sh_h[local_halo_id]  = h[halo_global_id];
    sh_uh[local_halo_id] = uh[halo_global_id];
    sh_vh[local_halo_id] = vh[halo_global_id];
  }
  else if (threadIdx.x == blockDim.x - 1 && j == nx + 1)
  {
    global_id = ID_2D(i, j, nx);
    local_halo_id = SH_ID(local_i, local_j + 1, blockDim_x);

    sh_h[local_halo_id]  = h[global_id];
    sh_uh[local_halo_id] = -uh[global_id];
    sh_vh[local_halo_id] =  vh[global_id];
  }

  // === BOTTOM Halo ===
  if (threadIdx.y == 0 && i > 0)
  {
    halo_global_id = ID_2D(i - 1, j, nx);
    local_halo_id  = SH_ID(local_i - 1, local_j, blockDim_x);

    sh_h[local_halo_id]  = h[halo_global_id];
    sh_uh[local_halo_id] = uh[halo_global_id];
    sh_vh[local_halo_id] = vh[halo_global_id];
  }
  else if (threadIdx.y == 0 && i == 0)
  {
    global_id = ID_2D(i, j, nx);
    local_halo_id = SH_ID(local_i - 1, local_j, blockDim_x);

    sh_h[local_halo_id]  = h[global_id];
    sh_uh[local_halo_id] =  uh[global_id];
    sh_vh[local_halo_id] = -vh[global_id];
  }

  // === TOP Halo ===
  if (threadIdx.y == blockDim.y - 1 && i < ny + 1)
  {
    halo_global_id = ID_2D(i + 1, j, nx);
    local_halo_id  = SH_ID(local_i + 1, local_j, blockDim_x);

    sh_h[local_halo_id]  = h[halo_global_id];
    sh_uh[local_halo_id] = uh[halo_global_id];
    sh_vh[local_halo_id] = vh[halo_global_id];
  }
  else if (threadIdx.y == blockDim.y - 1 && i == ny + 1)
  {
    global_id = ID_2D(i, j, nx);
    local_halo_id = SH_ID(local_i + 1, local_j, blockDim_x);

    sh_h[local_halo_id]  = h[global_id];
    sh_uh[local_halo_id] =  uh[global_id];
    sh_vh[local_halo_id] = -vh[global_id];
  }

  #undef ID_2D
  #undef SH_ID
}


__global__ void shallowWaterSolver(float *__restrict__ h, float *__restrict__ uh, float *__restrict__ vh, float lambda_x, float lambda_y, int nx, int ny, float dt, float finalRuntime)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int local_i = threadIdx.y + 1;
  unsigned int local_j = threadIdx.x + 1;

  unsigned int id;
  unsigned int local_id, local_id_left, local_id_right, local_id_bottom, local_id_top;

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

  # define SH_ID(i, j, blockDim_x) ((i) * (blockDim_x + 2) + (j))
  # define ID_2D(i, j, nx) ((i) * (nx + 2) + (j))

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);
    local_id = SH_ID(local_i, local_j, blockDim.x);

    sh_h[local_id]  = h[id];
    sh_uh[local_id] = uh[id];
    sh_vh[local_id] = vh[id];
  }

  haloExchange(sh_h, sh_uh, sh_vh, h, uh, vh, i, j, local_i, local_j, nx, ny, blockDim.x);

  __syncthreads();

  float programRuntime = 0.0f;
  float g = 9.81f;
  float g_half = 0.5f * g;

  while (programRuntime < finalRuntime)
  {
    haloExchange(sh_h, sh_uh, sh_vh, h, uh, vh, i, j, local_i, local_j, nx, ny, blockDim.x);

    __syncthreads();

    if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
    {
      local_id = SH_ID(local_i, local_j, blockDim.x);

      float h_val  = sh_h[local_id];
      float uh_val = sh_uh[local_id];
      float vh_val = sh_vh[local_id];

      float inv_h = 1.0f / h_val;
      float h2 = h_val * h_val;

      sh_fh[local_id] = uh_val;
      sh_gh[local_id] = vh_val;

      float uh2 = uh_val * uh_val;
      float vh2 = vh_val * vh_val;
      float uv = uh_val * vh_val;

      sh_fuh[local_id] = __fmaf_rn(uh2, inv_h, g_half * h2);
      sh_fvh[local_id] = uv * inv_h;           

      sh_guh[local_id] = uv * inv_h;           
      sh_gvh[local_id] = __fmaf_rn(vh2, inv_h, g_half * h2);
    }

    __syncthreads();

    if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
    {
      local_id       = SH_ID(local_i, local_j, blockDim.x);
      local_id_left  = SH_ID(local_i, local_j - 1, blockDim.x);
      local_id_right = SH_ID(local_i, local_j + 1, blockDim.x);
      local_id_bottom = SH_ID(local_i - 1, local_j, blockDim.x);
      local_id_top    = SH_ID(local_i + 1, local_j, blockDim.x);

      float h_l  = sh_h[local_id_left];
      float h_r  = sh_h[local_id_right];
      float h_b  = sh_h[local_id_bottom];
      float h_t  = sh_h[local_id_top];

      float uh_l = sh_uh[local_id_left];
      float uh_r = sh_uh[local_id_right];
      float uh_b = sh_uh[local_id_bottom];
      float uh_t = sh_uh[local_id_top];

      float vh_l = sh_vh[local_id_left];
      float vh_r = sh_vh[local_id_right];
      float vh_b = sh_vh[local_id_bottom];
      float vh_t = sh_vh[local_id_top];

      float fh_l = sh_fh[local_id_left];
      float fh_r = sh_fh[local_id_right];
      float gh_b = sh_gh[local_id_bottom];
      float gh_t = sh_gh[local_id_top];

      float fuh_l = sh_fuh[local_id_left];
      float fuh_r = sh_fuh[local_id_right];
      float guh_b = sh_guh[local_id_bottom];
      float guh_t = sh_guh[local_id_top];

      float fvh_l = sh_fvh[local_id_left];
      float fvh_r = sh_fvh[local_id_right];
      float gvh_b = sh_gvh[local_id_bottom];
      float gvh_t = sh_gvh[local_id_top];

      sh_h[local_id] = __fmaf_rn(-lambda_x, (fh_r - fh_l), __fmaf_rn(-lambda_y, (gh_t - gh_b), 0.25f * (h_l + h_r + h_b + h_t)));
      sh_uh[local_id] = __fmaf_rn(-lambda_x, (fuh_r - fuh_l), __fmaf_rn(-lambda_y, (guh_t - guh_b), 0.25f * (uh_l + uh_r + uh_b + uh_t)));
      sh_vh[local_id] = __fmaf_rn(-lambda_x, (fvh_r - fvh_l), __fmaf_rn(-lambda_y, (gvh_t - gvh_b), 0.25f * (vh_l + vh_r + vh_b + vh_t)));
    }

    __syncthreads();

    // Correct boundary condition application using shared memory coordinates
    if (local_j == 0 && local_i > 0 && local_i < blockDim.y + 1)
    {
      sh_h[SH_ID(local_i, local_j, blockDim.x)] = sh_h[SH_ID(local_i, local_j + 1, blockDim.x)];
      sh_uh[SH_ID(local_i, local_j, blockDim.x)] = -sh_uh[SH_ID(local_i, local_j + 1, blockDim.x)];
      sh_vh[SH_ID(local_i, local_j, blockDim.x)] = sh_vh[SH_ID(local_i, local_j + 1, blockDim.x)];
    }

    if (local_j == blockDim.x + 1 && local_i > 0 && local_i < blockDim.y + 1)
    {
      sh_h[SH_ID(local_i, local_j, blockDim.x)] = sh_h[SH_ID(local_i, local_j - 1, blockDim.x)];
      sh_uh[SH_ID(local_i, local_j, blockDim.x)] = -sh_uh[SH_ID(local_i, local_j - 1, blockDim.x)];
      sh_vh[SH_ID(local_i, local_j, blockDim.x)] = sh_vh[SH_ID(local_i, local_j - 1, blockDim.x)];
    }

    if (local_i == 0 && local_j > 0 && local_j < blockDim.x + 1)
    {
      sh_h[SH_ID(local_i, local_j, blockDim.x)] = sh_h[SH_ID(local_i + 1, local_j, blockDim.x)];
      sh_uh[SH_ID(local_i, local_j, blockDim.x)] = sh_uh[SH_ID(local_i + 1, local_j, blockDim.x)];
      sh_vh[SH_ID(local_i, local_j, blockDim.x)] = -sh_vh[SH_ID(local_i + 1, local_j, blockDim.x)];
    }

    if (local_i == blockDim.y + 1 && local_j > 0 && local_j < blockDim.x + 1)
    {
      sh_h[SH_ID(local_i, local_j, blockDim.x)] = sh_h[SH_ID(local_i - 1, local_j, blockDim.x)];
      sh_uh[SH_ID(local_i, local_j, blockDim.x)] = sh_uh[SH_ID(local_i - 1, local_j, blockDim.x)];
      sh_vh[SH_ID(local_i, local_j, blockDim.x)] = -sh_vh[SH_ID(local_i - 1, local_j, blockDim.x)];
    }

    __syncthreads();

    programRuntime += dt;
  }

  // Final write-back to global memory
  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);
    local_id = SH_ID(local_i, local_j, blockDim.x);

    h[id]  = sh_h[local_id];
    uh[id] = sh_uh[local_id];
    vh[id] = sh_vh[local_id];
  }
}
// ****************************************************************************************************************** //

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
  size_t sharedMemSize = ((10 * (blockSize.x+2) * (blockSize.y+2) * sizeof(float)) + 127) & ~127;

  cudaFuncSetAttribute(shallowWaterSolver, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

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

    shallowWaterSolver<<<gridSize, blockSize, sharedMemSize>>>(d_h, d_uh, d_vh, lambda_x, lambda_y, nx, ny, dt, finalRuntime);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
      printf("CUDA Error launching shallowWaterSolver: %s\n", cudaGetErrorString(err));
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