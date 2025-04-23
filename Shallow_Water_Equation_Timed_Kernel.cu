# include "common.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
#include <chrono>
# include <cuda_runtime.h>

# define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

//************************************************ UTILITIES ************************************************//

void getArgs(int *nx, double *dt, float *x_length, double *t_final, int argc, char *argv[])
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
    *t_final = 0.5;
  }else{
    *t_final = atof ( argv[4] );
  }
}
/******************************************************************************/

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
        id = ID_2D(i + 1, j + 1, nx);
        fprintf ( file, "%24.16g\t%24.16g\t%24.16g\t %24.16g\t %24.16g\n", x[j], y[i], h[id], uh[id], vh[id]);
      }
    
    //Close the file.
    fclose (file);
  }

  return;
}
/******************************************************************************/

void initial_conditions(int nx, int ny, float dx, float dy,  float x_length, float x[],float y[], float h[], float uh[] ,float vh[])
{
  int i,j, id, id1;

  for ( i = 1; i < nx+1; i++ )
    {
      x[i-1] = -x_length/2+dx/2+(i-1)*dx;
      y[i-1] = -x_length/2+dy/2+(i-1)*dy;
    }

  for ( i = 1; i < nx+1; i++ )
    for( j = 1; j < ny+1; j++)
    {
      float xx = x[j-1];
      float yy = y[i-1];
      id=ID_2D(i,j,nx);
      h[id] = 1.0 + 0.4*exp ( -5 * ( xx*xx + yy*yy) );
    }
  
  for ( i = 1; i < nx+1; i++ )
    for( j = 1; j < ny+1; j++)
    {
      id=ID_2D(i,j,nx);
      uh[id] = 0.0;
      vh[id] = 0.0;
    }

  //set boundaries
  //bottom
  i=0;
  for( j = 1; j < nx+1; j++)
    {
      id=ID_2D(i,j,nx);
      id1=ID_2D(i+1,j,nx);

      h[id] = h[id1];
      uh[id] = 0.0;
      vh[id] = 0.0;
    }

  //top
  i=nx+1;
  for( j = 1; j < nx+1; j++)
    {
      id=ID_2D(i,j,nx);
      id1=ID_2D(i-1,j,nx);

      h[id] = h[id1];
      uh[id] = 0.0;
      vh[id] = 0.0;
    }

  //left
  j=0;
  for( i = 1; i < ny+1; i++)
    {
      id=ID_2D(i,j,nx);
      id1=ID_2D(i,j+1,nx);

      h[id] = h[id1];
      uh[id] = 0.0;
      vh[id] = 0.0;
    }

  //right
  j=nx+1;
  for( i = 1; i < ny+1; i++)
    {
      id=ID_2D(i,j,nx);
      id1=ID_2D(i,j-1,nx);

      h[id] = h[id1];
      uh[id] = 0.0;
      vh[id] = 0.0;
    }

  return;
}
/******************************************************************************/

__global__ void computeFluxesGPU(float *h, float *uh, float *vh, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, int nx, int ny) 
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
/******************************************************************************/

__global__ void computeVariablesGPU(float *hm, float *uhm, float *vhm, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, float *h, float *uh, float *vh, float lambda_x, float lambda_y, int nx, int ny)
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  
  unsigned int id = ((i) * (nx + 2) + (j));

  unsigned int id_left   = ((i) * (nx + 2) + (j - 1));
  unsigned int id_right  = ((i) * (nx + 2) + (j + 1));
  unsigned int id_bottom = ((i - 1) * (nx + 2) + (j));
  unsigned int id_top    = ((i + 1) * (nx + 2) + (j));

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
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
/******************************************************************************/

__global__ void updateVariablesGPU(float *h, float *uh, float *vh, float *hm, float *uhm, float *vhm, int nx, int ny)
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int id = ((i) * (nx + 2) + (j));

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
    h[id] = hm[id];
    uh[id] = uhm[id];
    vh[id] = vhm[id];
  }
}
/******************************************************************************/

__global__ void applyBoundaryConditionsGPU(float *h, float *uh, float *vh, int nx, int ny)
{
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int id_ghost;

  unsigned int id = ((i) * (nx + 2) + (j));
  unsigned int id_left   = ((i) * (nx + 2) + (j - 1));
  unsigned int id_right  = ((i) * (nx + 2) + (j + 1));
  unsigned int id_bottom = ((i - 1) * (nx + 2) + (j));
  unsigned int id_top    = ((i + 1) * (nx + 2) + (j));

  // Left Boundary (j = 1) - Reflective
  if (j == 1 && i > 0 && i < ny + 1) 
  {
    h[id_left]  = h[id];
    uh[id_left] = -uh[id];  // Flip x-momentum
    vh[id_left] = vh[id];   
  }

  // Right Boundary (j = nx) - Reflective
  if (j == nx && i > 0 && i < ny + 1) 
  {
    h[id_right]  = h[id];
    uh[id_right] = -uh[id];  // Flip x-momentum
    vh[id_right] = vh[id];   
  }

  // Bottom Boundary (i = 1) - Reflective
  if (i == 1 && j > 0 && j < nx + 1) 
  {
    h[id_bottom]  = h[id];
    uh[id_bottom] = uh[id];   
    vh[id_bottom] = -vh[id];  // Flip y-momentum
  }

  // Top Boundary (i = ny) - Reflective
  if (i == ny && j > 0 && j < nx + 1) 
  {
    h[id_top]  = h[id];
    uh[id_top] = uh[id];   
    vh[id_top] = -vh[id];  // Flip y-momentum
  }
}
// ****************************************************************************************************************** //

// ****************************************************** MAIN ****************************************************** //
int main ( int argc, char *argv[] )
{ 
// ************************************************** INSTANTIATION ************************************************* //
  int k, l;
  int nx; 
  int ny; 

  float *x;
  float *y;

  float dx;
  float dy;
  float x_length;

  double dt;
  double time; 
  double t_final;
  
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

  // get command line arguments
  getArgs(&nx, &dt, &x_length, &t_final, argc, argv);
  ny = nx; // we assume this, does not have to be this way

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
  CHECK(cudaMalloc((void **)&d_h, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_uh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_vh, (nx+2)*(ny+2) * sizeof ( float )));

  CHECK(cudaMalloc((void **)&d_fh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_fuh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_fvh, (nx+2)*(ny+2) * sizeof ( float )));

  CHECK(cudaMalloc((void **)&d_gh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_guh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_gvh, (nx+2)*(ny+2) * sizeof ( float )));

  CHECK(cudaMalloc((void **)&d_hm, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_uhm, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_vhm, (nx+2)*(ny+2) * sizeof ( float )));

  // *********************************************************************** INITIAL CONDITIONS ********************************************************************** //

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D\n" );
  printf ( "\n" );
  
  for(k = 1; k < 11; k++)
  {
    // set time to zero and step counter to zero
    time = 0.0f;
    l = 0;
    
    // instantiate section timing variables
    double time_elapsed_cf = 0.0;
    double time_elapsed_cv = 0.0;
    double time_elapsed_uv = 0.0;
    double time_elapsed_bc = 0.0;

    initial_conditions(nx, ny, dx, dy, x_length, x, y, h, uh, vh);

    // Move data to the device for calculations
    CHECK(cudaMemcpy(d_h, h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_uh, uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vh, vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));

    // ******************************************************************** COMPUTATION SECTION ******************************************************************** //

    // start timer
    auto start_time = std::chrono::steady_clock::now();

    while (time < t_final) // time loop begins
    {
      // Take a time step and increase step counter
      time = time + dt;
      l++;

      // *********************************************************************************************************************************************************** //

      // Start timing compute fluxes calculations
      auto start_time_cf = std::chrono::steady_clock::now();

      // **** COMPUTE FLUXES ****
      computeFluxesGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, nx, ny);

      // Stop timing compute fluxes calculations
      auto end_time_cf = std::chrono::steady_clock::now();

      // calculate time elapsed for compute fluxes
      time_elapsed_cf = time_elapsed_cf + std::chrono::duration<double>(end_time_cf - start_time_cf).count();

      // *********************************************************************************************************************************************************** //

      // Start timing compute variable calculations
      auto start_time_cv = std::chrono::steady_clock::now();
      
      // **** COMPUTE VARIABLES ****
      computeVariablesGPU<<<gridSize, blockSize>>>(d_hm, d_uhm, d_vhm, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, d_h, d_uh, d_vh, lambda_x, lambda_y, nx, ny);
    
      // Stop timing compute variable calculations
      auto end_time_cv = std::chrono::steady_clock::now();

      // calculate time elapsed for compute variables
      time_elapsed_cv = time_elapsed_cv + std::chrono::duration<double>(end_time_cv - start_time_cv).count();
      // *********************************************************************************************************************************************************** //

      // Start timing update variables calculations
      auto start_time_uv = std::chrono::steady_clock::now();

      // **** UPDATE VARIABLES ****
      updateVariablesGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_hm, d_uhm, d_vhm, nx, ny);

      // Stop timing update variables calculations
      auto end_time_uv = std::chrono::steady_clock::now();

      // calculate time elapsed for update variables
      time_elapsed_uv = time_elapsed_uv + std::chrono::duration<double>(end_time_uv - start_time_uv).count();

      // *********************************************************************************************************************************************************** //

      // Start timing apply boundary condition calculations
      auto start_time_bc = std::chrono::steady_clock::now();

      // **** APPLY BOUNDARY CONDITIONS ****
      applyBoundaryConditionsGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, nx, ny);  

      // Stop timing apply boundary condition calculations
      auto end_time_bc = std::chrono::steady_clock::now();

      // calculate time elapsed for apply boundary conditions
      time_elapsed_bc = time_elapsed_bc + std::chrono::duration<double>(end_time_bc - start_time_bc).count();

    } // end time loop

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
  CHECK(cudaFree(d_h));
  CHECK(cudaFree(d_uh));
  CHECK(cudaFree(d_vh));

  CHECK(cudaFree(d_fh));
  CHECK(cudaFree(d_fuh));
  CHECK(cudaFree(d_fvh));

  CHECK(cudaFree(d_gh));
  CHECK(cudaFree(d_guh));
  CHECK(cudaFree(d_gvh));

  CHECK(cudaFree(d_hm));
  CHECK(cudaFree(d_uhm));
  CHECK(cudaFree(d_vhm));

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