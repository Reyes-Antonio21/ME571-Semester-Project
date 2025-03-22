# include "common.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <cuda_runtime.h>

#define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

#define EPSILON 1e-6f  // Small value to prevent division by zero

//************************************************ UTILITIES ************************************************//

void getArgs(int *nx, double *dt, float *x_length, float *t_final, int argc, char *argv[])
{
  // Get the quadrature file root name:

  if ( argc <= 1 ){
    *nx = 401;
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

void initial_conditions ( int nx, int ny, float dx, float dy,  float x_length, float x[],float y[], float h[], float uh[] ,float vh[])
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

__global__ void initialConditionsGPU( int nx, int ny, float dx, float dy,  float x_length, float x[],float y[], float h[], float uh[] ,float vh[])
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int id, id_boundary;

  if (i > 0 && i < ny + 1)
  {
    x[i - 1] = -x_length / 2 + dx / 2 + (i - 1) * dx;
    y[i - 1] = -x_length / 2 + dy / 2 + (i - 1) * dy; 
  }

  if ( i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);

    float xx = x[j - 1];
    float yy = y[i - 1];

    h[id] = 1.0 + 0.4 * exp( -5 * ( xx * xx + yy * yy) );
  }
  
  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);

    uh[id] = 0.0;
    vh[id] = 0.0;
  }

  //set boundaries
  //bottom
  if (i == 0 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);
    id_boundary = ID_2D(i + 1, j, nx);

    h[id] = h[id_boundary];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }

  //top
  if (i == ny + 1 && j > 0 && j < nx + 1)
  {
    id = ID_2D(i, j, nx);
    id_boundary = ID_2D(i - 1, j, nx);

    h[id] = h[id_boundary];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }

  //left
  if ( j == 0 && i > 0 && i < ny + 1)
  {
    id = ID_2D(i, j, nx);
    id_boundary = ID_2D(i, j + 1, nx);

    h[id] = h[id_boundary];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }

  //right
  if (j == nx + 1 && i > 0 && i < ny + 1)
  {
    id = ID_2D(i, j, nx);
    id_boundary = ID_2D(i, j - 1, nx);

    h[id] = h[id_boundary];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }
  return;
}
/******************************************************************************/

__global__ void computeFluxesGPU(float *h, float *uh, float *vh, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, int nx, int ny) 
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i >= nx + 2 || j >= ny + 2) // Bounds check
  return;
  
  unsigned int id = ID_2D(i, j, nx);

  float g = 9.81f; // Gravitational acceleration
  float h_safe = fmaxf(h[id], EPSILON); // Prevent division by zero
  
  // Compute fluxes safely
  fh[id] = uh[id];

  fuh[id] = uh[id] * uh[id] / h_safe + 0.5f * g * h_safe * h_safe;

  fvh[id] = uh[id] * vh[id] / h_safe;

  gh[id] = vh[id];

  guh[id] = uh[id] * vh[id] / h_safe;

  gvh[id] = vh[id] * vh[id] / h_safe + 0.5f * g * h_safe * h_safe;
  
}
/******************************************************************************/

__global__ void computeVariablesGPU(float *hm, float *uhm, float *vhm, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, float *h, float *uh, float *vh, float lambda_x, float lambda_y, int nx, int ny)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int id, id_left, id_right, id_bottom, id_top;

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
    id = ID_2D(i, j, nx);

    id_left   = ID_2D(i, j - 1, nx);
    id_right  = ID_2D(i, j + 1, nx);
    id_bottom = ID_2D(i - 1, j, nx);
    id_top    = ID_2D(i + 1, j, nx);

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
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int id;

  if (i > 0 && i < ny + 1 && j > 0 && j < nx + 1)  // Ensure proper bounds
  {
    id = ID_2D(i, j, nx);

    h[id] = hm[id];
    uh[id] = uhm[id];
    vh[id] = vhm[id];
  }
}
/******************************************************************************/

__global__ void applyBoundaryConditionsGPU(float *h, float *uh, float *vh, int nx, int ny, int bc_type)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int id, id_ghost;

  if (bc_type == 1) // Dirichlet Boundary Conditions
  {  
    // Left Boundary (j = 0)
    if (j == 0 && i >= 1 && i <= ny) 
    {
      id = ID_2D(i, 1, nx);
      id_ghost = ID_2D(i, 0, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Right Boundary (j = nx + 1)
    if (j == nx + 1 && i >= 1 && i <= ny) 
    {
      id = ID_2D(i, nx, nx);
      id_ghost = ID_2D(i, nx + 1, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Bottom Boundary (i = 0)
    if (i == 0 && j >= 1 && j <= nx) 
    {
      id = ID_2D(1, j, nx);
      id_ghost = ID_2D(0, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Top Boundary (i = ny + 1)
    if (i == ny + 1 && j >= 1 && j <= nx) 
    {
      id = ID_2D(ny, j, nx);
      id_ghost = ID_2D(ny + 1, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }
  }

  else if (bc_type == 2) // Periodic Boundary Conditions
  {  
    // Left to Right Periodic Boundary (wraps leftmost to rightmost)
    if (j == 0 && i >= 1 && i <= ny) 
    {
      id = ID_2D(i, nx, nx);
      id_ghost = ID_2D(i, 0, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Right to Left Periodic Boundary (wraps rightmost to leftmost)
    if (j == nx + 1 && i >= 1 && i <= ny) 
    {
      id = ID_2D(i, 1, nx);
      id_ghost = ID_2D(i, nx + 1, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Bottom to Top Periodic Boundary (wraps bottom to top)
    if (i == 0 && j >= 1 && j <= nx) 
    {
      id = ID_2D(ny, j, nx);
      id_ghost = ID_2D(0, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }

    // Top to Bottom Periodic Boundary (wraps top to bottom)
    if (i == ny + 1 && j >= 1 && j <= nx) 
    {
      id = ID_2D(1, j, nx);
      id_ghost = ID_2D(ny + 1, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];
      vh[id_ghost] = vh[id];
    }  
  }
  else if (bc_type == 3) // Reflective Boundary Conditions
  {  
    // Left Boundary (j = 1) - Reflective
    if (j == 1 && i > 0 && i < ny + 1) 
    {
      id = ID_2D(i, j, nx);
      id_ghost = ID_2D(i, j - 1, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = -uh[id];  // Flip normal velocity
      vh[id_ghost] = vh[id];   // Keep tangential velocity
    }

    // Right Boundary (j = nx) - Reflective
    if (j == nx && i > 0 && i < ny + 1) 
    {
      id = ID_2D(i, j, nx);
      id_ghost = ID_2D(i, j + 1, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = -uh[id];  // Flip normal velocity
      vh[id_ghost] = vh[id];   // Keep tangential velocity
    }

    // Bottom Boundary (i = 1) - Reflective
    if (i == 1 && j > 0 && j < nx + 1) 
    {
      id = ID_2D(i, j, nx);
      id_ghost = ID_2D(i - 1, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];   // Keep tangential velocity
      vh[id_ghost] = -vh[id];  // Flip normal velocity
    }

    // Top Boundary (i = ny) - Reflective
    if (i == ny && j > 0 && j < nx + 1) 
    {
      id = ID_2D(i, j, nx);
      id_ghost = ID_2D(i + 1, j, nx);
      h[id_ghost]  = h[id];
      uh[id_ghost] = uh[id];   // Keep tangential velocity
      vh[id_ghost] = -vh[id];  // Flip normal velocity
    }
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

  float *x, *d_x;
  float *y, *d_y;

  float dx;
  float dy;
  double dt;
  float x_length;

  double time; 
  float t_final;

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

  CHECK(cudaMalloc((void **)&d_x, nx * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_y, ny * sizeof ( float )));

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

  // ************************************************ INITIAL CONDITIONS ************************************************ //

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D\n" );
  printf ( "\n" );

  // set initial time & step counter
  // set time to zero and step counter to zero
  time = 0.0f;
  k = 0;

  // Apply the initial conditions.
  //initialConditionsGPU<<<gridSize, blockSize>>>(nx, ny, dx, dy, x_length, d_x, d_y, d_h, d_uh, d_vh);

  // Move data to the Host for initial conditions file write
  //CHECK(cudaMemcpy(h, d_h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
  //CHECK(cudaMemcpy(uh, d_uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
  //CHECK(cudaMemcpy(vh, d_vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));

  //CHECK(cudaMemcpy(x, d_x, nx * sizeof ( float ), cudaMemcpyDeviceToHost));
  //CHECK(cudaMemcpy(y, d_y, nx * sizeof ( float ), cudaMemcpyDeviceToHost));

  // **** INITIAL CONDITIONS ****
  //Apply the initial conditions.
  //printf("Before initial conditions\n");
  initial_conditions(nx, ny, dx, dy, x_length, x, y, h, uh, vh);

  // Write initial condition to a file
  writeResults(h, uh, vh, x, y, time, nx, ny);

  // Move data to the device for calculations
  CHECK(cudaMemcpy(d_h, h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_uh, uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vh, vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));

  // ******************************************************************** COMPUTATION SECTION ******************************************************************** //

  // start timer
  clock_t time_start = clock();

  while (time < t_final) // time loop begins
  {
    // Take a time step and increase step counter
    time = time + dt;
    k++;

    // **** COMPUTE FLUXES ****
    computeFluxesGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, nx, ny);
    
    // **** COMPUTE VARIABLES ****
    computeVariablesGPU<<<gridSize, blockSize>>>(d_hm, d_uhm, d_vhm, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, d_h, d_uh, d_vh, lambda_x, lambda_y, nx, ny);
  
    // **** UPDATE VARIABLES ****
    updateVariablesGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_hm, d_uhm, d_vhm, nx, ny);

    // **** APPLY BOUNDARY CONDITIONS ****
    applyBoundaryConditionsGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, nx, ny, 3);  

    CHECK(cudaMemcpy(h, d_h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(uh, d_uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vh, d_vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));

    writeResults(h, uh, vh, x, y, time, nx, ny);

    CHECK(cudaMemcpy(d_h, h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_uh, uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vh, vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));

  } // end time loop

  // stop timer
  clock_t time_end = clock();
  double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;

  // Print out the results
  printf("Problem size: %d, time steps taken: %d,  elapsed time: %f s\n", nx, k, time_elapsed);

  // ******************************************************************** POSTPROCESSING ******************************************************************** //

  // Move data back to the host
  //CHECK(cudaMemcpy(h, d_h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
  //CHECK(cudaMemcpy(uh, d_uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
  //CHECK(cudaMemcpy(vh, d_vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));

  //writeResults(h, uh, vh, x, y, time, nx, ny);

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

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));

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
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
// ******************************************************************************************************************************************** //

// ************************************************ SERIAL CODE ************************************************ //
// **** COMPUTE FLUXES ****
//Compute fluxes (including ghosts)
/* 
for ( i = 0; i < ny+2; i++ )
  for ( j = 0; j < nx+2; j++)
  {
    id = ID_2D(i,j,nx);

    fh[id] = uh[id]; //flux for the height equation: u*h
    fuh[id] = uh[id] * uh[id] / h[id] + 0.5 * g * h[id] * h[id]; //flux for the momentum equation: u^2*h + 0.5*g*h^2
    fvh[id] = uh[id] * vh[id] / h[id]; //flux for the momentum equation: u*v**h 
    gh[id] = vh[id]; //flux for the height equation: v*h
    guh[id] = uh[id] * vh[id] / h[id]; //flux for the momentum equation: u*v**h 
    gvh[id] = vh[id] * vh[id] / h[id] + 0.5 * g * h[id] * h[id]; //flux for the momentum equation: v^2*h + 0.5*g*h^2
  }

// **** COMPUTE VARIABLES ****
//Compute updated variables
for ( i = 1; i < ny + 1; i++ )
  for ( j = 1; j < nx + 1; j++ )
  {
    id=ID_2D(i,j,nx);
    id_left=ID_2D(i,j-1,nx);
    id_right=ID_2D(i,j+1,nx);
    id_bottom=ID_2D(i-1,j,nx);
    id_top=ID_2D(i+1,j,nx);

    hm[id] = 0.25*(h[id_left]+h[id_right]+h[id_bottom]+h[id_top]) 
      - lambda_x * ( fh[id_right] - fh[id_left] ) 
      - lambda_y * ( gh[id_top] - gh[id_bottom] );

    uhm[id] = 0.25*(uh[id_left]+uh[id_right]+uh[id_bottom]+uh[id_top]) 
      - lambda_x * ( fuh[id_right] - fuh[id_left] ) 
      - lambda_y * ( guh[id_top] - guh[id_bottom] );

    vhm[id] = 0.25*(vh[id_left]+vh[id_right]+vh[id_bottom]+vh[id_top]) 
      - lambda_x * ( fvh[id_right] - fvh[id_left] ) 
      - lambda_y * ( gvh[id_top] - gvh[id_bottom] );
  }

// **** UPDATE VARIABLES ****
//update interior state variables
for (i = 1; i < ny+1; i++)
  for (j = 1; j < nx+1; j++)
  {
  id=ID_2D(i,j,nx);
  h[id] = hm[id];
  uh[id] = uhm[id];
  vh[id] = vhm[id];
  }

// **** APPLY BOUNDARY CONDITIONS ****
//Update the ghosts (boundary conditions)

//left
j = 1;
for(i = 1; i < ny + 1; i++)
  {

    id = ID_2D(i, j, nx);

    id_left = ID_2D(i, j - 1, nx);

    h[id_left]  = h[id];

    uh[id_left] = - uh[id];

    vh[id_left] = vh[id];

  }

//right
j = nx;
for(i = 1; i < ny + 1; i++)
  {

    id = ID_2D(i, j, nx);

    id_right = ID_2D(i, j + 1, nx);

    h[id_right]  = h[id];

    uh[id_right] = - uh[id];

    vh[id_right] = vh[id];

  }

//bottom
i = 1;
for(j = 1; j < nx + 1; j++)
  {

    id = ID_2D(i, j, nx);

    id_bottom = ID_2D(i - 1, j, nx);

    h[id_bottom]  = h[id];

    uh[id_bottom] = uh[id];

    vh[id_bottom] = - vh[id];

  }

//top
i = ny;
for(j = 1; j < nx + 1; j++)
  {

    id = ID_2D(i, j, nx);

    id_top = ID_2D(i + 1, j, nx);

    h[id_top]  = h[id];

    uh[id_top] = uh[id];

    vh[id_top] = - vh[id];

  }
*/

/*
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
    
  i = 0
  for( j = 1; j < nx+1; j++)
  {
    id=ID_2D(i,j,nx);
    id1=ID_2D(i+1,j,nx);

    h[id] = h[id1];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }

  i=nx+1;
  for( j = 1; j < nx+1; j++)
  {
    id=ID_2D(i,j,nx);
    id1=ID_2D(i-1,j,nx);

    h[id] = h[id1];
    uh[id] = 0.0;
    vh[id] = 0.0;
  } 
      
  j=0;
  for( i = 1; i < ny+1; i++)
  {
    id=ID_2D(i,j,nx);
    id1=ID_2D(i,j+1,nx);

    h[id] = h[id1];
    uh[id] = 0.0;
    vh[id] = 0.0;
  } 
  
  j=nx+1;
  for( i = 1; i < ny+1; i++)
  {
    id=ID_2D(i,j,nx);
    id1=ID_2D(i,j-1,nx);

    h[id] = h[id1];
    uh[id] = 0.0;
    vh[id] = 0.0;
  }

*/

/*
//Move data to the device for all GPU calculations
  CHECK(cudaMemcpy(d_h, h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_uh, uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vh, vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));

*/
