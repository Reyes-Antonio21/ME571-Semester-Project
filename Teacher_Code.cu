# include "common.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <cuda_runtime.h>

#define ID_2D(i,j,nx) ((i)*(nx+2)+(j))


//utilities
void getArgs(int *nx, float *dt, float *x_length, float *t_final, int argc, char *argv[]);

void write_results ( char *output_filename, int nx, int ny, float x[], float y[], float h[], float uh[], float vh[]);

void initial_conditions ( int nx, int ny, float dx, float dy,  float x_length, float x[],float y[], float h[], float uh[] ,float vh[]);

__global__ void applyBoundaryConditionsGPU(float *h, float *uh, float *vh, int nx, int ny, int bc_type) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= nx + 2 || j >= ny + 2) return; // Ensure we are within valid bounds

  int id = ID_2D(i, j, nx);

  // Apply boundary conditions based on type
  if (bc_type == 0) {  // Dirichlet (Fixed Value)
      if (i == 0 || i == nx+1) { h[id] = 1.0f; uh[id] = 0.0f; vh[id] = 0.0f; }
      if (j == 0 || j == ny+1) { h[id] = 1.0f; uh[id] = 0.0f; vh[id] = 0.0f; }
  }
  else if (bc_type == 1) {  // Neumann (Zero Gradient)
      if (i == 0) { h[id] = h[ID_2D(1, j, nx)]; uh[id] = uh[ID_2D(1, j, nx)]; vh[id] = vh[ID_2D(1, j, nx)]; }
      if (i == nx+1) { h[id] = h[ID_2D(nx, j, nx)]; uh[id] = uh[ID_2D(nx, j, nx)]; vh[id] = vh[ID_2D(nx, j, nx)]; }
      if (j == 0) { h[id] = h[ID_2D(i, 1, nx)]; uh[id] = uh[ID_2D(i, 1, nx)]; vh[id] = vh[ID_2D(i, 1, nx)]; }
      if (j == ny+1) { h[id] = h[ID_2D(i, ny, nx)]; uh[id] = uh[ID_2D(i, ny, nx)]; vh[id] = vh[ID_2D(i, ny, nx)]; }
  }
  else if (bc_type == 2) {  // Periodic
      if (i == 0) { h[id] = h[ID_2D(nx, j, nx)]; uh[id] = uh[ID_2D(nx, j, nx)]; vh[id] = vh[ID_2D(nx, j, nx)]; }
      if (i == nx+1) { h[id] = h[ID_2D(1, j, nx)]; uh[id] = uh[ID_2D(1, j, nx)]; vh[id] = vh[ID_2D(1, j, nx)]; }
      if (j == 0) { h[id] = h[ID_2D(i, ny, nx)]; uh[id] = uh[ID_2D(i, ny, nx)]; vh[id] = vh[ID_2D(i, ny, nx)]; }
      if (j == ny+1) { h[id] = h[ID_2D(i, 1, nx)]; uh[id] = uh[ID_2D(i, 1, nx)]; vh[id] = vh[ID_2D(i, 1, nx)]; }
  }
  else if (bc_type == 3) {  // Reflective Boundary Conditions
      if (i == 0) {  
          h[id] = h[ID_2D(1, j, nx)];  // Mirror h
          uh[id] = -uh[ID_2D(1, j, nx)];  // Reflect normal velocity (x-direction)
          vh[id] = vh[ID_2D(1, j, nx)];   // Tangential velocity unchanged
      }
      if (i == nx+1) {  
          h[id] = h[ID_2D(nx, j, nx)];
          uh[id] = -uh[ID_2D(nx, j, nx)];
          vh[id] = vh[ID_2D(nx, j, nx)];
      }
      if (j == 0) {  
          h[id] = h[ID_2D(i, 1, nx)];
          vh[id] = -vh[ID_2D(i, 1, nx)];  // Reflect normal velocity (y-direction)
          uh[id] = uh[ID_2D(i, 1, nx)];   // Tangential velocity unchanged
      }
      if (j == ny+1) {  
          h[id] = h[ID_2D(i, ny, nx)];
          vh[id] = -vh[ID_2D(i, ny, nx)];
          uh[id] = uh[ID_2D(i, ny, nx)];
      }
  }
}
/******************************************************************************/

__global__ void computeFluxesGPU(float *h,  float *uh,  float *vh, float *fh, float *fuh, float *fvh, float *gh, float *guh, float *gvh, int nx, int ny)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= nx || j >= ny) return; // Ensure we stay inside computational domain

  unsigned int id = ID_2D(i+1, j+1, nx); // Offset to skip ghost cells

  float g = 9.81; // Gravitational acceleration

  // Compute fluxes
  fh[id] = uh[id];  // flux for height equation: u*h
  fuh[id] = uh[id] * uh[id] / h[id] + 0.5 * g * h[id] * h[id]; // momentum equation: u²h + 0.5*g*h²
  fvh[id] = uh[id] * vh[id] / h[id]; // momentum equation: u*v*h

  gh[id] = vh[id];  // flux for height equation: v*h
  guh[id] = uh[id] * vh[id] / h[id]; // momentum equation: u*v*h
  gvh[id] = vh[id] * vh[id] / h[id] + 0.5 * g * h[id] * h[id]; // momentum equation: v²h + 0.5*g*h²

  __syncthreads(); 
}
/******************************************************************************/

int main ( int argc, char *argv[] )
{
  float dx;
  float dy;
  float dt;
  float *h;
  float *fh, *h_fh;
  float *gh, *h_gh;
  float *hm;
  int i,j, id, id_left, id_right, id_bottom, id_top;
  int nx, ny;
  float t_final;
  float *uh;
  float *fuh, *h_fuh;
  float *guh, *h_guh;
  float *uhm;
  float *vh;
  float *fvh, *h_fvh;
  float *gvh, *h_gvh;
  float *vhm;
  float *x;
  float *y;
  float x_length, time;

  //printf ( "\n" );
  //printf ( "SHALLOW_WATER_2D\n" );
  //printf ( "\n" );

  //get command line arguments
  getArgs(&nx, &dt, &x_length, &t_final, argc, argv);
  
  //printf ( "  NX = %d\n", nx );
  //printf ( "  DT = %g\n", dt );
  //printf ( "  X_LENGTH = %g\n", x_length );
  //printf ( "  T_FINAL = %g\n", t_final );
  
  ny=nx; // we assume this, does not have to be this way

  // **** ALLOCATE MEMORY ****
  
  //Allocate space (nx+2)((nx+2) long, to accound for ghosts
  //height array
  h  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  hm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  h_fh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  h_gh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  //x momentum array
  uh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  uhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fuh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  guh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  h_fuh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  h_guh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  //y momentum array
  vh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  vhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  h_fvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  h_gvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );

  // location arrays
  x = ( float * ) malloc ( nx * sizeof ( float ) );
  y = ( float * ) malloc ( ny * sizeof ( float ) );

  //Allocate memory on the device
  float *d_h, *d_uh, *d_vh;
  float *d_fh, *d_fuh, *d_fvh;
  float *d_gh, *d_guh, *d_gvh;

  CHECK(cudaMalloc((void **)&d_h, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_uh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_vh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_fh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_fuh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_fvh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_gh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_guh, (nx+2)*(ny+2) * sizeof ( float )));
  CHECK(cudaMalloc((void **)&d_gvh, (nx+2)*(ny+2) * sizeof ( float )));

  //Define the locations of the nodes and time steps and the spacing.
  dx = x_length / ( float ) ( nx );
  dy = x_length / ( float ) ( nx );

    // **** INITIAL CONDITIONS ****
  //Apply the initial conditions.
  //printf("Before initial conditions\n");
  initial_conditions ( nx, ny, dx, dy, x_length,  x, y, h, uh, vh);

  //printf("Before write results\n");
  //Write initial condition to a file
  write_results("tc2d_init.dat",nx,ny,x,y,h,uh,vh);


  // **** TIME LOOP ****
  float lambda_x = 0.5*dt/dx;
  float lambda_y = 0.5*dt/dy;


  time=0;
  int k=0; //time-step counter
  //start timer
  clock_t time_start = clock();

  while (time<t_final) //time loop begins
    {
      //  Take a time step
      time=time+dt;
      k++;
      //printf("time = %f\n",time);
      // **** COMPUTE FLUXES ****
      //Compute fluxes (including ghosts) 
      /*for ( i = 0; i < ny+2; i++ )
	    for ( j = 0; j < nx+2; j++)
      {
        id=ID_2D(i,j,nx);

        fh[id] = uh[id]; //flux for the height equation: u*h
        fuh[id] = uh[id]*uh[id]/h[id] + 0.5*g*h[id]*h[id]; //flux for the momentum equation: u^2*h + 0.5*g*h^2
        fvh[id] = uh[id]*vh[id]/h[id]; //flux for the momentum equation: u*v**h 
        gh[id] = vh[id]; //flux for the height equation: v*h
        guh[id] = uh[id]*vh[id]/h[id]; //flux for the momentum equation: u*v**h 
        gvh[id] = vh[id]*vh[id]/h[id] + 0.5*g*h[id]*h[id]; //flux for the momentum equation: v^2*h + 0.5*g*h^2
	    }
      */

      //Move data to the device
      CHECK(cudaMemcpy(d_h, h, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_uh, uh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_vh, vh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyHostToDevice));

      //Define the block and grid sizes
      int dimx = 32;
      int dimy = 32;
      dim3 blockSize(dimx, dimy);
      dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

      // Apply boundary conditions first (bc_type = 3 for reflective)
      applyBoundaryConditionGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, nx, ny, 3);
      cudaDeviceSynchronize();

      // Compute fluxes after boundary conditions are enforced
      computeFluxesGPU<<<gridSize, blockSize>>>(d_h, d_uh, d_vh, d_fh, d_fuh, d_fvh, d_gh, d_guh, d_gvh, nx, ny);
      cudaDeviceSynchronize();

      //Move fluxes back - for now
      CHECK(cudaMemcpy(fh, d_fh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(fuh, d_fuh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(fvh, d_fvh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(gh, d_gh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(guh, d_guh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(gvh, d_gvh, (nx+2)*(ny+2) * sizeof ( float ), cudaMemcpyDeviceToHost));
      
      // **** COMPUTE VARIABLES ****
      //Compute updated variables
      for ( i = 1; i < ny+1; i++ )
	      for ( j = 1; j < nx+1; j++ )
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
    } //end time loop

  clock_t time_end = clock();
  double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;
    
  printf("Problem size: %d, time steps taken: %d,  elapsed time: %f s\n", nx,k,time_elapsed);

  // **** POSTPROCESSING ****
  // Write data to file
  write_results("tc2d_final.dat", nx, ny, x, y, h, uh, vh);

  CHECK(cudaFree(d_h));
  CHECK(cudaFree(d_uh));
  CHECK(cudaFree(d_vh));
  CHECK(cudaFree(d_fh));
  CHECK(cudaFree(d_fuh));
  CHECK(cudaFree(d_fvh));
  CHECK(cudaFree(d_gh));
  CHECK(cudaFree(d_guh));
  CHECK(cudaFree(d_gvh));


  //Free memory.
  free ( h );
  free ( uh );
  free ( vh ); 
  free ( fh );
  free ( fuh );
  free ( fvh ); 
  free ( gh );
  free ( guh );
  free ( gvh ); 

  free ( h_fh );
  free ( h_fuh );
  free ( h_fvh ); 
  free ( h_gh );
  free ( h_guh );
  free ( h_gvh ); 

  free ( x );
  free ( y );

  //Terminate.

  //printf ( "\n" );
  //printf ( "SHALLOW_WATER_2D:\n" );
  //printf ( "  Normal end of execution.\n" );
  //printf ( "\n" );

  return 0;
}
/******************************************************************************/

void initial_conditions ( int nx, int ny, float dx, float dy,  float x_length, float x[],float y[], float h[], float uh[] ,float vh[]){
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

void write_results ( char *output_filename, int nx, int ny, float x[], float y[], float h[], float uh[], float vh[])
{
  int i,j, id;
  FILE *output;
   
  //Open the file.
  output = fopen ( output_filename, "wt" );
    
  if ( !output )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "WRITE_RESULTS - Fatal error!\n" );
    fprintf ( stderr, "  Could not open the output file.\n" );
    exit ( 1 );
  }
    
  //Write the data.
  for ( i = 0; i < ny; i++ ) 
    for ( j = 0; j < nx; j++ )
    {
        id=ID_2D(i+1,j+1,nx);
	      fprintf ( output, "  %24.16g\t%24.16g\t%24.16g\t %24.16g\t %24.16g\n", x[j], y[i],h[id], uh[id], vh[id]);
    }
    
  //Close the file.
  fclose ( output );
  
  return;
}
/******************************************************************************/

void getArgs(int *nx, float *dt, float *x_length, float *t_final, int argc, char *argv[])
{
  /*
    Get the quadrature file root name:
  */
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