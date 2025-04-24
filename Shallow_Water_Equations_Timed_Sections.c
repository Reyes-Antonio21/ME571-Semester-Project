# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>

#define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

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

void getArgs(int *nx, double *dt, float *x_length, double *totalRuntime, int argc, char *argv[])
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
    *totalRuntime = 0.5;
  }else{
    *totalRuntime = atof ( argv[4] );
  }
    
  
}
/******************************************************************************/

int main ( int argc, char *argv[] )
{
  double dt;
  double programRuntime;
  double totalRuntime;
  
  int i, j, k, l;

  int id;   
  int id_left;
  int id_right;
  int id_bottom;
  int id_top;

  int nx;
  int ny; 

  int x_start;
  int y_start;

  float dx;
  float dy;

  float x_length;

  float g = 9.81f; 

  float *h;
  float *uh;
  float *vh;

  float *fh;
  float *fuh;
  float *fvh;

  float *gh;
  float *guh;
  float *gvh;

  float *hm;
  float *uhm;
  float *vhm;

  float *x;
  float *y;

  //get command line arguments
  getArgs(&nx, &dt, &x_length, &totalRuntime, argc, argv);
  ny = nx; // we assume this, does not have to be this way

  //Define the locations of the nodes and time steps and the spacing.
  dx = x_length / ( float ) ( nx );
  dy = x_length / ( float ) ( nx );

  float lambda_x = 0.5*dt/dx;
  float lambda_y = 0.5*dt/dy;

  // ************************************************ MEMORY ALLOCATIONS ************************************************ //
  
  //Allocate space (nx+2)((nx+2) long, to accound for ghosts
  //height array
  h  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  hm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  
  //x momentum array
  uh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  uhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fuh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  guh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  
  //y momentum array
  vh  = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  vhm = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  fvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  gvh = ( float * ) malloc ( (nx+2)*(ny+2) * sizeof ( float ) );
  
  // location arrays
  x = ( float * ) malloc ( nx * sizeof ( float ) );
  y = ( float * ) malloc ( ny * sizeof ( float ) );

  for (k = 1; k < 7; k++)
  {
    // set time to zero and step counter to zero
    programRuntime = 0.0f;
    l = 0;

    // instantiate section timing variables
    double time_elapsed_cf = 0.0;
    double time_elapsed_cv = 0.0;
    double time_elapsed_uv = 0.0;
    double time_elapsed_bc = 0.0;

    initial_conditions ( nx, ny, dx, dy, x_length, x, y, h, uh, vh);

    // start timer
    clock_t time_start = clock();

    while (programRuntime < totalRuntime) //time loop begins
    {
    //  Take a time step
    programRuntime += dt;
    l++;
      
    // **** COMPUTE FLUXES ****
    //Compute fluxes (including ghosts)

    // Start timing compute fluxes section
    clock_t flux_start = clock();
      
    for ( i = 0; i < ny + 2; i++ )
      for ( j = 0; j < nx + 2; j++)
      {
      id = ID_2D(i, j, nx);

      fh[id] = uh[id]; //flux for the height equation: u*h

      fuh[id] = uh[id] * uh[id] / h[id] + 0.5 * g * h[id] * h[id]; //flux for the momentum equation: u^2*h + 0.5*g*h^2

      fvh[id] = uh[id] * vh[id] / h[id]; //flux for the momentum equation: u*v**h 

      gh[id] = vh[id]; //flux for the height equation: v*h

      guh[id] = uh[id] * vh[id] / h[id]; //flux for the momentum equation: u*v**h 

      gvh[id] = vh[id] * vh[id] / h[id] + 0.5 * g * h[id] * h[id]; //flux for the momentum equation: v^2*h + 0.5*g*h^2
      }
      
    // End timing the compute fluxes section
    clock_t flux_end = clock();
    time_elapsed_cf += (double)(flux_end - flux_start) / CLOCKS_PER_SEC;

    // **** COMPUTE VARIABLES ****
    //Compute updated variables

    // Start timing compute variables section
    clock_t compute_variables_start = clock();
      
    for ( i = 1; i < ny + 1; i++ )
      for ( j = 1; j < nx + 1; j++ )
      {

      id = ID_2D(i, j, nx);
        
      id_left = ID_2D(i, j - 1, nx);

      id_right = ID_2D(i, j + 1, nx);

      id_bottom = ID_2D(i - 1, j, nx);

      id_top = ID_2D(i + 1, j, nx);

      hm[id] = 0.25 * (h[id_left] + h[id_right] + h[id_bottom] + h[id_top]) 
        - lambda_x * ( fh[id_right] - fh[id_left] ) 
        - lambda_y * ( gh[id_top] - gh[id_bottom] );

      uhm[id] = 0.25 * (uh[id_left] + uh[id_right] + uh[id_bottom] + uh[id_top]) 
        - lambda_x * ( fuh[id_right] - fuh[id_left] ) 
        - lambda_y * ( guh[id_top] - guh[id_bottom] );

      vhm[id] = 0.25 * (vh[id_left] + vh[id_right] + vh[id_bottom] + vh[id_top]) 
        - lambda_x * ( fvh[id_right] - fvh[id_left] ) 
        - lambda_y * ( gvh[id_top] - gvh[id_bottom] );
      }

    // End timing the compute variables section
    clock_t compute_variables_end = clock();
    time_elapsed_cv += (double)(compute_variables_end - compute_variables_start) / CLOCKS_PER_SEC;

    // **** UPDATE VARIABLES ****
    //update interior state variables

    // Start timing update variables section 
    clock_t update_variables_start = clock();
      
    for (i = 1; i < ny + 1; i++)
      for (j = 1; j < nx + 1; j++)
      {

      id = ID_2D(i, j, nx);

      h[id] = hm[id];

      uh[id] = uhm[id];

      vh[id] = vhm[id];

      }

    // End timing update variables section
    clock_t update_variables_end = clock();
    time_elapsed_uv += (double)(update_variables_end - update_variables_start) / CLOCKS_PER_SEC;

    // **** APPLY BOUNDARY CONDITIONS ****
    //Update the ghosts (boundary conditions)

    // Start timing apply boundary conditions section
    clock_t apply_boundary_conditions_start = clock();
      
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

    // End timing apply boundary conditions section
    clock_t apply_boundary_conditions_end = clock();
    time_elapsed_bc += (double)(apply_boundary_conditions_end - apply_boundary_conditions_start) / CLOCKS_PER_SEC;
    
    } //end time loop

    clock_t time_end = clock();
    double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;

    double avg_time_elapsed_cf = time_elapsed_cf / (double) l;
    double avg_time_elapsed_cv = time_elapsed_cv / (double) l;
    double avg_time_elapsed_uv = time_elapsed_uv / (double) l;
    double avg_time_elapsed_bc = time_elapsed_bc / (double) l;

    printf("Problem size: %d, Time steps: %d, Iteration: %d, Elapsed time: %f s, Average elapsed time for compute fluxes: %f s, Average elapsed time for compute variables: %f s, Average elapsed time for update variables: %f s, Average elapsed time for apply boundary conditions: %f s\n", nx, l, k, time_elapsed, avg_time_elapsed_cf, avg_time_elapsed_cv, avg_time_elapsed_uv, avg_time_elapsed_bc);

  } // End for loop for 5 iterations of calculation
  
  // **** POSTPROCESSING ****
  
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
  free ( x );
  free ( y );

 //Terminate.

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
/******************************************************************************/