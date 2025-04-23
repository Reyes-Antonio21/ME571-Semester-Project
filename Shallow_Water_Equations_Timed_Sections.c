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

void write_results ( char *output_filename, int nx, int ny, float x[], float y[], float h[], float uh[], float vh[])
/******************************************************************************/

{
  int i,j, id;
  FILE *output;
   
  //Open the file.
  output = fopen ( output_filename, "wt" );
    
  if ( !output ){
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "WRITE_RESULTS - Fatal error!\n" );
    fprintf ( stderr, "  Could not open the output file.\n" );
    exit ( 1 );
  }
    
  //Write the data.
  for ( i = 0; i < ny; i++ ) 
    for ( j = 0; j < nx; j++ ){
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
/******************************************************************************/

int main ( int argc, char *argv[] )
{
  float dx;
  float dy;
  float dt;
  float g = 9.81; //[m^2/s] gravitational constant
  float *h;
  float *fh;
  float *gh;
  float *hm;
  int i, j, k, l, id, id_left, id_right, id_bottom, id_top;
  int nx, ny;
  float t_final;
  float *uh;
  float *fuh;
  float *guh;
  float *uhm;
  float *vh;
  float *fvh;
  float *gvh;
  float *vhm;
  float *x;
  float *y;
  float x_length, time;

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D\n" );
  printf ( "\n" );

  //get command line arguments
  getArgs(&nx, &dt, &x_length, &t_final, argc, argv);
   
  printf ( "  NX = %d\n", nx );
  printf ( "  DT = %g\n", dt );
  printf ( "  X_LENGTH = %g\n", x_length );
  printf ( "  T_FINAL = %g\n", t_final );
  
  ny = nx; // we assume this, does not have to be this way

  // **** ALLOCATE MEMORY ****
  
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

  for (k = 1; k < 4; k++)
  {
    //Define time-step counter
    l = 0;
    
    //Define the locations of the nodes and time steps and the spacing.
    dx = x_length / ( float ) ( nx );
    dy = x_length / ( float ) ( nx );

    // **** INITIAL CONDITIONS ****
    //Apply the initial conditions.
    printf("Before initial conditions\n");
    initial_conditions ( nx, ny, dx, dy, x_length, x, y, h, uh, vh);

    printf("Before write results\n");
    //Write initial condition to a file
    //write_results("sw2d_init.dat",nx,ny,x,y,h,uh,vh);

    // **** TIME LOOP ****
    float lambda_x = 0.5*dt/dx;
    float lambda_y = 0.5*dt/dy;

    double total_flux_time = 0.0;
    double total_compute_variables_time = 0.0;
    double total_update_variables_time = 0.0;
    double total_apply_boundary_conditions_time = 0.0;

    time = 0;
    clock_t time_start = clock();

    while (time < t_final) //time loop begins
    {

    //  Take a time step
    time = time + dt;
      
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
    total_flux_time += (double)(flux_end - flux_start) / CLOCKS_PER_SEC;

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
    total_compute_variables_time += (double)(compute_variables_end - compute_variables_start) / CLOCKS_PER_SEC;

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
    total_update_variables_time += (double)(update_variables_end - update_variables_start) / CLOCKS_PER_SEC;

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
    total_apply_boundary_conditions_time += (double)(apply_boundary_conditions_end - apply_boundary_conditions_start) / CLOCKS_PER_SEC;

    l = l + 1;
    
    } //end time loop

    clock_t time_end = clock();
    double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;
    double avg_flux_time = total_flux_time / (double) l;
    double avg_compute_variables_time = total_compute_variables_time / (double) l;
    double avg_update_variables_time = total_update_variables_time / (double) l;
    double avg_apply_boundary_conditions_time = total_apply_boundary_conditions_time / (double) l;

    printf("Problem size: %d, Time Elapsed: %f s, Time steps taken: %d \n", nx, time_elapsed, l);
    printf("Average Flux computation time: %fs\n", avg_flux_time);
    printf("Average Compute Variables time: %fs\n", avg_compute_variables_time);
    printf("Average Update Variables time: %fs\n", avg_update_variables_time);
    printf("Average Apply Boundary Conditions time: %fs\n", avg_apply_boundary_conditions_time);

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