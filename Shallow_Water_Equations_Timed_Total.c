# define _POSIX_C_SOURCE 199309L
# include <time.h>
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>

#define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

typedef struct {
  struct timespec start;
  struct timespec end;
} Timer;

void initial_conditions (int nx, int ny, double dx, double dy, double x_length, float x[], float y[], float h[], float uh[] ,float vh[])
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

void getArgs(int *nx, double *dt, double *x_length, double *totalRuntime, int argc, char *argv[])
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

void timer_start(Timer* t) 
{
  clock_gettime(CLOCK_MONOTONIC, &(t->start));
}

void timer_stop(Timer* t) 
{
  clock_gettime(CLOCK_MONOTONIC, &(t->end));
}

double timer_elapsed(Timer* t) 
{
  double start_sec = (double)t->start.tv_sec + (double)t->start.tv_nsec / 1e9;
  double end_sec   = (double)t->end.tv_sec   + (double)t->end.tv_nsec / 1e9;
  return end_sec - start_sec;
}

int main ( int argc, char *argv[] )
{
  double dt;
  double programRuntime;
  double totalRuntime;

  double x_length;
  
  double dx;
  double dy;

  int i, j, k, l;

  int id;   
  int id_left;
  int id_right;
  int id_bottom;
  int id_top;

  int nx;
  int ny; 

  float g = 9.81f;
  float g_half = 0.5f * g;

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
  ny = nx; // we assume the grid is square

  //Define the locations of the nodes and time steps and the spacing.
  dx = x_length / ( double ) ( nx );
  dy = x_length / ( double ) ( nx );

  double lambda_x = 0.5 * dt/dx;
  double lambda_y = 0.5 * dt/dy;

  printf ( "\n" );
  printf ( "SHALLOW_WATER_2D\n" );
  printf ( "\n" );

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

  for (k = 1; k < 6; k++)
  {
    // set time to zero and step counter to zero
    programRuntime = 0.0f;
    l = 0;

    // **** INITIAL CONDITIONS ****
    initial_conditions(nx, ny, dx, dy, x_length, x, y, h, uh, vh);

    Timer main_timer;
    timer_start(&main_timer);

    while (programRuntime < totalRuntime)
    {
      //  Take a time step
      programRuntime += dt;
      l++;
        
      // **** COMPUTE FLUXES ****
        
      // Compute fluxes (including ghosts)
      for (int i = 0; i < ny + 2; i++)
        for (int j = 0; j < nx + 2; j++) 
        {
          int id = ID_2D(i, j, nx);
      
          float h_val = h[id];

          float uh_val = uh[id];

          float vh_val = vh[id];

          float inv_h = 1.0f / h_val;

          float h2 = h_val * h_val; 
      
          fh[id]  = uh_val;

          gh[id]  = vh_val;
      
          float uh2 = uh_val * uh_val; 

          float vh2 = vh_val * vh_val; 

          float uv  = uh_val * vh_val; 
      
          fuh[id] = uh2 * inv_h + g_half * h2;

          fvh[id] = uv  * inv_h;

          guh[id] = uv  * inv_h;

          gvh[id] = vh2 * inv_h + g_half * h2;
        }
        
      // **** COMPUTE VARIABLES ****

      //Compute updated variables
      for (int i = 1; i < ny + 1; i++)
        for (int j = 1; j < nx + 1; j++) 
        {
          int id = ID_2D(i, j, nx);
          int id_left = ID_2D(i, j - 1, nx);
          int id_right = ID_2D(i, j + 1, nx);
          int id_bottom = ID_2D(i - 1, j, nx);
          int id_top = ID_2D(i + 1, j, nx);
      
          // Load neighbor values into local registers
          float h_l  = h[id_left];
          float h_r  = h[id_right];
          float h_b  = h[id_bottom];
          float h_t  = h[id_top];
      
          float uh_l = uh[id_left];
          float uh_r = uh[id_right];
          float uh_b = uh[id_bottom];
          float uh_t = uh[id_top];
      
          float vh_l = vh[id_left];
          float vh_r = vh[id_right];
          float vh_b = vh[id_bottom];
          float vh_t = vh[id_top];
      
          float fh_l = fh[id_left];
          float fh_r = fh[id_right];
          float gh_b = gh[id_bottom];
          float gh_t = gh[id_top];
      
          float fuh_l = fuh[id_left];
          float fuh_r = fuh[id_right];
          float guh_b = guh[id_bottom];
          float guh_t = guh[id_top];
      
          float fvh_l = fvh[id_left];
          float fvh_r = fvh[id_right];
          float gvh_b = gvh[id_bottom];
          float gvh_t = gvh[id_top];
      
          hm[id] = 0.25f * (h_l + h_r + h_b + h_t)
                 - (float) lambda_x * (fh_r - fh_l)
                 - (float) lambda_y * (gh_t - gh_b);
      
          uhm[id] = 0.25f * (uh_l + uh_r + uh_b + uh_t)
                  - (float) lambda_x * (fuh_r - fuh_l)
                  - (float) lambda_y * (guh_t - guh_b);
      
          vhm[id] = 0.25f * (vh_l + vh_r + vh_b + vh_t)
                  - (float) lambda_x * (fvh_r - fvh_l)
                  - (float) lambda_y * (gvh_t - gvh_b);
        }
      
      // **** UPDATE VARIABLES ****

      //update interior state variables  
      for (i = 1; i < ny + 1; i++)
        for (j = 1; j < nx + 1; j++)
        {
          id = ID_2D(i, j, nx);

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

        float h_val = h[id];
        float uh_val = uh[id];
        float vh_val = vh[id];

        h[id_left]  = h_val;
        uh[id_left] = -uh_val;
        vh[id_left] = vh_val;
      }

      //right
      j = nx;
      for(i = 1; i < ny + 1; i++)
      {
        id = ID_2D(i, j, nx);

        id_right = ID_2D(i, j + 1, nx);

        float h_val = h[id];
        float uh_val = uh[id];
        float vh_val = vh[id];

        h[id_right]  = h_val;
        uh[id_right] = -uh_val;
        vh[id_right] = vh_val;
      }

      //bottom
      i = 1;
      for(j = 1; j < nx + 1; j++)
      {
        id = ID_2D(i, j, nx);

        id_bottom = ID_2D(i - 1, j, nx);

        float h_val = h[id];
        float uh_val = uh[id];
        float vh_val = vh[id];

        h[id_bottom]  = h_val;
        uh[id_bottom] = uh_val;
        vh[id_bottom] = -vh_val;
      }

      //top
      i = ny;
      for(j = 1; j < nx + 1; j++)
      {
        id = ID_2D(i, j, nx);

        id_top = ID_2D(i + 1, j, nx);

        float h_val = h[id];
        float uh_val = uh[id];
        float vh_val = vh[id];

        h[id_top]  = h_val;
        uh[id_top] = uh_val;
        vh[id_top] = -vh_val;
      }
    }

    timer_stop(&main_timer);
    double time_elapsed = timer_elapsed(&main_timer);

    printf("Problem size: %d, Time steps: %d, Iteration: %d, Elapsed Time: %f s\n", nx, l, k, time_elapsed);
  }
  
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