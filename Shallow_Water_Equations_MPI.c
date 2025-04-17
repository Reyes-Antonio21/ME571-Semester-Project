# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <mpi.h>

#define ID_2D(i,j,nx) ((i)*(nx+2)+(j))

/****************************************************************************** Helper Functions ******************************************************************************/

int computeGlobalStart(int coord, int total_cells, int divisions) 
{
  int base = coord * (total_cells / divisions);
  int remainder = total_cells % divisions;

  if (coord < remainder) {
      return base + coord;
  } else {
      return base + remainder;
  }
}
/******************************************************************************/

void getArgs(int *nx, int *ny, double *dt, float *x_length, float *y_length, double *totalRuntime, int argc, char *argv[])
{
  if (argc <= 1)
  {
    *nx = 800;
  }
  else
  {
    *nx = atoi (argv[1]);
  }

  if (argc <= 2)
  {
    *ny = 800;
  }
  else
  {
    *ny = atoi (argv[2]);
  }
  
  if (argc <= 3)
  {
    *dt = 0.001;
  }
  else
  {
    *dt = atof (argv[3]);
  }
  
  if (argc <= 4)
  {
    *x_length = 10.0;
  }
  else
  {
    *x_length = atof (argv[4]);
  }
  
  if (argc <= 5)
  {
    *y_length = 10.0;
  }
  else
  {
    *y_length = atof (argv[5]);
  }

  if (argc <= 6)
  {
    *totalRuntime = 0.5;
  }
  else
  {
    *totalRuntime = atof (argv[6]);
  }
}
/******************************************************************************/

void initialConditions(int nx_local, int ny_local, int px, int py, int dims[2], int nx, int ny, float x_length, float y_length, float dx, float dy, float *h, float *uh, float *vh)
{
  int i, j;
  int id, id_ghost;

  float *x_coords = malloc((nx_local + 2) * sizeof(float));
  float *y_coords = malloc((ny_local + 2) * sizeof(float));

  int global_x_start = computeGlobalStart(py, nx, dims[1]);
  int global_y_start = computeGlobalStart(px, ny, dims[0]);

  for (int j = 0; j < nx_local + 2; j++) 
  {
    int global_j = global_x_start + j - 1;

    x_coords[j] = -x_length / 2 + dx / 2 + global_j * dx;
  }

  for (int i = 0; i < ny_local + 2; i++) 
  {
    int global_i = global_y_start + i - 1;
    
    y_coords[i] = -y_length / 2 + dy / 2 + global_i * dy;
  }

  for (int i = 0; i < ny_local + 2; i++) 
    for (int j = 0; j < nx_local + 2; j++) 
    {
      int id = ID_2D(i, j, nx_local);

      float x = x_coords[j];
      float y = y_coords[i];
      
      h[id] = 1.0f + 0.4f * exp(-5.0f * (x * x + y * y));

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }

  // Apply physical domain boundary conditions
  // Bottom boundary
  if (px == 0) 
  {  
    for (int j = 1; j <= nx_local; j++) 
    {
      int id = ID_2D(0, j, nx_local);

      int id_ghost = ID_2D(1, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Top boundary
  if (px == dims[0] - 1) 
  { 
    for (int j = 1; j <= nx_local; j++) 
    {
      int id = ID_2D(ny_local + 1, j, nx_local);

      int id_ghost = ID_2D(ny_local, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Left boundary
  if (py == 0) 
  {  
    for (int i = 1; i <= ny_local; i++) 
    {
      int id = ID_2D(i, 0, nx_local);

      int id_ghost = ID_2D(i, 1, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Right boundary
  if (py == dims[1] - 1) 
  {  
    for (int i = 1; i <= ny_local; i++) 
    {
      int id = ID_2D(i, nx_local + 1, nx_local);

      int id_ghost = ID_2D(i, nx_local, nx_local);

      h[id] = h[id_ghost];
      
      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }  
  
  free(x_coords);
  free(y_coords);

  return;
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

int main (int argc, char *argv[])
{
  /****************************************************************************** Instantiation ******************************************************************************/
  // Start the clock
  clock_t time_start = clock();
  
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Initialize variables
  // MPI variables
  int px; 
  int py;

  int rank;
  int size;

  int north;
  int south;
  int east;
  int west;

  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  int coords[2];

  // Variables
  double dt;
  double programRuntime;
  double totalRuntime;

  int i, j, k;

  int nx; 
  int ny;

  int nx_local;
  int ny_local;
  int nx_extra;
  int ny_extra;

  int id;

  int id_left;
  int id_right;
  int id_top;
  int id_bottom;

  float dx;
  float dy;

  float x_length;
  float y_length;

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

  // Get command line arguments
  getArgs(&nx, &ny, &dt, &x_length, &y_length, &totalRuntime, argc, argv);
  
  // Define the locations of the nodes, time steps, and spacing
  dx = x_length / ( float ) ( nx );
  dy = y_length / ( float ) ( ny );

  // Define the time step and the grid spacing
  float lambda_x = 0.5f * (float) dt/dx;
  float lambda_y = 0.5f * (float) dt/dy;

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Create a Cartesian topology
  MPI_Dims_create(size, 2, dims);

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  px = coords[1];
  py = coords[0];

  nx_local = nx / dims[1];
  ny_local = ny / dims[0];

  nx_extra = nx % dims[1];
  ny_extra = ny % dims[0];

  if (px < nx_extra)
  {
    nx_local++;
  }

  if (py < ny_extra)
  {
    ny_local++;
  }

/****************************************************************************** ALLOCATE MEMORY ******************************************************************************/
  // Allocate space (nx+2)((nx+2) long, to account for ghosts
  h = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  uh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  vh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));

  fh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fuh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  gh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  guh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  gvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));

  /****************************************************************************** MAIN LOOP ******************************************************************************/
  if (rank == 0)
  {
    printf ("SHALLOW_WATER_2D:\n");
    printf ("  A program to solve the shallow water equations.\n");
    printf ("\n");
    printf ("  The problem is solved on a rectangular grid.\n");
    printf ("\n");
    printf ("  The grid has %d x %d nodes.\n", nx, ny);
    printf (" The total program runtime is %g.\n", totalRuntime);
    printf ("  The time step is %g.\n", dt);
    printf ("  The grid spacing is %g x %g.\n", dx, dy);
    printf ("  The grid length is %g x %g.\n", x_length, y_length);
    printf (" The number of processes is %d.\n", size);
    printf (" The processor grid dimensions are %d x %d.\n", dims[0], dims[1]);
  }

  // **** INITIAL CONDITIONS ****
  initialConditions(nx_local, ny_local, px, py, dims, nx, ny, x_length, y_length, dx, dy, h, uh, vh);

  // Time-stepping loop and MPI halo exchange
  MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
  MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

  MPI_Request requests[8];

  programRuntime = 0.0f;

  while (programRuntime < totalRuntime) 
  {
    // Halo exchange for h (simplified for illustration)
    MPI_Isend(&h[ID_2D(1,1,nx_local)], nx_local, MPI_FLOAT, north, 0, cart_comm, &requests[0]);
    MPI_Irecv(&h[ID_2D(ny_local+1,1,nx_local)], nx_local, MPI_FLOAT, south, 0, cart_comm, &requests[1]);
    MPI_Isend(&h[ID_2D(ny_local,1,nx_local)], nx_local, MPI_FLOAT, south, 1, cart_comm, &requests[2]);
    MPI_Irecv(&h[ID_2D(0,1,nx_local)], nx_local, MPI_FLOAT, north, 1, cart_comm, &requests[3]);

    for (int i = 1; i <= ny_local; i++) 
    {
      MPI_Isend(&h[ID_2D(i,1,nx_local)], 1, MPI_FLOAT, west, 2, cart_comm, &requests[4]);
      MPI_Irecv(&h[ID_2D(i,nx_local+1,nx_local)], 1, MPI_FLOAT, east, 2, cart_comm, &requests[5]);
      MPI_Isend(&h[ID_2D(i,nx_local,nx_local)], 1, MPI_FLOAT, east, 3, cart_comm, &requests[6]);
      MPI_Irecv(&h[ID_2D(i,0,nx_local)], 1, MPI_FLOAT, west, 3, cart_comm, &requests[7]);
    }
    
    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    for (int i = 1; i <= ny_local; i++) 
      for (int j = 1; j <= nx_local; j++) 
      {
        int id = ID_2D(i, j, nx_local);
        fh[id] = uh[id];
        fuh[id] = uh[id] * uh[id] / h[id] + 0.5f * g * h[id] * h[id];
        fvh[id] = uh[id] * vh[id] / h[id];
        gh[id] = vh[id];
        guh[id] = uh[id] * vh[id] / h[id];
        gvh[id] = vh[id] * vh[id] / h[id] + 0.5f * g * h[id] * h[id];
      }

    for (int i = 1; i <= ny_local; i++)
      for (int j = 1; j <= nx_local; j++) 
      {
        int id = ID_2D(i, j, nx_local);
        int id_left = ID_2D(i, j - 1, nx_local);
        int id_right = ID_2D(i, j + 1, nx_local);
        int id_bottom = ID_2D(i - 1, j, nx_local);
        int id_top = ID_2D(i + 1, j, nx_local);

        h[id] = 0.25f * (h[id_left] + h[id_right] + h[id_bottom] + h[id_top])
              - lambda_x * (fh[id_right] - fh[id_left])
              - lambda_y * (gh[id_top] - gh[id_bottom]);

        uh[id] = 0.25f * (uh[id_left] + uh[id_right] + uh[id_bottom] + uh[id_top])
              - lambda_x * (fuh[id_right] - fuh[id_left])
              - lambda_y * (guh[id_top] - guh[id_bottom]);

        vh[id] = 0.25f * (vh[id_left] + vh[id_right] + vh[id_bottom] + vh[id_top])
              - lambda_x * (fvh[id_right] - fvh[id_left])
              - lambda_y * (gvh[id_top] - gvh[id_bottom]);
      }

      programRuntime += dt;
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

  // === Gather local data (h, uh, vh) and write using writeResults() ===
  float *h_sendbuf = malloc(nx_local * ny_local * sizeof(float));
  float *uh_sendbuf = malloc(nx_local * ny_local * sizeof(float));
  float *vh_sendbuf = malloc(nx_local * ny_local * sizeof(float));

  for (int i = 1; i <= ny_local; i++) 
    for (int j = 1; j <= nx_local; j++) 
    {
      int idx = (i - 1) * nx_local + (j - 1);
      int id = ID_2D(i, j, nx_local);
      h_sendbuf[idx] = h[id];
      uh_sendbuf[idx] = uh[id];
      vh_sendbuf[idx] = vh[id];
    }

  int *recvcounts = NULL;
  int *displs = NULL;
  float *h_global = NULL;
  float *uh_global = NULL;
  float *vh_global = NULL;

  if (rank == 0) 
  {
    recvcounts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    h_global = malloc(nx * ny * sizeof(float));
    uh_global = malloc(nx * ny * sizeof(float));
    vh_global = malloc(nx * ny * sizeof(float));

    int offset = 0;
    for (int r = 0; r < size; r++) 
    {
      int coords_r[2];
      MPI_Cart_coords(cart_comm, r, 2, coords_r);
      int px_r = coords_r[0];
      int py_r = coords_r[1];

      int nx_local_r = nx / dims[1];
      int ny_local_r = ny / dims[0];
      if (py_r < nx % dims[1]) nx_local_r++;
      if (px_r < ny % dims[0]) ny_local_r++;

      recvcounts[r] = nx_local_r * ny_local_r;
    }
    displs[0] = 0;
    for (int r = 1; r < size; r++) 
    {
      displs[r] = displs[r - 1] + recvcounts[r - 1];
    }
  }

  MPI_Gatherv(h_sendbuf, nx_local * ny_local, MPI_FLOAT,
              h_global, recvcounts, displs, MPI_FLOAT,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(uh_sendbuf, nx_local * ny_local, MPI_FLOAT,
              uh_global, recvcounts, displs, MPI_FLOAT,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(vh_sendbuf, nx_local * ny_local, MPI_FLOAT, vh_global, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0) 
  {
    float *x = malloc(nx * sizeof(float));
    float *y = malloc(ny * sizeof(float));
    
    float dx = x_length / nx;
    float dy = y_length / ny;

    for (i = 0; i < nx; i++) 
    {
      x[i] = -x_length / 2 + dx / 2 + i * dx;
    }
    for (i = 0; i < ny; i++) 
    {
      y[i] = -y_length / 2 + dy / 2 + i * dy;
    }

    writeResults(h_global, uh_global, vh_global, x, y, programRuntime, nx, ny);

    free(x);
    free(y);
    free(h_global);
    free(uh_global);
    free(vh_global);
    free(recvcounts);
    free(displs);
  }

  free(h_sendbuf);
  free(uh_sendbuf);
  free(vh_sendbuf);

  MPI_Finalize();

  printf("Time-stepping loop completed.\n");
    
  clock_t time_end = clock();
  double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;

  printf("Problem size: %d, Time Elapsed: %f s, Time Steps Taken: %f \n", nx, time_elapsed, programRuntime/dt);

  return 0;
  }
  /***************************************************************************** END OF MAIN FUNCTION ****************************************************************************/