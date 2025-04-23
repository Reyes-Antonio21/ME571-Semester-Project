# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <mpi.h>

/******************************************************************************* STRUCTURES & DEFINITIONS ******************************************************************************/
# define ID_2D(i, j, nx) ((i) * (nx + 2) + (j))

/****************************************************************************** Helper Functions ******************************************************************************/
int computeLocalSize(int global_size, int coord, int divisions) 
{
  int base = global_size / divisions;
  int rem  = global_size % divisions;
  return base + (coord < rem ? 1 : 0);
}
/******************************************************************************/

int computeGlobalStart(int coord, int global_size, int divisions) 
{
  int base = global_size / divisions;
  int rem  = global_size % divisions;

  if (coord < rem)
  {
    return coord * (base + 1);
  }
  else
  {
    return rem * (base + 1) + (coord - rem) * base;
  }
}
/******************************************************************************/

void getArgs(int *nx_global, int *ny_global, double *dt, float *x_length, float *y_length, double *totalRuntime, int argc, char *argv[])
{
  if (argc <= 1)
  {
    *nx_global = 100;
  }
  else
  {
    *nx_global = atoi (argv[1]);
  }

  if (argc <= 2)
  {
    *ny_global = 100;
  }
  else
  {
    *ny_global = atoi (argv[2]);
  }
  
  if (argc <= 3)
  {
    *dt = 0.008;
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

void write_results_mpi (int N, int N_loc, double time, float dx, float u[], int rank, int numProcessors)
{
  char filename[50];
  //Create the filename based on the time step.
  sprintf(filename, "tc2d_%08.6f.dat", time);

  int i, j, id;

  float x, y;
   
  float *u_local       = malloc((N_loc)*(N_loc)*sizeof(float));
  float *u_global      = malloc((N)*(N)*sizeof(float));
  float *u_write       = malloc((N)*(N)*sizeof(float));
  
  //pack the data for gather (to avoid sending ghosts)
  int id_loc = 0;
  for(j=1;j<N_loc+1;j++)
  {
    for(i=1;i<N_loc+1;i++)
    {
      id = ID_2D(i,j,N_loc);
      u_local[id_loc] = u[id];
      id_loc++;
    }
  }

  //gather data on rank 0
  MPI_Gather(u_local,id_loc,MPI_FLOAT,u_global,id_loc,MPI_FLOAT,0,MPI_COMM_WORLD);

  //unpack data so that it is in nice array format
  int id_write, id_global;
  int irank_x, irank_y;
  int q = sqrt(numProcessors);

  if(rank==0)
  {
  
    for(int p=0; p<numProcessors;p++){
      irank_x = p%q;
      irank_y = p/q;
      for(j=0;j<N_loc;j++){
	for(i=0;i<N_loc;i++){
	  id_global = p*N_loc*N_loc + j*N_loc + i;
	  id_write  = irank_x*N_loc*N_loc*q + j*N_loc*q + irank_y*N_loc + i;

	  u_write[id_write] = u_global[id_global];
	}
      }
    }

    //Open the file.
  FILE *file = fopen (filename, "wt" );
    
  if (!file)
  {
    fprintf (stderr, "\n" );

    fprintf (stderr, "WRITE_RESULTS - Fatal error!\n");

    fprintf (stderr, "  Could not open the output file.\n");

    exit (1);
  }
    //Write the data.
    for ( i = 0; i < N; i++ ) 
      for ( j = 0; j < N; j++ ){
        id=j*N+i;
	      x = i*dx; //I am a bit lazy here with not gathering x and y
	      y = j*dx;
	
	fprintf ( file, "%24.16g\t%24.16g\t%24.16g\t%24.16g\t%24.16g\n", x, y,u_write[id], 0.0, 0.0); //added extra zeros for backward-compatibility with plotting routines
      }

    //Close the file.
    fclose ( file );

  }
  free(u_global); 
  free(u_write);
  free(u_local);
  return;
}
/******************************************************************************/

void initialConditions(int nx_local, int ny_local, int x_start, int y_start, float dx, float dy, int px, int py, int px_size, int py_size, float x_length, float y_length, float *x, float *y, float *h, float *uh, float *vh)
{
  int i, j, id, id_ghost;

  int global_i, global_j;

  for (j = 0; j < nx_local; j++) 
  {
    global_j = x_start + j;

    x[j] = -x_length / 2.0f + dx / 2.0f + global_j * dx;
  }

  for (i = 0; i < ny_local; i++) 
  {
    global_i = y_start + i;
    
    y[i] = -y_length / 2.0f + dy / 2.0f + global_i * dy;
  }

  for (i = 0; i < ny_local; i++) 
    for (j = 0; j < nx_local; j++) 
    {
      id = ID_2D(i + 1, j + 1, nx_local);

      float xx = x[j];
      float yy = y[i];
      
      h[id] = 1.0f + 0.4f * expf(-5.0f * (xx * xx + yy * yy));

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }

  // Apply physical domain boundary conditions
  // Bottom boundary
  if (py == 0) 
  {  
    for (j = 1; j < nx_local + 1; j++) 
    {
      id = ID_2D(0, j, nx_local);

      id_ghost = ID_2D(1, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = -uh[id_ghost];

      vh[id] = vh[id_ghost];
    }
  }

  // Top boundary
  if (py == py_size - 1) 
  { 
    for (j = 1; j < nx_local + 1; j++) 
    {
      id = ID_2D(ny_local + 1, j, nx_local);

      id_ghost = ID_2D(ny_local, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = -uh[id_ghost];

      vh[id] = vh[id_ghost];
    }
  }

  // Left boundary
  if (px == 0) 
  {  
    for (i = 1; i <= ny_local; i++) 
    {
      id = ID_2D(i, 0, nx_local);

      id_ghost = ID_2D(i, 1, nx_local);

      h[id] = h[id_ghost];

      uh[id] = uh[id_ghost];

      vh[id] = -vh[id_ghost];
    }
  }

  // Right boundary
  if (px == px_size - 1) 
  {  
    for (i = 1; i <= ny_local; i++) 
    {
      id = ID_2D(i, nx_local + 1, nx_local);

      id_ghost = ID_2D(i, nx_local, nx_local);

      h[id] = h[id_ghost];
      
      uh[id] = uh[id_ghost];

      vh[id] = -vh[id_ghost];
    }
  }  

  return;
}
/******************************************************************************/

void haloExchange(float *data, int nx_local, int ny_local, MPI_Comm cart_comm, MPI_Datatype column_type, int north, int south, int west, int east, int base_tag)
{
  MPI_Request requests[8];

  // Row-wise communication (top/bottom)
  MPI_Isend(&data[ID_2D(1, 1, nx_local)], nx_local, MPI_FLOAT, north, base_tag + 0, cart_comm, &requests[0]);
  MPI_Irecv(&data[ID_2D(ny_local + 1, 1, nx_local)], nx_local, MPI_FLOAT, south, base_tag + 0, cart_comm, &requests[1]);

  MPI_Isend(&data[ID_2D(ny_local, 1, nx_local)], nx_local, MPI_FLOAT, south, base_tag + 1, cart_comm, &requests[2]);
  MPI_Irecv(&data[ID_2D(0, 1, nx_local)], nx_local, MPI_FLOAT, north, base_tag + 1, cart_comm, &requests[3]);

  // Column-wise communication (left/right)
  MPI_Isend(&data[ID_2D(1, 1, nx_local)], 1, column_type, west, base_tag + 2, cart_comm, &requests[4]);
  MPI_Irecv(&data[ID_2D(1, nx_local + 1, nx_local)], 1, column_type, east, base_tag + 2, cart_comm, &requests[5]);

  MPI_Isend(&data[ID_2D(1, nx_local, nx_local)], 1, column_type, east, base_tag + 3, cart_comm, &requests[6]);
  MPI_Irecv(&data[ID_2D(1, 0, nx_local)], 1, column_type, west, base_tag + 3, cart_comm, &requests[7]);

  MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
}
/******************************************************************************/

/****************************************************************************** MAIN FUNCTION ******************************************************************************/
int main (int argc, char *argv[])
{
  /****************************************************************************** INSTANTIATION ******************************************************************************/
  
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Initialize variables
  // MPI variables
  int px; 
  int py;

  int px_size;
  int py_size;

  int rank;
  int numProcessors;

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

  double time_start;
  double time_end;
  double time_elapsed;
  double time_max;

  int i, j, k, l, m;

  int id;   
  int id_left;
  int id_right;
  int id_bottom;
  int id_top;

  int nx_global; 
  int ny_global;

  int nx_local;
  int ny_local;

  int x_start;
  int y_start;

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

  float *hm;
  float *uhm;
  float *vhm;

  float *x;
  float *y;

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);

  // Get command line arguments
  getArgs(&nx_global, &ny_global, &dt, &x_length, &y_length, &totalRuntime, argc, argv);

  // Define the locations of the nodes, time steps, and spacing
  dx = x_length / ( float ) ( nx_global );
  dy = y_length / ( float ) ( ny_global );

  // Define the time step and the grid spacing
  float lambda_x = 0.5f * (float) dt/dx;
  float lambda_y = 0.5f * (float) dt/dy;

  // Create a Cartesian topology
  MPI_Dims_create(numProcessors, 2, dims);

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, rank, 2, coords);
  
  px = coords[1];  // x-axis rank
  py = coords[0];  // y-axis rank

  px_size = dims[1]; // number of processes in x direction
  py_size = dims[0]; // number of processes in y direction

  // Calculate the local grid numProcessors
  nx_local = computeLocalSize(nx_global, px, px_size);
  ny_local = computeLocalSize(ny_global, py, py_size);

  x_start = computeGlobalStart(px, nx_global, px_size);
  y_start = computeGlobalStart(py, ny_global, py_size);

  for (l = 0; l < numProcessors; l++) 
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == l) 
    {
      printf("Rank %d: Global x-start position for rank %d: %d, Global y-start position for rank %d: %d\n", rank, rank, x_start, rank, y_start);
      fflush(stdout); // Ensure immediate flush to console
    }
  }

  // Define column data type for vertical halo exchange
  MPI_Datatype column_type;
  MPI_Type_vector(ny_local, 1, nx_local + 2, MPI_FLOAT, &column_type);
  MPI_Type_commit(&column_type);

  // Identify neighbors in Cartesian grid
  MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // shift in y-direction (rows)
  MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // shift in x-direction (columns)

  /****************************************************************************** ALLOCATE MEMORY ******************************************************************************/
  //Allocate space (nx_global+2)((nx_global+2) long, to account for ghosts
  // height array
  h  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  hm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  gh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  // x-momentum array
  uh  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  uhm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fuh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  guh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  //  y-momentum array
  vh  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  vhm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  gvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));

  // coordinate Array
  x = malloc((nx_local) * sizeof(float));
  y = malloc((ny_local) * sizeof(float));
  
  /****************************************************************************** MAIN LOOP ******************************************************************************/
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("\n");
    printf ("SHALLOW_WATER_2D:\n");
    printf (" A program to solve the shallow water equations.\n");
    printf (" The problem is solved on a rectangular grid.\n");
    printf (" The grid has %d x %d nodes.\n", nx_global, ny_global);
    printf (" The total program runtime is %g.\n", totalRuntime);
    printf (" The time step is %g.\n", dt);
    printf (" The grid spacing is %g x %g.\n", dx, dy);
    printf (" The grid length is %g x %g.\n", x_length, y_length);
    printf (" The number of processes is %d.\n", numProcessors);
    printf (" The processor grid dimensions are %d x %d.\n", dims[0], dims[1]);
  }

  // **** INITIAL CONDITIONS ****

  for (k = 1; k < 11; k++)
  {
    programRuntime = 0.0f;
    m = 0;

    initialConditions(nx_local, ny_local, x_start, y_start, dx, dy, px, py, px_size, py_size, x_length, y_length, x, y, h, uh, vh);

    MPI_Barrier(cart_comm);
    // Start timing the program
    time_start = MPI_Wtime();

    // **** TIME LOOP ****
    while (programRuntime < totalRuntime) 
    {
      programRuntime += dt; 

      // === h field ===
      haloExchange(h, nx_local, ny_local, cart_comm, column_type, north, south, west, east, 0);

      // === uh field ===
      haloExchange(uh, nx_local, ny_local, cart_comm, column_type, north, south, west, east, 4);

      // === vh field ===
      haloExchange(vh, nx_local, ny_local, cart_comm, column_type, north, south, west, east, 8);

      for (i = 1; i <= ny_local; i++)   
        for (j = 1; j <= nx_local; j++) 
        {
          id = ID_2D(i, j, nx_local);

          fh[id] = uh[id];

          fuh[id] = uh[id] * uh[id] / h[id] + 0.5f * g * h[id] * h[id];

          fvh[id] = uh[id] * vh[id] / h[id];

          gh[id] = vh[id];

          guh[id] = uh[id] * vh[id] / h[id];

          gvh[id] = vh[id] * vh[id] / h[id] + 0.5f * g * h[id] * h[id];
        }

      for (i = 1; i <= ny_local; i++)
        for (j = 1; j <= nx_local; j++) 
        {
          id = ID_2D(i, j, nx_local);
          id_left = ID_2D(i, j - 1, nx_local);
          id_right = ID_2D(i, j + 1, nx_local);
          id_bottom = ID_2D(i - 1, j, nx_local);
          id_top = ID_2D(i + 1, j, nx_local);

          hm[id] = 0.25f * (h[id_left] + h[id_right] + h[id_bottom] + h[id_top])
                - lambda_x * (fh[id_right] - fh[id_left])
                - lambda_y * (gh[id_top] - gh[id_bottom]);

          uhm[id] = 0.25f * (uh[id_left] + uh[id_right] + uh[id_bottom] + uh[id_top])
                - lambda_x * (fuh[id_right] - fuh[id_left])
                - lambda_y * (guh[id_top] - guh[id_bottom]);

          vhm[id] = 0.25f * (vh[id_left] + vh[id_right] + vh[id_bottom] + vh[id_top])
                - lambda_x * (fvh[id_right] - fvh[id_left])
                - lambda_y * (gvh[id_top] - gvh[id_bottom]);
        }
      
      for (i = 1; i < ny_local + 1; i++)
        for (j = 1; j < nx_local + 1; j++)
        {
          id = ID_2D(i, j, nx_local);

          h[id] = hm[id];

          uh[id] = uhm[id];

          vh[id] = vhm[id];
        }

      // === LEFT boundary (global domain) ===
      if (py == 0) 
      {
        j = 1;
        for (i = 1; i <= ny_local; i++) 
        {
          id = ID_2D(i, j, nx_local);
          id_left = ID_2D(i, j - 1, nx_local);

          h[id_left] = h[id];

          uh[id_left] = -uh[id];   // reverse x-momentum

          vh[id_left] = vh[id];
        }
      }

      // === RIGHT boundary (global domain) ===
      if (py == py_size - 1) 
      {
        j = nx_local;
        for (i = 1; i <= ny_local; i++) 
        {
          id = ID_2D(i, j, nx_local);
          id_right = ID_2D(i, j + 1, nx_local);

          h[id_right] = h[id];

          uh[id_right] = -uh[id];

          vh[id_right] = h[id];
        }
      }

      // === BOTTOM boundary (global domain) ===
      if (px == 0) 
      {
        i = 1;
        for (j = 1; j <= nx_local; j++) 
        {
          id = ID_2D(i, j, nx_local);
          id_bottom = ID_2D(i - 1, j, nx_local);

          h[id_bottom] = h[id];

          uh[id_bottom] = uh[id];

          vh[id_bottom] = -vh[id];  // reverse y-momentum
        }
      }

      // === TOP boundary (global domain) ===
      if (px == px_size - 1) 
      {
        i = ny_local;
        for (j = 1; j <= nx_local; j++) 
        {
          id = ID_2D(i, j, nx_local);
          id_top = ID_2D(i + 1, j, nx_local);

          h[id_top] = h[id];

          uh[id_top] = uh[id];

          vh[id_top] = -vh[id];
        }
      }

      m++;
    }

    // Stop timing the program
    time_end = MPI_Wtime();
    time_elapsed = time_end - time_start;
    MPI_Reduce(&time_elapsed, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank == 0) 
    {
      printf("Problem size: %d, iteration: %d, Time steps: %d, Elapsed time: %f s\n", nx_global, k, m, time_elapsed);
    }
  }
  /****************************************************************************** Post-Processing ******************************************************************************/

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
  free ( hm );
  free ( uhm );
  free ( vhm );
  free ( x );
  free ( y );

  MPI_Type_free(&column_type);
  MPI_Comm_free(&cart_comm);

  if(rank == 0)
  {
    printf("All processes have completed successfully.\n");
  }
  fflush(stdout); // Ensure immediate flush to console

  // Finalize MPI environment
  MPI_Finalize();

  return 0;
  }
  /***************************************************************************** END OF MAIN FUNCTION ****************************************************************************/
