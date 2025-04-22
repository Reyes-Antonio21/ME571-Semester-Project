# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <mpi.h>

# define ID_2D(i,j,nx_global) ((i)*(nx_global+2)+(j))

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

void getArgs(int *nx_global, int *ny_global, double *dt, float *x_length, float *y_length, double *totalRuntime, int argc, char *argv[])
{
  if (argc <= 1)
  {
    *nx_global = 800;
  }
  else
  {
    *nx_global = atoi (argv[1]);
  }

  if (argc <= 2)
  {
    *ny_global = 800;
  }
  else
  {
    *ny_global = atoi (argv[2]);
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

void initialConditions(int nx_local, int ny_local, int nx_global, int ny_global, int px, int py, float dx, float dy, float x_length, float y_length, int dims[2], float *h, float *uh, float *vh)
{
  int i, j, id, id_ghost;

  int global_i, global_j;

  float *x_coords = malloc((nx_local + 2) * sizeof(float));
  float *y_coords = malloc((ny_local + 2) * sizeof(float));

  int global_x_start = computeGlobalStart(px, nx_global, dims[1]);
  int global_y_start = computeGlobalStart(py, ny_global, dims[0]);

  for (j = 1; j < nx_local + 1; j++) 
  {
    global_j = global_x_start + j - 1;

    x_coords[j] = -x_length / 2 + dx / 2 + global_j * dx;
  }

  for (i = 1; i < ny_local + 1; i++) 
  {
    global_i = global_y_start + i - 1;
    
    y_coords[i] = -y_length / 2 + dy / 2 + global_i * dy;
  }

  for (i = 1; i < ny_local + 1; i++) 
    for (j = 1; j < nx_local + 1; j++) 
    {
      id = ID_2D(i, j, nx_local);

      float x = x_coords[j];
      float y = y_coords[i];
      
      h[id] = 1.0f + 0.4f * expf(-5.0f * (x * x + y * y));

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }

  // Apply physical domain boundary conditions
  // Bottom boundary
  if (px == 0) 
  {  
    for (j = 1; j <= nx_local; j++) 
    {
      id = ID_2D(0, j, nx_local);

      id_ghost = ID_2D(1, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Top boundary
  if (px == dims[0] - 1) 
  { 
    for (j = 1; j <= nx_local; j++) 
    {
      id = ID_2D(ny_local + 1, j, nx_local);

      id_ghost = ID_2D(ny_local, j, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Left boundary
  if (py == 0) 
  {  
    for (i = 1; i <= ny_local; i++) 
    {
      id = ID_2D(i, 0, nx_local);

      id_ghost = ID_2D(i, 1, nx_local);

      h[id] = h[id_ghost];

      uh[id] = 0.0f;

      vh[id] = 0.0f;
    }
  }

  // Right boundary
  if (py == dims[1] - 1) 
  {  
    for (i = 1; i <= ny_local; i++) 
    {
      id = ID_2D(i, nx_local + 1, nx_local);

      id_ghost = ID_2D(i, nx_local, nx_local);

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

void write_results_mpi(char *output_filename, int nx_global, int ny_global, int nx_local, int ny_local, float x[], float y[], float h[], float uh[], float vh[], int rank, int numProcessors)
{
  int i, j;

  int idx, idy,;

  char filename[50];

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
   
  float *h_local  = malloc((nx_local) * (ny_local) * sizeof(float));
  float *h_global = malloc((nx_global) * (ny_global) * sizeof(float));
  float *h_write  = malloc((nx_global) * (ny_global) * sizeof(float));

  float *uh_local  = malloc((nx_local) * (ny_local) * sizeof(float));
  float *uh_global = malloc((nx_global) * (ny_global) * sizeof(float));
  float *uh_write  = malloc((nx_global) * (ny_global) * sizeof(float));

  float *vh_local  = malloc((nx_local) * (ny_local) * sizeof(float));
  float *vh_global = malloc((nx_global) * (ny_global) * sizeof(float));
  float *vh_write  = malloc((nx_global) * (ny_global) * sizeof(float));

  float *x_global = malloc((nx_global) * sizeof(float));
  float *x_local  = malloc((nx_local) * sizeof(float));
  float *x_write  = malloc((nx_global) * sizeof(float));

  float *y_global = malloc((ny_global) * sizeof(float));
  float *y_local  = malloc((ny_local) * sizeof(float));
  float *y_write  = malloc((ny_global) * sizeof(float));
  

  //pack the data for gather (to avoid sending ghosts)
  int id_local = 0;
  for(j = 1; j < nx_local + 1; j++)
  {
    for(i = 1; i < ny_local + 1; i++)
    {
      idx = ID_2D(i,j, nx_local);
      idy = ID_2D(j,i, ny_local);

      h_local[id_local] = u[idx];

      id_local++;
    }
  }

  // Gather data to rank 0
  MPI_Gather(h_local, id_local, MPI_FLOAT, h_global, id_local, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Unpack data into the global array
  int id_write, id_global;
  int rank_x, rank_y;

  if(rank == 0)
  {
    for(int p = 0; p < numProcessors;p++)
    {
      rank_x = p % q;
      rank_y = p/q;
      for(j = 0; j < N_loc; j++)
      {
	      for(i = 0; i < N_loc; i++)
        {
          id_global = p * N_loc * N_loc + j * N_loc + i;
          id_write  = rank_x*N_loc*N_loc*q + j*N_loc*q + rank_y*N_loc + i;
	        h_write[id_write] = h_global[id_global];
	      }
      }
    }

    //Write the data.
    for ( i = 0; i < ny_global; i++ ) 
      for ( j = 0; j < nx_global; j++ )
      {
        id = j * N + i;
	
	      fprintf ( output, "  %24.16g\t%24.16g\t%24.16g\t%24.16g\t%24.16g\n", x, y, h_write[id], 0.0, 0.0); //added extra zeros for backward-compatibility with plotting routines
      }

    //Close the file.
    fclose ( output );
  }

  free(h_global); 
  free(h_write);
  free(h_local);

  free(uh_global); 
  free(uh_write);
  free(uh_local);

  free(vh_global); 
  free(vh_write);
  free(vh_local);

  free(x_global);
  free(x_write);
  free(x_local);

  free(y_global);
  free(y_write);
  free(y_local);

  return;
}
/******************************************************************************/

int main (int argc, char *argv[])
{
  /****************************************************************************** INSTANTIATION ******************************************************************************/
  
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Initialize variables
  // MPI variables
  int px; 
  int py;

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
  int nx_extra;
  int ny_extra;

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

  nx_local = nx_global / dims[1];
  ny_local = ny_global / dims[0];

  nx_extra = nx_global % dims[1];
  ny_extra = ny_global % dims[0];

  if (px < nx_extra)
  {
    nx_local++;
  }

  if (py < ny_extra)
  {
    ny_local++;
  }

  x_start = (nx_local * px + (px < nx_extra ? px : nx_extra));
  y_start = (ny_local * py + (py < ny_extra ? py : ny_extra));

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
  //height array
  h  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  hm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  gh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  //x momentum array
  uh  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  uhm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fuh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  guh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  //y momentum array
  vh  = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  vhm = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  fvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  gvh = (float*) malloc((nx_local + 2) * (ny_local + 2) * sizeof(float));
  
  /****************************************************************************** MAIN LOOP ******************************************************************************/
  if (rank == 0)
  {
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

  for (k = 0; k < 10; k++)
  {
    programRuntime = 0.0f;
    m = 0;

    initialConditions(nx_local, ny_local, nx_global, ny_global, px, py, dx, dy, x_length, y_length, dims, h, uh, vh);

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
      if (py == dims[1] - 1) 
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
      if (px == dims[0] - 1) 
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