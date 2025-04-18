# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <mpi.h>

# define ID_2D(i,j,nx_global) ((i)*(nx_global+2)+(j))
# define FILE_NAME "swe_2d.h5"

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

void initialConditions(int nx_local, int ny_local, int rank_x, int rank_y, int dims[2], int nx_global, int ny_global, float x_length, float y_length, float dx, float dy, float *h, float *uh, float *vh)
{
  int i, j, id, id_ghost;

  float *x_coords = malloc((nx_local + 2) * sizeof(float));
  float *y_coords = malloc((ny_local + 2) * sizeof(float));

  int global_x_start = computeGlobalStart(rank_x, nx_global, dims[1]);
  int global_y_start = computeGlobalStart(rank_y, ny_global, dims[0]);

  for (j = 0; j < nx_local + 2; j++) 
  {
    int global_j = global_x_start + j - 1;

    x_coords[j] = -x_length / 2 + dx / 2 + global_j * dx;
  }

  for (i = 0; i < ny_local + 2; i++) 
  {
    int global_i = global_y_start + i - 1;
    
    y_coords[i] = -y_length / 2 + dy / 2 + global_i * dy;
  }

  for (i = 0; i < ny_local + 2; i++) 
    for (j = 0; j < nx_local + 2; j++) 
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
  if (rank_x == 0) 
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
  if (rank_x == dims[0] - 1) 
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
  if (rank_y == 0) 
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
  if (rank_y == dims[1] - 1) 
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

int main (int argc, char *argv[])
{
  /****************************************************************************** Instantiation ******************************************************************************/
  
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Initialize variables
  // MPI variables
  int rank_x; 
  int rank_y;

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

  int i, j;

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

  // Get command line arguments
  getArgs(&nx_global, &ny_global, &dt, &x_length, &y_length, &totalRuntime, argc, argv);
  
  // Define the locations of the nodes, time steps, and spacing
  dx = x_length / ( float ) ( nx_global );
  dy = y_length / ( float ) ( ny_global );

  // Define the time step and the grid spacing
  float lambda_x = 0.5f * (float) dt/dx;
  float lambda_y = 0.5f * (float) dt/dy;

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Hello from rank %d of %d\n", rank, size);

  // Create a Cartesian topology
  MPI_Dims_create(size, 2, dims);

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, rank, 2, coords);
  
  rank_x = coords[0];  // corresponds to dimension 0 (rows)
  rank_y = coords[1];  // corresponds to dimension 1 (columns)

  nx_local = nx_global / dims[1];
  ny_local = ny_global / dims[0];

  nx_extra = nx_global % dims[1];
  ny_extra = ny_global % dims[0];

  if (rank_x < nx_extra)
  {
    nx_local++;
  }

  if (rank_y < ny_extra)
  {
    ny_local++;
  }

  printf("Rank %d: nx_local = %d, ny_local = %d\n", rank, nx_local, ny_local);

  /****************************************************************************** ALLOCATE MEMORY ******************************************************************************/
  //Allocate space (nx_global+2)((nx_global+2) long, to account for ghosts
  //height array
  h  = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  hm = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  fh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  gh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  
  //x momentum array
  uh  = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  uhm = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  fuh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  guh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  
  //y momentum array
  vh  = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  vhm = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  fvh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  gvh = ( float * ) malloc ( (nx_local + 2) * (ny_local + 2) * sizeof ( float ) );
  
  /****************************************************************************** MAIN LOOP ******************************************************************************/
  if (rank == 0)
  {
    printf ("SHALLOW_WATER_2D:\n");
    printf ("  A program to solve the shallow water equations.\n");
    printf ("\n");
    printf ("  The problem is solved on a rectangular grid.\n");
    printf ("\n");
    printf ("  The grid has %d x %d nodes.\n", nx_global, ny_global);
    printf (" The total program runtime is %g.\n", totalRuntime);
    printf ("  The time step is %g.\n", dt);
    printf ("  The grid spacing is %g x %g.\n", dx, dy);
    printf ("  The grid length is %g x %g.\n", x_length, y_length);
    printf (" The number of processes is %d.\n", size);
    printf (" The processor grid dimensions are %d x %d.\n", dims[0], dims[1]);
  }

  // **** INITIAL CONDITIONS ****

  programRuntime = 0.0f;

  initialConditions(nx_local, ny_local, rank_x, rank_y, dims, nx_global, ny_global, x_length, y_length, dx, dy, h, uh, vh);

  printf("Rank %d: global_x_start = %d, global_y_start = %d\n", rank, computeGlobalStart(rank_x, nx_global, dims[0]), computeGlobalStart(rank_y, ny_global, dims[1]));

  // Define column data type for vertical halo exchange
  MPI_Datatype column_type;
  MPI_Type_vector(ny_local, 1, nx_local + 2, MPI_FLOAT, &column_type);
  MPI_Type_commit(&column_type);

  // Identify neighbors in Cartesian grid
  MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // shift in y-direction (rows)
  MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // shift in x-direction (columns)

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
    if (rank_y == 0) 
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
    if (rank_y == dims[1] - 1) 
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
    if (rank_x == 0) 
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
    if (rank_x == dims[0] - 1) 
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
  free ( hm );
  free ( uhm );
  free ( vhm );

  MPI_Type_free(&column_type);
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();

  printf("Time-stepping loop completed.\n");
    
  printf("Problem size: %d, Time Steps Taken: %f \n", nx_global, programRuntime/dt);

  return 0;
  }
  /***************************************************************************** END OF MAIN FUNCTION ****************************************************************************/