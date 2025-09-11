#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 103
#define M 100
#define MAX_ITERS 10000
#define TOL 1e-3
#define Ti 10
#define Tj 10

// Allocate a contiguous 2D array with row pointers
double **alloc_2d(int n, int m, double **data_block) {
    *data_block = malloc(n * m * sizeof(double));
    double **array = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        array[i] = &(*data_block)[i * m];
    }
    return array;
}

int main(int argc, char **argv) {
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            fprintf(stderr, "Error: MPI does not provide THREAD_MULTIPLE\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Row distribution
    int rows_per_rank = N / size;
    int extra = N % size;

    int local_N = rows_per_rank + (rank < extra ? 1 : 0);

    // Global starting row for this rank
    int global_start = rank * rows_per_rank + (rank < extra ? rank : extra);

    // Allocate with ghost rows
    int local_rows = local_N + 2;
    double *data, *new_data;
    double **u = alloc_2d(local_rows, M, &data);
    double **new_u = alloc_2d(local_rows, M, &new_data);

    // Initialize
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop collapse(2)
            {
                for (int i = 0; i < local_rows; i++) {
                    for (int j = 0; j < M; j++) {
                        int global_i = global_start + (i - 1);
        
                        if (global_i == 0 || j == M - 1) {
                            u[i][j] = 100.0;   // hot boundary
                        } else {
                            u[i][j] = 0.0;
                        }
                        new_u[i][j] = u[i][j];
                    }
                }
            }
        }
        #pragma omp taskwait
    }

    int iter = 0;
    double global_diff = 1e9;
    double t_start = MPI_Wtime();

    while (iter < MAX_ITERS && global_diff > TOL) {
        // 1. Exchange ghost rows
        if (rank > 0) {
            #pragma omp task depend(out: u[0][0:M])
            {
                MPI_Sendrecv(u[1], M, MPI_DOUBLE, rank - 1, 0,
                             u[0], M, MPI_DOUBLE, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (rank < size - 1) {
            #pragma omp task depend(out: u[local_N + 1][0:M])
            {
                MPI_Sendrecv(u[local_N], M, MPI_DOUBLE, rank + 1, 0,
                             u[local_N + 1], M, MPI_DOUBLE, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // 2. Jacobi update
        double local_diff = 0.0;
        #pragma omp task depend(out: local_diff)
        {
            #pragma omp parallel
            {
                #pragma omp single
                #pragma omp taskgroup task_reduction(max: local_diff)
                {
                    for (int ii = 1; ii <= local_N; ii += Ti) {
                        for (int jj = 1; jj < M - 1; jj += Tj) {
                            int iend = (ii + Ti - 1 < local_N) ? (ii + Ti - 1) : local_N;
                            int jend = (jj + Tj - 1 < M - 2) ? (jj + Tj - 1) : (M - 2);

                            // Top boundary tiles (depend on top halo)
                            if (ii == 1 && rank > 0) {
                                #pragma omp task depend(in: u[0][0:M]) in_reduction(max: local_diff) shared(u,new_u)
                                {
                                    double division_diff = 0.0;
                                    for (int i = ii; i <= iend; i++) {
                                        for (int j = jj; j <= jend; j++) {
                                            double val = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                                                u[i][j - 1] + u[i][j + 1]);
                                            double d = fabs(val - u[i][j]);
                                            if (d > division_diff) division_diff = d;
                                            new_u[i][j] = val;
                                        }
                                    }
                                    local_diff = division_diff;
                                }
                            }
                            // Bottom boundary tiles (depend on bottom halo)
                            else if (iend == local_N && rank < size - 1) {
                                #pragma omp task depend(in: u[local_N+1][0:M]) in_reduction(max: local_diff) shared(u,new_u)
                                {
                                    double division_diff = 0.0;
                                    for (int i = ii; i <= iend; i++) {
                                        for (int j = jj; j <= jend; j++) {
                                            double val = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                                                u[i][j - 1] + u[i][j + 1]);
                                            double d = fabs(val - u[i][j]);
                                            if (d > division_diff) division_diff = d;
                                            new_u[i][j] = val;
                                        }
                                    }
                                    local_diff = division_diff;
                                }
                            }
                            // Interior tiles (no halo dependency)
                            else {
                                #pragma omp task in_reduction(max: local_diff) shared(u,new_u)
                                {
                                    double division_diff = 0.0;
                                    for (int i = ii; i <= iend; i++) {
                                        for (int j = jj; j <= jend; j++) {
                                            double val = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                                                u[i][j - 1] + u[i][j + 1]);
                                            double d = fabs(val - u[i][j]);
                                            if (d > division_diff) division_diff = d;
                                            new_u[i][j] = val;
                                        }
                                    }
                                    local_diff = division_diff;
                                }
                            }
                        }
                    }
                }
            }
        }
        // 3. Copy back
        #pragma omp task depend(in: local_diff)
        {
            #pragma omp taskloop collapse(2)
            for (int i = 1; i <= local_N; i++) {
                for (int j = 1; j < M - 1; j++) {
                    u[i][j] = new_u[i][j];
                }
            }
        }
        

        // 4. Global convergence
        #pragma omp task depend(in: local_diff) depend(out: global_diff)
        {
            MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }


        iter++;
    }

    
    double t_end = MPI_Wtime();
    if (rank == 0) {
        printf("Time elapsed = %f seconds\n", t_end - t_start);
        printf("Finished at iter %d with global diff = %f\n", iter, global_diff);
    }

    free(u);
    free(new_u);
    free(data);
    free(new_data);

    MPI_Finalize();
    return 0;
}
