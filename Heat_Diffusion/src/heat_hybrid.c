#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 103
#define M 100
#define MAX_ITERS 10000
#define TOL 1e-3

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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    #pragma omp parallel for collapse(2)
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

    int iter = 0;
    double global_diff = 1e9;
    double t_start = MPI_Wtime();

    while (iter < MAX_ITERS && global_diff > TOL) {
        // 1. Exchange ghost rows
        if (rank > 0) {
            MPI_Sendrecv(u[1], M, MPI_DOUBLE, rank - 1, 0,
                         u[0], M, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(u[local_N], M, MPI_DOUBLE, rank + 1, 0,
                         u[local_N + 1], M, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 2. Jacobi update
        double local_diff = 0.0;
        #pragma omp parallel for collapse(2) reduction(max:local_diff)
        for (int i = 1; i <= local_N; i++) {
            for (int j = 1; j < M - 1; j++) {
                int global_i = global_start + (i - 1);
                if (global_i == 0 || global_i == N - 1) continue;

                new_u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                      u[i][j - 1] + u[i][j + 1]);
                double d = fabs(new_u[i][j] - u[i][j]);
                if (d > local_diff) local_diff = d;
            }
        }

        // 3. Copy back
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= local_N; i++) {
            for (int j = 1; j < M - 1; j++) {
                u[i][j] = new_u[i][j];
            }
        }

        // 4. Global convergence
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
