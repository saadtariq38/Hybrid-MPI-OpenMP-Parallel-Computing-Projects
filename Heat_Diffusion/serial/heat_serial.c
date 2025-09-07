#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100     // rows
#define M 100     // cols
#define MAX_ITERS 10000
#define TOL 1e-3

int main() {
    // Allocate one big block for the grid
    double *data = malloc(N * M * sizeof(double));
    double *new_data = malloc(N * M * sizeof(double));

    // Row pointers into the block
    double **u = malloc(N * sizeof(double *));
    double **new_u = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        u[i]     = &data[i * M];
        new_u[i] = &new_data[i * M];
    }

    // Initialize: hot top row & right col = 100, rest = 0
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (i == 0 || j == M - 1) {
                u[i][j] = 100.0;   // hot boundary
            } else {
                u[i][j] = 0.0;
            }
            new_u[i][j] = u[i][j];
        }
    }

    int iter = 0;
    double diff = 1e9;

    while (iter < MAX_ITERS && diff > TOL) {
        diff = 0.0;

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                new_u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                      u[i][j - 1] + u[i][j + 1]);
                double d = fabs(new_u[i][j] - u[i][j]);
                if (d > diff) diff = d;
            }
        }

        // Copy new_u into u (only interior points)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < M - 1; j++) {
                u[i][j] = new_u[i][j];
            }
        }

        iter++;
        if (iter % 100 == 0)
            printf("Iter %d, max diff = %f\n", iter, diff);
    }

    printf("Finished at iter %d with diff = %f\n", iter, diff);
    printf("Center temp = %f\n", u[N/2][M/2]);

    // Cleanup
    free(u);
    free(new_u);
    free(data);
    free(new_data);

    return 0;
}
