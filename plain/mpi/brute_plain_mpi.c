#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const char *CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 

int increment_indices(int *idx, int len, int base) {
    for (int i = len - 1; i >= 0; --i) {
        if (idx[i] + 1 < base) {
            idx[i]++;
            return 1;
        } else {
            idx[i] = 0;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <target_password> <max_len>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *target = argv[1];
    int max_len = atoi(argv[2]);
    int charset_len = strlen(CHARSET);
    size_t target_len = strlen(target);

    if (rank == 0) {
        printf("MPI Brute Force: Target=\"%s\", max_len=%d, processes=%d\n", target, max_len, size);
    }

    double t0 = MPI_Wtime();
    char candidate[64];
    
    // Iterate through all lengths
    for (int len = 1; len <= max_len; ++len) {
        // Optimization: Skip lengths that don't match target length
        if (len != target_len) continue;

        if (len == 1) {
            // Simple case for length 1
            for (int i = rank; i < charset_len; i += size) {
                candidate[0] = CHARSET[i];
                candidate[1] = '\0';
                if (strcmp(candidate, target) == 0) {
                    printf("Rank %d FOUND: \"%s\" in %.4f s\n", rank, candidate, MPI_Wtime() - t0);
                    MPI_Abort(MPI_COMM_WORLD, 0);
                }
            }
        } else {
            // For length > 1, partition based on the first character
            // Each rank takes every size-th character as the starting character
            for (int i = rank; i < charset_len; i += size) {
                candidate[0] = CHARSET[i];
                
                // The remaining suffix has length (len - 1)
                int suffix_len = len - 1;
                int *indices = calloc(suffix_len, sizeof(int));
                if (!indices) { perror("calloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

                // Initialize suffix to all first chars
                for (int k = 0; k < suffix_len; ++k) candidate[1 + k] = CHARSET[0];
                candidate[len] = '\0';

                while (1) {
                    if (strcmp(candidate, target) == 0) {
                        printf("Rank %d FOUND: \"%s\" in %.4f s\n", rank, candidate, MPI_Wtime() - t0);
                        free(indices);
                        MPI_Abort(MPI_COMM_WORLD, 0);
                    }

                    if (!increment_indices(indices, suffix_len, charset_len)) break;
                    
                    // Update candidate suffix
                    for (int k = 0; k < suffix_len; ++k) candidate[1 + k] = CHARSET[indices[k]];
                }
                free(indices);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
