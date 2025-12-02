#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const char *CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789";

void index_to_candidate(int *idx, int len, char *out) {
    for (int i = 0; i < len; i++)
        out[i] = CHARSET[idx[i]];
    out[len] = '\0';
}

void int_to_indices(long long n, int base, int len, int *idx) {
    for (int i = len - 1; i >= 0; i--) {
        idx[i] = n % base;
        n /= base;
    }
}

void increment_indices(int *idx, int len, int base) {
    for (int i = len - 1; i >= 0; i--) {
        idx[i]++;
        if (idx[i] < base) return;
        idx[i] = 0;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <target_password> <max_len>\n", argv[0]);
        return 1;
    }

    const char *target = argv[1];
    int max_len = atoi(argv[2]);
    int charset_len = strlen(CHARSET);
    int target_len = strlen(target);

    double t0 = omp_get_wtime();

    int found = 0;
    char found_str[64];

    for (int len = 1; len <= max_len && !found; len++) {

        long long total = 1;
        for (int i = 0; i < len; i++) total *= charset_len;

        printf("Trying length %d → %lld combinations\n", len, total);

        #pragma omp parallel shared(found, found_str)
        {
            int *indices = malloc(sizeof(int) * len);
            char *candidate = malloc(len + 1);
            
            // Use a large chunk size to amortize the cost of int_to_indices
            // and to allow incremental updates within the chunk.
            long long chunk_size = 10000; 
            
            #pragma omp for schedule(dynamic, 1) nowait
            for (long long base = 0; base < total; base += chunk_size) {
                if (found) continue;

                long long end = base + chunk_size;
                if (end > total) end = total;

                // Initialize indices for the start of this chunk
                int_to_indices(base, charset_len, len, indices);

                for (long long n = base; n < end; n++) {
                    if (found) break;

                    index_to_candidate(indices, len, candidate);

                    if (len == target_len && strcmp(candidate, target) == 0) {
                        #pragma omp critical
                        {
                            if (!found) {
                                found = 1;
                                strcpy(found_str, candidate);
                            }
                        }
                        break; // Break inner loop
                    }

                    increment_indices(indices, len, charset_len);
                }
            }

            free(indices);
            free(candidate);
        }

        if (found) {
            double elapsed = omp_get_wtime() - t0;
            printf("\nFOUND: \"%s\" (len=%d) in %.4f s\n", found_str, len, elapsed);
            break;
        }
    }

    if (!found) {
        double elapsed = omp_get_wtime() - t0;
        printf("Not found (max_len=%d). Time: %.4f s\n", max_len, elapsed);
    }

    return 0;
}
