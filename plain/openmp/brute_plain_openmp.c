#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

    clock_t t0 = clock();

    int found = 0;
    char found_str[64];

    for (int len = 1; len <= max_len && !found; len++) {

        long long total = 1;
        for (int i = 0; i < len; i++) total *= charset_len;

        printf("Trying length %d → %lld combinations\n", len, total);

        #pragma omp parallel shared(found, found_str)
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();

            long long chunk = total / nth;
            long long start = tid * chunk;
            long long end = (tid == nth - 1) ? total : start + chunk;

            int *indices = malloc(sizeof(int) * len);
            char *candidate = malloc(len + 1);

            int_to_indices(start, charset_len, len, indices);

            for (long long n = start; n < end; n++) {
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
                    break;
                }

                increment_indices(indices, len, charset_len);
            }

            free(indices);
            free(candidate);
        }

        if (found) {
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("\nFOUND: \"%s\" (len=%d) in %.4f s\n", found_str, len, elapsed);
            break;
        }
    }

    if (!found) {
        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("Not found (max_len=%d). Time: %.4f s\n", max_len, elapsed);
    }

    return 0;
}
