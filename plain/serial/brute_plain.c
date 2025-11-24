#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <target_password> <max_len>\n", argv[0]);
        return 1;
    }
    const char *target = argv[1];
    int max_len = atoi(argv[2]);
    if (max_len <= 0 || max_len > 10) {
        fprintf(stderr, "Choose max_len between 1 and 10 (small for tests).\n");
        return 1;
    }

    int charset_len = strlen(CHARSET);
    size_t target_len = strlen(target);
    printf("Target: \"%s\" (len=%zu), max_len=%d, charset_len=%d\n",
           target, target_len, max_len, charset_len);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    char candidate[64];

    int found = 0;
    for (int len = 1; len <= max_len && !found; ++len) {
        int *indices = calloc(len, sizeof(int));
        if (!indices) { perror("calloc"); return 1; }
        // build first candidate (all CHARSET[0])
        for (int i = 0; i < len; ++i) candidate[i] = CHARSET[0];
        candidate[len] = '\0';

        while (1) {
            // Compare only if lengths match
            if (strlen(target) == (size_t)len) {
                if (strcmp(candidate, target) == 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
                    printf("FOUND: \"%s\" (len=%d) in %.4f s\n", candidate, len, elapsed);
                    found = 1;
                    break;
                }
            }
            if (!increment_indices(indices, len, charset_len)) break;
            for (int i = 0; i < len; ++i) candidate[i] = CHARSET[indices[i]];
        }
        free(indices);
    }

    if (!found) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Not found up to length %d. Time: %.4f s\n", max_len, elapsed);
    }
    return 0;
}
