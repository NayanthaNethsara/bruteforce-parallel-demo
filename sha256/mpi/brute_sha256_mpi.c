#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <mpi.h>

const char *CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 
const int CS_LEN = 36;

void sha256_hex(const unsigned char *data, size_t data_len, char out_hex[65]) {
    unsigned char hash[32];
    unsigned int len = 0;
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) { fprintf(stderr,"EVP_MD_CTX_new failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (1 != EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL)) { fprintf(stderr,"DigestInit failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (1 != EVP_DigestUpdate(mdctx, data, data_len)) { fprintf(stderr,"DigestUpdate failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (1 != EVP_DigestFinal_ex(mdctx, hash, &len)) { fprintf(stderr,"DigestFinal failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    EVP_MD_CTX_free(mdctx);

    for (int i = 0; i < 32; ++i)
        sprintf(out_hex + (i * 2), "%02x", hash[i]);
    out_hex[64] = 0;
}

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

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <target_sha256_hex> <max_len>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *target = argv[1];
    int max_len = atoi(argv[2]);

    if (rank == 0) {
        printf("SHA256 MPI Brute Force: Target=\"%s\", max_len=%d, processes=%d\n", target, max_len, size);
    }

    double t0 = MPI_Wtime();
    char candidate[64];
    char hash_hex[65];

    for (int len = 1; len <= max_len; ++len) {
        // Partition based on first character
        for (int i = rank; i < CS_LEN; i += size) {
            candidate[0] = CHARSET[i];
            
            int suffix_len = len - 1;
            int *indices = calloc(suffix_len, sizeof(int));
            if (!indices && suffix_len > 0) { perror("calloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

            // Init suffix
            for (int k = 0; k < suffix_len; ++k) candidate[1 + k] = CHARSET[0];
            candidate[len] = '\0';

            while (1) {
                sha256_hex((unsigned char*)candidate, len, hash_hex);
                if (strcmp(hash_hex, target) == 0) {
                    printf("Rank %d FOUND: \"%s\" (len=%d) in %.4f s\n", rank, candidate, len, MPI_Wtime() - t0);
                    free(indices);
                    MPI_Abort(MPI_COMM_WORLD, 0);
                }

                if (suffix_len == 0) break; // length 1, no suffix to increment

                if (!increment_indices(indices, suffix_len, CS_LEN)) break;
                for (int k = 0; k < suffix_len; ++k) candidate[1 + k] = CHARSET[indices[k]];
            }
            free(indices);
        }
    }

    MPI_Finalize();
    return 0;
}
