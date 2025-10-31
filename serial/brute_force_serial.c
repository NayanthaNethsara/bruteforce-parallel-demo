#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <time.h>

const char *CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 
const int CS_LEN = 36;

void sha256_hex(const unsigned char *data, size_t data_len, char out_hex[65]) {
    unsigned char hash[32];
    unsigned int len = 0;
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) { fprintf(stderr,"EVP_MD_CTX_new failed\n"); exit(1); }
    if (1 != EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL)) { fprintf(stderr,"DigestInit failed\n"); exit(1); }
    if (1 != EVP_DigestUpdate(mdctx, data, data_len)) { fprintf(stderr,"DigestUpdate failed\n"); exit(1); }
    if (1 != EVP_DigestFinal_ex(mdctx, hash, &len)) { fprintf(stderr,"DigestFinal failed\n"); exit(1); }
    EVP_MD_CTX_free(mdctx);

    for (int i = 0; i < 32; ++i)
        sprintf(out_hex + (i * 2), "%02x", hash[i]);
    out_hex[64] = 0;
}

/* increment indices vector that represents a candidate in base CS_LEN
   returns 0 when wrapped (no more combinations), 1 otherwise */
int increment_indices(int *idx, int len) {
    for (int i = len - 1; i >= 0; --i) {
        if (idx[i] + 1 < CS_LEN) {
            idx[i]++;
            return 1;
        } else {
            idx[i] = 0;
        }
    }
    return 0; // overflowed all positions
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_sha256_hex> <max_len>\n", argv[0]);
        return 1;
    }
    const char *target = argv[1];
    int max_len = atoi(argv[2]);
    if (strlen(target) != 64) {
        fprintf(stderr, "Error: target SHA-256 must be 64 hex chars.\n");
        return 1;
    }

    OpenSSL_add_all_algorithms(); // ensure OpenSSL algorithms available

    clock_t t0 = clock();
    char candidate[256];
    char hash_hex[65];
    int found = 0;

    // Try lengths from 1..max_len
    for (int len = 1; len <= max_len && !found; ++len) {
        int *indices = calloc(len, sizeof(int)); // initial all zeros
        // build first candidate
        for (int i = 0; i < len; ++i) candidate[i] = CHARSET[0];
        candidate[len] = 0;

        while (1) {
            // compute hash
            sha256_hex((unsigned char*)candidate, len, hash_hex);
            if (strcmp(hash_hex, target) == 0) {
                double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
                printf("FOUND: \"%s\" (len=%d) in %.3f seconds\n", candidate, len, elapsed);
                found = 1;
                break;
            }
            // increment candidate
            if (!increment_indices(indices, len)) break; // no more combos for this length
            for (int i = 0; i < len; ++i) candidate[i] = CHARSET[indices[i]];
        }
        free(indices);
    }

    if (!found) {
        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("Not found up to length %d. Time: %.3f seconds\n", max_len, elapsed);
    }
    return 0;
}
