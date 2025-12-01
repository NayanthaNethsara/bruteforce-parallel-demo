#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <omp.h>
#include <time.h>

const char *CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 
const int CS_LEN = 36;

// Returns current time in seconds
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

    printf("SHA256 OpenMP Brute Force: Target=\"%s\", max_len=%d\n", target, max_len);

    double t0 = get_time_sec();
    volatile int found = 0;

    // Iterate through lengths
    for (int len = 1; len <= max_len && !found; ++len) {
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            
            char candidate[64];
            char hash_hex[65];
            int *indices = calloc(len, sizeof(int));
            
            // Partition based on first character
            for (int start_char = thread_id; start_char < CS_LEN; start_char += num_threads) {
                if (found) break;

                indices[0] = start_char;
                // Reset other indices
                for(int k=1; k<len; ++k) indices[k] = 0;
                
                // Build initial candidate for this start_char
                for(int k=0; k<len; ++k) candidate[k] = CHARSET[indices[k]];
                candidate[len] = '\0';
                
                while(1) {
                    if (found) break;

                    sha256_hex((unsigned char*)candidate, len, hash_hex);
                    if (strcmp(hash_hex, target) == 0) {
                        #pragma omp critical
                        {
                            if (!found) {
                                double elapsed = get_time_sec() - t0;
                                printf("FOUND: \"%s\" (len=%d) in %.4f s\n", candidate, len, elapsed);
                                found = 1;
                            }
                        }
                    }
                    
                    if (found) break;

                    // Increment suffix only
                    int carry = 1;
                    for (int i = len - 1; i >= 1; --i) {
                        if (indices[i] + 1 < CS_LEN) {
                            indices[i]++;
                            carry = 0;
                            break;
                        } else {
                            indices[i] = 0;
                        }
                    }
                    
                    if (carry) break; // Done with this start_char prefix
                    
                    // Update candidate string
                    for(int k=1; k<len; ++k) candidate[k] = CHARSET[indices[k]];
                }
            }
            free(indices);
        }
    }

    if (!found) {
        double elapsed = get_time_sec() - t0;
        printf("Not found up to length %d. Time: %.4f s\n", max_len, elapsed);
    }
    return 0;
}
