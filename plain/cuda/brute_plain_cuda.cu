#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Hardcoded charset on device
__constant__ char d_CHARSET[37]; // 36 chars + null terminator if needed, but we just need chars

const char *h_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 

// Kernel to check passwords
// Each thread checks one index in the search space
__global__ void check_password_kernel(char *d_target, int target_len, int len, unsigned long long total_combinations, int *d_found, char *d_result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_combinations) return;
    if (*d_found) return; // Early exit if already found

    char candidate[16]; // Max length support up to ~10-15
    
    // Convert linear index 'idx' to base-36 string 'candidate'
    unsigned long long temp = idx;
    for (int i = len - 1; i >= 0; --i) {
        candidate[i] = d_CHARSET[temp % 36];
        temp /= 36;
    }
    candidate[len] = '\0';

    // Simple string compare
    bool match = true;
    for (int i = 0; i < len; ++i) {
        if (candidate[i] != d_target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        *d_found = 1;
        // Copy to result
        for (int i = 0; i <= len; ++i) {
            d_result[i] = candidate[i];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <target_password> <max_len>\n", argv[0]);
        return 1;
    }

    const char *target = argv[1];
    int max_len = atoi(argv[2]);
    size_t target_len = strlen(target);
    
    printf("CUDA Brute Force: Target=\"%s\", max_len=%d\n", target, max_len);

    // Copy charset to constant memory
    cudaMemcpyToSymbol(d_CHARSET, h_CHARSET, 37);

    // Allocate device memory
    char *d_target;
    cudaMalloc(&d_target, target_len + 1);
    cudaMemcpy(d_target, target, target_len + 1, cudaMemcpyHostToDevice);

    int *d_found;
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));

    char *d_result;
    cudaMalloc(&d_result, 64);

    int h_found = 0;
    char h_result[64];

    // Iterate through lengths
    for (int len = 1; len <= max_len; ++len) {
        // Optimization: Skip if length doesn't match target
        if (len != target_len) continue;

        unsigned long long total_combinations = 1;
        for (int i = 0; i < len; ++i) total_combinations *= 36;

        printf("Checking length %d: %llu combinations...\n", len, total_combinations);

        int blockSize = 256;
        int gridSize = (total_combinations + blockSize - 1) / blockSize;

        check_password_kernel<<<gridSize, blockSize>>>(d_target, target_len, len, total_combinations, d_found, d_result);
        
        cudaDeviceSynchronize();

        // Check if found
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found) {
            cudaMemcpy(h_result, d_result, 64, cudaMemcpyDeviceToHost);
            printf("FOUND: \"%s\"\n", h_result);
            break;
        }
    }

    if (!h_found) {
        printf("Not found.\n");
    }

    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_result);

    return 0;
}
