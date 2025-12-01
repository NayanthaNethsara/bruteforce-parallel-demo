#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

// SHA-256 Constants and Macros
#define ROTRIGHT(word,bits) (((word) >> (bits)) | ((word) << (32-(bits))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant__ uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__constant__ char d_CHARSET[37]; 
const char *h_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"; 

// Helper to convert hex string to byte array
void hex_to_bytes(const char *hex, unsigned char *bytes) {
    for (int i = 0; i < 32; ++i) {
        sscanf(hex + 2*i, "%02hhx", &bytes[i]);
    }
}

// Returns current time in seconds
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__device__ void sha256_transform(uint32_t state[8], const unsigned char data[], uint64_t len) {
    uint32_t m[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    // Initialize state
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;

    // Prepare message block (simplified for short messages < 55 bytes)
    // We assume data fits in one 512-bit block for typical passwords
    // Copy data to local buffer
    unsigned char block[64];
    for(int i=0; i<64; ++i) block[i] = 0;
    for(int i=0; i<len; ++i) block[i] = data[i];
    
    // Padding
    block[len] = 0x80;
    // Length in bits at the end (big endian)
    uint64_t bitlen = len * 8;
    block[63] = bitlen & 0xFF;
    block[62] = (bitlen >> 8) & 0xFF;
    block[61] = (bitlen >> 16) & 0xFF;
    block[60] = (bitlen >> 24) & 0xFF;
    // Assuming len < 55 bytes, high 32 bits of length are 0

    // Decode block into 32-bit words (big endian)
    for (int i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (block[j] << 24) | (block[j + 1] << 16) | (block[j + 2] << 8) | (block[j + 3]);

    for (int i = 16; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (int i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__global__ void check_sha256_kernel(unsigned char *d_target_hash, int len, unsigned long long total_combinations, int *d_found, char *d_result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_combinations) return;
    if (*d_found) return;

    char candidate[16]; 
    unsigned long long temp = idx;
    for (int i = len - 1; i >= 0; --i) {
        candidate[i] = d_CHARSET[temp % 36];
        temp /= 36;
    }
    
    uint32_t state[8];
    sha256_transform(state, (unsigned char*)candidate, len);

    // Compare with target hash (which is in bytes)
    // state is 8x 32-bit integers. Target is 32 bytes.
    // We need to check byte by byte or word by word.
    // d_target_hash is byte array.
    
    bool match = true;
    for(int i=0; i<8; ++i) {
        // Extract bytes from state[i] (big endian)
        unsigned char b3 = (state[i] >> 24) & 0xFF;
        unsigned char b2 = (state[i] >> 16) & 0xFF;
        unsigned char b1 = (state[i] >> 8) & 0xFF;
        unsigned char b0 = (state[i]) & 0xFF;
        
        if (d_target_hash[i*4 + 0] != b3) { match = false; break; }
        if (d_target_hash[i*4 + 1] != b2) { match = false; break; }
        if (d_target_hash[i*4 + 2] != b1) { match = false; break; }
        if (d_target_hash[i*4 + 3] != b0) { match = false; break; }
    }

    if (match) {
        *d_found = 1;
        for (int i = 0; i < len; ++i) d_result[i] = candidate[i];
        d_result[len] = '\0';
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_sha256_hex> <max_len>\n", argv[0]);
        return 1;
    }
    const char *target_hex = argv[1];
    int max_len = atoi(argv[2]);
    
    if (strlen(target_hex) != 64) {
        fprintf(stderr, "Error: target SHA-256 must be 64 hex chars.\n");
        return 1;
    }

    printf("SHA256 CUDA Brute Force: Target=\"%s\", max_len=%d\n", target_hex, max_len);

    unsigned char target_bytes[32];
    hex_to_bytes(target_hex, target_bytes);

    cudaMemcpyToSymbol(d_CHARSET, h_CHARSET, 37);

    unsigned char *d_target_hash;
    cudaMalloc(&d_target_hash, 32);
    cudaMemcpy(d_target_hash, target_bytes, 32, cudaMemcpyHostToDevice);

    int *d_found;
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));

    char *d_result;
    cudaMalloc(&d_result, 64);

    int h_found = 0;
    char h_result[64];

    double t0 = get_time_sec();

    for (int len = 1; len <= max_len; ++len) {
        unsigned long long total_combinations = 1;
        for (int i = 0; i < len; ++i) total_combinations *= 36;

        printf("Checking length %d: %llu combinations...\n", len, total_combinations);

        int blockSize = 256;
        int gridSize = (total_combinations + blockSize - 1) / blockSize;

        check_sha256_kernel<<<gridSize, blockSize>>>(d_target_hash, len, total_combinations, d_found, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found) {
            cudaMemcpy(h_result, d_result, 64, cudaMemcpyDeviceToHost);
            double elapsed = get_time_sec() - t0;
            printf("FOUND: \"%s\" in %.4f s\n", h_result, elapsed);
            break;
        }
    }

    if (!h_found) {
        double elapsed = get_time_sec() - t0;
        printf("Not found. Time: %.4f s\n", elapsed);
    }

    cudaFree(d_target_hash);
    cudaFree(d_found);
    cudaFree(d_result);

    return 0;
}
