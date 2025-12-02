# SHA256 CUDA Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
The CUDA implementation for SHA256 brute force is a highly parallelized, compute-intensive solution. It assigns a unique index to each thread, which generates a candidate password and computes its SHA256 hash entirely on the GPU.

**Design Decisions:**
- **Custom SHA256 Implementation:** Since standard libraries like OpenSSL are not available in CUDA device code, we implemented the SHA256 algorithm (transform function) from scratch using bitwise operations (`ROTRIGHT`, `CH`, `MAJ`, `SIG0`, `SIG1`). This allows the entire hashing process to occur in parallel on thousands of CUDA cores.
- **Index-to-Candidate:** Similar to the plain CUDA version, threads convert their linear index to a base-36 string. This avoids memory lookups for candidate generation.
- **Constant Memory:** The SHA256 constants (`k` array) and the charset are stored in `__constant__` memory. This is critical for performance because these values are read-only and accessed by all threads simultaneously, allowing the GPU to broadcast them from the constant cache.

**Load Balancing:**
The workload is perfectly uniform. Every thread performs exactly one SHA256 transform block (for short passwords) and one comparison. There is zero divergence until a match is found.

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (No CUDA)
- **Target:** NVIDIA GPU (Compute Capability 5.0+)

**Software Environment:**
- **Compiler:** `nvcc`
- **Toolkit:** CUDA Toolkit 11.0+

**Configuration Parameters:**
- **Block Size:** 256 threads.
- **Grid Size:** Dynamic based on search space size.
- **Max Length:** Command-line argument.

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Throughput:** GPUs are specialized for the kind of bitwise arithmetic used in SHA256 (32-bit integer operations). A mid-range GPU can compute billions of hashes per second.
- **Comparison:** We expect the CUDA version to be orders of magnitude faster than the OpenMP or MPI CPU versions (e.g., 100x speedup).

**Bottlenecks:**
- **Register Usage:** The SHA256 algorithm requires a fair amount of state (8 working variables + message schedule array). This high register pressure can limit the number of active warps per SM (occupancy).
- **Global Memory Access:** Reading the target hash from global memory is fast (cached), but writing the result is rare. The main bottleneck is pure compute.

**Scalability:**
- Scales perfectly with the number of Streaming Multiprocessors (SMs) on the GPU.
- Can be extended to multi-GPU using MPI+CUDA or NCCL.

## 4. Critical Reflection
**Challenges:**
- **Implementation Complexity:** Porting SHA256 to CUDA was the most complex part of this project. It required understanding the algorithm deeply and mapping it to GPU constraints (no dynamic allocation, no standard lib).
- **Endianness:** Handling byte ordering (Big Endian for SHA256 vs Little Endian for x86/CUDA) required careful bit manipulation.

**Limitations:**
- **Password Length:** The current implementation assumes the password fits in a single SHA256 block (55 bytes). For longer passwords, a loop would be needed in the kernel, increasing complexity and register usage.

**Lessons Learned:**
- **Custom Kernels:** Sometimes you have to reinvent the wheel (implement SHA256) to unlock the power of the hardware.
- **Constant Cache:** Using `__constant__` for lookup tables is a massive optimization for cryptographic algorithms on GPU.
