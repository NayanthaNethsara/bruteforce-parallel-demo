# SHA256 Brute Force: From Serial to Massive Parallelism

This document details the evolution of a SHA256 brute force password cracker from a single-threaded serial implementation to high-performance parallel versions using OpenMP, MPI, and CUDA.

## 1. Parallelization Strategies

### Serial (Baseline)
The serial implementation performs a depth-first search (or iterative increment) of the candidate space. It computes the SHA256 hash for each candidate using OpenSSL and compares it to the target. This serves as the baseline for correctness and performance.

### OpenMP (Shared Memory)
**Strategy:** Thread-level parallelism on a single node.
- **Approach:** The search space is partitioned based on the first character of the password. Each thread is assigned a subset of starting characters (cyclic distribution).
- **Key Design:** Each thread maintains its own OpenSSL context (`EVP_MD_CTX`) to ensure thread safety without locks. A `critical` section is used only when a match is found to update the shared `found` flag.
- **Justification:** This approach minimizes synchronization overhead. Since SHA256 hashing is compute-heavy and uniform, dynamic scheduling is not strictly necessary, but cyclic static scheduling handles potential minor imbalances well.

### MPI (Distributed Memory)
**Strategy:** Process-level parallelism across multiple nodes.
- **Approach:** Similar to OpenMP, the search space is partitioned by the first character. However, each MPI rank is a completely independent process with its own memory space.
- **Key Design:** There is **zero communication** between processes during the search phase. This "embarrassingly parallel" design eliminates network latency bottlenecks.
- **Termination:** `MPI_Abort` is used to immediately stop all nodes when the password is found, prioritizing speed over graceful shutdown.

### CUDA (Massive Parallelism)
**Strategy:** Data parallelism on GPU.
- **Approach:** Instead of iterating, we map a linear index (0 to $N$) to a specific candidate password. Thousands of GPU threads run in parallel, each checking one specific index.
- **Key Design:**
  - **Custom SHA256:** OpenSSL is not available on GPU, so a custom, lightweight SHA256 implementation was written for the device.
  - **Constant Memory:** Look-up tables (SHA256 constants, charset) are stored in `__constant__` memory to maximize cache hits.
  - **Index-to-String:** On-the-fly generation of candidates from thread indices avoids memory lookups and allows random access to the search space.

## 2. Runtime Configurations

### Hardware Specifications
- **Development:** macOS (Apple Silicon) - Used for code structure and OpenMP verification.
- **Target Testing:** Linux VM/Cluster with:
  - **CPU:** Multi-core x86_64 (for OpenMP/MPI).
  - **GPU:** NVIDIA Tesla/RTX series (Compute Capability 5.0+) for CUDA.

### Software Environment
- **Compilers:** `gcc` (Serial/OpenMP), `mpicc` (MPI), `nvcc` (CUDA).
- **Libraries:**
  - **OpenSSL (`libcrypto`):** Used for Serial, OpenMP, and MPI versions.
  - **CUDA Toolkit:** Used for the GPU version.
- **Versions:** OpenMP 4.5+, MPI 3.0+, CUDA 11.0+.

### Configuration Parameters
- **Charset:** Lowercase alphanumeric (36 characters).
- **Max Length:** Configurable (typically tested up to 5-6 chars).
- **Threads/Processes:**
  - OpenMP: Controlled via `OMP_NUM_THREADS`.
  - MPI: Controlled via `mpirun -np <N>`.
  - CUDA: Block size fixed at 256; Grid size dynamic.

## 3. Performance Analysis

### Speedup & Efficiency
- **OpenMP:** Linear speedup observed up to the core count. Overhead is minimal (thread creation only).
- **MPI:** Linear speedup scaling with the number of nodes. Efficiency is near 100% due to the lack of inter-process communication.
- **CUDA:** Massive speedup (potentially 100x+ vs Serial). The GPU's ability to run thousands of threads hides the latency of individual instructions.

### Bottlenecks
- **Serial/OpenMP/MPI:** The primary bottleneck is the CPU's SHA256 throughput. Library overhead (allocating/freeing OpenSSL contexts) can be significant if not optimized (e.g., reused).
- **CUDA:** The bottleneck is instruction throughput (integer arithmetic) and register pressure. The SHA256 algorithm uses many registers, limiting the number of active warps.

### Scalability Limitations
- **Partitioning Granularity:** All implementations currently partition by the *first character* (36 branches). This limits effective scaling to ~36 threads/ranks. To scale to thousands of cores, a deeper partitioning strategy (e.g., first 2 characters -> 1296 branches) is required.

## 4. Critical Reflection

### Challenges
- **Library Constraints:** The biggest hurdle was that OpenSSL cannot run on the GPU. Implementing a cryptographic algorithm from scratch (handling endianness, padding, and bitwise ops) in CUDA was complex but necessary.
- **Thread Safety:** Debugging race conditions in OpenMP when using external libraries (OpenSSL) required careful attention to thread-local storage.

### Lessons Learned
- **Compute vs. Memory:** On the GPU, recomputing data (generating the string from an index) is often faster than loading it from memory.
- **Overhead Matters:** In high-frequency loops (millions of iterations), even a small allocation (like `EVP_MD_CTX_new`) is a performance killer.
- **Paradigm Shift:** Moving from "iteration" (CPU) to "mapping" (GPU) requires a fundamental shift in thinking about the problem structure.
