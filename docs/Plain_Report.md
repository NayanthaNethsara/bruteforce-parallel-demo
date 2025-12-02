# Plaintext Brute Force: From Serial to Massive Parallelism

This document details the evolution of a plaintext brute force password cracker from a single-threaded serial implementation to high-performance parallel versions using OpenMP, MPI, and CUDA.

## 1. Parallelization Strategies

### Serial (Baseline)
The serial implementation uses a recursive or iterative approach to generate every possible string combination from the character set. For each candidate, it performs a simple string comparison (`strcmp`) against the target. This provides the baseline for measuring speedup.

### OpenMP (Shared Memory)
**Strategy:** Thread-level parallelism on a single node.
- **Approach:** The search space is partitioned based on the first character of the password.
- **Key Design:**
  - **Work Sharing:** Threads are assigned specific starting characters (e.g., Thread 0 takes 'a', Thread 1 takes 'b').
  - **Memory:** Each thread has its own private candidate buffer to avoid race conditions during string generation.
  - **Synchronization:** Minimal synchronization is needed. A volatile `found` flag allows threads to exit early once the password is discovered.
- **Justification:** This coarse-grained parallelism is simple to implement and effective for uniform search spaces.

### MPI (Distributed Memory)
**Strategy:** Process-level parallelism across multiple nodes.
- **Approach:** Similar to OpenMP, the search space is partitioned by the first character.
- **Key Design:**
  - **Independence:** Each MPI rank runs a completely independent search loop. There is no communication during the computation phase.
  - **Termination:** `MPI_Abort` is used to broadcast the "found" signal instantly to all nodes.
- **Justification:** The "share-nothing" architecture eliminates communication overhead, allowing linear scaling across clusters.

### CUDA (Massive Parallelism)
**Strategy:** Data parallelism on GPU.
- **Approach:** We map a linear index (0 to $N$) to a unique candidate password string.
- **Key Design:**
  - **Index-to-String:** A kernel function converts the global thread ID into a base-36 string on the fly. This allows random access to the search space without dependency on previous states.
  - **Batch Processing:** To handle massive search spaces without overflowing GPU grid limits, the kernel is launched in batches (e.g., 100M at a time) with an offset.
  - **Memory:** The target string and result buffer are stored in global memory, while the charset is in `__constant__` memory for fast broadcast access.
- **Justification:** GPUs excel at massive throughput. By launching millions of threads, we can check millions of passwords in parallel, far exceeding CPU capabilities.

## 2. Runtime Configurations

### Hardware Specifications
- **Development:** macOS (Apple Silicon) - Used for code structure and OpenMP verification.
- **Target Testing:** Linux VM/Cluster with:
  - **CPU:** Multi-core x86_64 (for OpenMP/MPI).
  - **GPU:** NVIDIA Tesla/RTX series (Compute Capability 5.0+) for CUDA.

### Software Environment
- **Compilers:** `gcc` (Serial/OpenMP), `mpicc` (MPI), `nvcc` (CUDA).
- **Libraries:** Standard C libraries (`libc`), CUDA Toolkit.
- **Versions:** OpenMP 4.5+, MPI 3.0+, CUDA 11.0+.

### Configuration Parameters
- **Charset:** Lowercase alphanumeric (36 characters).
- **Max Length:** Configurable (typically tested up to 5-6 chars).
- **Threads/Processes:**
  - OpenMP: `OMP_NUM_THREADS`.
  - MPI: `mpirun -np <N>`.
  - CUDA: Block size 256; Grid size dynamic.

## 3. Performance Analysis

### Speedup & Efficiency
- **OpenMP:** Linear speedup on multi-core CPUs. The overhead is negligible.
- **MPI:** Linear speedup scaling with the number of nodes. Efficiency is near 100% due to zero communication.
- **CUDA:** Massive speedup. For simple string comparison, the GPU's sheer number of cores allows it to check orders of magnitude more passwords per second than a CPU.

### Bottlenecks
- **Serial/OpenMP/MPI:** The bottleneck is the CPU clock speed and branch prediction. String generation involves many small updates.
- **CUDA:** The bottleneck is integer arithmetic (modulo/division) for the index-to-string conversion. String comparison is very fast.

### Scalability Limitations
- **Partitioning:** Partitioning by the first character limits us to 36 concurrent tasks (threads/ranks). This is a hard limit for the current implementation's scalability.
- **Search Space:** For very short passwords (< 4 chars), the overhead of launching kernels or processes might outweigh the search time.

## 4. Critical Reflection

### Challenges
- **String Handling on GPU:** C-style strings are not native to GPUs. Managing char arrays in registers/local memory without dynamic allocation was a key challenge.
- **Termination:** Stopping a distributed (MPI) or massive parallel (CUDA) search instantly when the answer is found is tricky. We opted for "abort" mechanisms for speed, but this is not always clean.

### Lessons Learned
- **Algorithm Adaptation:** The recursive/iterative approach used on CPU does not map well to GPU. Changing to a mathematical mapping (Index -> String) was essential for CUDA.
- **Simplicity Wins:** For brute force, the simplest partitioning (by start char) often yields the best performance because it avoids complex load balancing overhead.
