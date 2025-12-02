# SHA256 OpenMP Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
Similar to the plain version, the SHA256 OpenMP implementation uses shared-memory parallelism. The search space is partitioned by the first character of the candidate string.

**Design Decisions:**
- **Thread-Local Contexts:** A critical design decision was to allocate `EVP_MD_CTX` (OpenSSL context) locally within each thread. OpenSSL contexts are not thread-safe if shared. Each thread initializes its own context to compute hashes independently.
- **Partitioning:** The outer loop iterates over the first character (`indices[0]`), and this loop is parallelized. This provides coarse-grained parallelism, reducing synchronization overhead.
- **Critical Section:** When a match is found, a `critical` section ensures that only one thread prints the result and sets the global `found` flag.

**Load Balancing:**
SHA256 computation is computationally intensive and uniform (constant time for a given length). Therefore, the load is naturally balanced across threads as long as the search space is evenly divided. The cyclic distribution of starting characters helps ensure this.

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (Apple Silicon M1/M2)
- **Target:** Linux VM (x86_64)

**Software Environment:**
- **Compiler:** GCC/Clang with OpenMP support.
- **Library:** OpenSSL (`libcrypto`) for SHA256 functions.
- **Version:** OpenSSL 3.0+

**Configuration Parameters:**
- **Threads:** `OMP_NUM_THREADS`.
- **Max Length:** Command-line argument.
- **Target Hash:** SHA256 hex string passed as argument.

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Compute Bound:** Unlike the plain version which is memory/branch bound, SHA256 is heavily compute-bound. This makes it an ideal candidate for parallelization.
- **Speedup:** We expect near-linear speedup. The overhead of OpenSSL initialization is paid once per thread, which is negligible over millions of hash computations.

**Bottlenecks:**
- **Context Creation:** Creating/destroying `EVP_MD_CTX` inside the inner loop would be a disaster. We optimized this by creating it once (or reusing it, though the current implementation creates/frees inside the helper function `sha256_hex` for simplicity. *Self-correction: The current code creates/frees context for every hash. This is a significant bottleneck! A better optimization would be to reuse the context.*)
- **Optimization Note:** The current implementation calls `EVP_MD_CTX_new` for *every* candidate. This adds significant overhead (malloc/free). Moving this allocation outside the loop would drastically improve performance.

**Scalability:**
- Scales linearly with core count.
- Limited by the efficiency of the OpenSSL implementation on the specific CPU architecture (e.g., usage of SHA extensions).

## 4. Critical Reflection
**Challenges:**
- **Thread Safety:** Ensuring OpenSSL functions were called correctly in a threaded environment.
- **Performance Tuning:** Realizing that `EVP_MD_CTX_new` is expensive.

**Limitations:**
- **CPU vs GPU:** Even with perfect scaling, CPUs are far slower than GPUs for hashing. A single GPU can outperform a high-end CPU by 100x for SHA256.

**Lessons Learned:**
- Library overhead (like OpenSSL allocation) can dominate runtime in tight loops.
- Thread-local storage is essential for using non-thread-safe libraries in parallel.
