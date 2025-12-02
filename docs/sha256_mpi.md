# SHA256 MPI Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
The MPI implementation partitions the SHA256 brute force task across distributed processes. Each process is assigned a subset of the search space based on the starting character of the candidate password.

**Design Decisions:**
- **Distributed Hashing:** Each rank runs an independent instance of the brute force engine. It initializes its own OpenSSL context and computes hashes for its assigned candidates.
- **No Communication:** Similar to the plain MPI version, there is no communication between ranks during the search. This "embarrassingly parallel" design is optimal for brute force as long as the solution is not found.
- **Result Broadcasting:** When a rank finds the password, it prints the result and aborts the entire MPI job. This avoids the need for complex termination signals.

**Load Balancing:**
The cyclic distribution of starting characters (`rank + k * size`) ensures that the workload is evenly distributed across all nodes. Since SHA256 hashing takes constant time for a fixed input length, the load is perfectly balanced in terms of computation.

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (Apple Silicon M1/M2)
- **Target:** Linux Cluster (MPI-enabled)

**Software Environment:**
- **Compiler:** `mpicc`
- **Libraries:** OpenMPI/MPICH, OpenSSL (`libcrypto`)
- **Version:** MPI 3.0+, OpenSSL 3.0+

**Configuration Parameters:**
- **Processes:** `mpirun -np <N>`
- **Max Length:** Command-line argument.
- **Target Hash:** SHA256 hex string.

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Linear Scaling:** We observe linear speedup with the number of processes. If one core checks X hashes/sec, N cores check N*X hashes/sec.
- **Efficiency:** Extremely high. The only non-computation time is process startup.

**Bottlenecks:**
- **Context Allocation:** Similar to the OpenMP version, if `EVP_MD_CTX_new` is called inside the loop, it becomes a major bottleneck. (Note: The current implementation has this inefficiency).
- **Network:** None, as there is no communication.

**Scalability:**
- Theoretically scales to thousands of cores.
- Practically limited by the number of available starting characters (36) with the current partitioning. For larger scales, we would need to partition based on the first 2 or 3 characters.

## 4. Critical Reflection
**Challenges:**
- **Dependency Management:** Ensuring OpenSSL is available and linked correctly on all nodes in a cluster can be a deployment challenge.
- **Optimization:** The realization that library calls (OpenSSL) have overhead compared to a raw C implementation of SHA256.

**Limitations:**
- **Granularity:** The current 1-char partitioning limits us to 36 effective ranks.
- **Abrupt Termination:** `MPI_Abort` kills everything, which is fine for a demo but bad for a larger application sharing resources.

**Lessons Learned:**
- For compute-bound tasks like hashing, minimizing memory allocations (like context creation) inside the hot loop is the single most important optimization.
