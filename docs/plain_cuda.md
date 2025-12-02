# Plain CUDA Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
The CUDA implementation utilizes a massive data-parallel approach. Instead of iterating through candidates, each thread is assigned a unique index in the linear search space (from 0 to $36^N - 1$). The thread converts this index into a candidate string (base-36 conversion) and checks it against the target.

**Design Decisions:**
- **Index-to-String Conversion:** This is the core of the parallelization. It allows random access to any point in the search space without dependency on previous states. This eliminates the need for communication between threads.
- **Kernel Launch:** We launch a kernel with enough blocks and threads to cover the entire search space for a given length. For length 5, there are $36^5 \approx 60$ million combinations, which fits easily into a modern GPU's grid.
- **Batch Processing:** To handle large search spaces (e.g., length 9 with $>10^{13}$ combinations) that exceed the maximum grid size ($2^{31}-1$ blocks) or cause TDR (Timeout Detection and Recovery) issues, we implemented a batching mechanism. The host code iterates through the search space in chunks (e.g., 100 million at a time), launching the kernel repeatedly with an `offset` parameter.
- **Memory Management:** The target password and result buffer are stored in global memory. The charset is stored in constant memory (`__constant__`) for fast cached access, as every thread reads from it.

**Load Balancing:**
The load is perfectly balanced because every thread performs exactly the same amount of work: one index-to-string conversion and one string comparison. There is no control flow divergence (except for the one thread that finds the match).

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (No CUDA hardware)
- **Target:** NVIDIA GPU (e.g., Tesla T4, V100, or RTX series) with Compute Capability 5.0+

**Software Environment:**
- **Compiler:** `nvcc` (NVIDIA CUDA Compiler)
- **Toolkit:** CUDA Toolkit 11.0+
- **Drivers:** Compatible NVIDIA Drivers

**Configuration Parameters:**
- **Block Size:** 256 threads per block (standard occupancy optimization).
- **Grid Size:** Calculated dynamically: `(total_combinations + blockSize - 1) / blockSize`.
- **Max Length:** Command-line argument.

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Speedup:** Massive speedup compared to CPU. A GPU can launch tens of thousands of threads simultaneously. For compute-bound tasks like this (simple arithmetic and memory reads), GPUs are orders of magnitude faster.
- **Throughput:** We expect to check hundreds of millions of passwords per second.

**Bottlenecks:**
- **Integer Division:** The index-to-string conversion uses modulo (`%`) and division (`/`) operations. These are relatively expensive on GPUs compared to bitwise operations.
- **Global Memory Atomic:** When a thread finds the password, it writes to a shared flag. This is a rare event, so it's not a bottleneck.

**Scalability:**
- Scales with the number of CUDA cores.
- Limited by the maximum grid size (which is very large, $2^{31}-1$ blocks), so it can handle very large search spaces.

## 4. Critical Reflection
**Challenges:**
- **Divergence:** While minimal here, ensuring threads don't diverge is key.
- **String Handling:** C-style strings are clumsy in CUDA. We used fixed-size arrays in registers/local memory.

**Limitations:**
- **Register Pressure:** If the password length is very long, the local `candidate` array increases register usage, potentially reducing occupancy.
- **Hardware:** Strictly requires NVIDIA hardware.

**Lessons Learned:**
- "Compute is cheap, memory is expensive." Recomputing the string from an index is better than trying to load it from memory.
- Constant memory is excellent for lookup tables like the charset.
