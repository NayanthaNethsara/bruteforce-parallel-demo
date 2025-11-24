# bruteforce-parallel-demo

Educational brute-force password cracking **simulation** comparing serial, OpenMP, MPI, and CUDA implementations.  
This repository demonstrates parallel programming techniques for performance comparison. **Do not use on real/unauthorized accounts.** Use only test passwords/hashes you create.

---

## Repository Structure

```
bruteforce-parallel-demo/
├── README.md
├── LICENSE
├── Makefile               # Unified build system
├── plain/                 # Plaintext brute-force implementations
│   ├── serial/
│   │   └── brute_plain.c
│   ├── openmp/
│   │   └── brute_plain_openmp.c
│   ├── mpi/
│   │   └── brute_plain_mpi.c
│   └── cuda/
│       └── brute_plain_cuda.cu
├── sha256/                # SHA-256 brute-force implementations
│   ├── serial/
│   │   └── brute_force_serial.c
│   ├── openmp/
│   │   └── brute_sha256_openmp.c
│   ├── mpi/
│   │   └── brute_sha256_mpi.c
│   └── cuda/
│       └── brute_sha256_cuda.cu
└── bin/                   # Compiled binaries (created by make)
```

---

## Quick Summary

- **plain/**: Plaintext password matching (no hashing, faster for testing parallelization overhead)
- **sha256/**: SHA-256 hash-based brute-force (requires OpenSSL, more realistic workload)

Each category has four implementations:

- **serial**: Single-threaded baseline
- **openmp**: Shared-memory parallelism using OpenMP
- **mpi**: Distributed-memory parallelism using MPI
- **cuda**: GPU parallelism using NVIDIA CUDA

---

## Requirements

### Core

- C compiler (GCC, Clang, or compatible)
- Make
- OpenSSL dev headers (for SHA-256 implementations)
  - **macOS**: `brew install openssl@3`
  - **Ubuntu/Debian**: `sudo apt-get install build-essential libssl-dev`

### Optional (for parallel implementations)

- **OpenMP**: Usually bundled with GCC (use `-fopenmp` flag)
- **MPI**: OpenMPI or MPICH
  - **macOS**: `brew install open-mpi`
  - **Ubuntu/Debian**: `sudo apt-get install libopenmpi-dev`
- **CUDA**: NVIDIA CUDA Toolkit (GPU required)
  - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

---

## Build

The Makefile automatically detects available toolchains and builds accordingly.

### Check your system

```bash
make info
```

This shows:

- Detected OS and compilers
- Which parallel implementations can be built
- OpenSSL configuration

### Build everything available

```bash
make
```

This builds:

- Serial implementations (always)
- OpenMP implementations (if compiler supports it)
- MPI implementations (if `mpicc` is found)
- CUDA implementations (if `nvcc` is found)

### Build specific implementations

```bash
make serial   # Serial only
make openmp   # OpenMP only
make mpi      # MPI only
make cuda     # CUDA only
```

Binaries are output to `bin/`:

- `plain_serial`, `plain_openmp`, `plain_mpi`, `plain_cuda`
- `sha256_serial`, `sha256_openmp`, `sha256_mpi`, `sha256_cuda`

---

## Usage

The Makefile provides run targets with sensible defaults for quick testing.

### Plaintext Brute-Force

**Serial (single-threaded):**

```bash
make run-plain-serial
# Or with custom args:
make run-plain-serial ARGS='abc 4'
```

**OpenMP (multi-threaded):**

```bash
make run-plain-openmp
# Set thread count:
OMP_NUM_THREADS=8 make run-plain-openmp ARGS='abc 4'
```

**MPI (distributed):**

```bash
make run-plain-mpi
# Specify process count:
make run-plain-mpi NP=8 ARGS='abc 4'
```

**CUDA (GPU):**

```bash
make run-plain-cuda
make run-plain-cuda ARGS='abc 4'
```

### SHA-256 Brute-Force

**Serial:**

```bash
make run-sha256-serial
# Default uses SHA-256 of "a" with max_len=1
```

**Generate a test hash:**

```bash
echo -n "a1" | openssl dgst -sha256
# Copy the hex digest and use it:
make run-sha256-serial ARGS='<hex_digest> 2'
```

**OpenMP:**

```bash
OMP_NUM_THREADS=8 make run-sha256-openmp ARGS='<hash> <max_len>'
```

**MPI:**

```bash
make run-sha256-mpi NP=8 ARGS='<hash> <max_len>'
```

**CUDA:**

```bash
make run-sha256-cuda ARGS='<hash> <max_len>'
```

### Direct Binary Execution

You can also run binaries directly:

```bash
bin/plain_serial <target_password> <max_len>
bin/sha256_serial <sha256_hex> <max_len>

# MPI requires mpirun:
mpirun -np 4 bin/plain_mpi <target_password> <max_len>

# OpenMP respects OMP_NUM_THREADS:
OMP_NUM_THREADS=8 bin/plain_openmp <target_password> <max_len>
```

---

## Performance Comparison

Use the different implementations to compare:

- **Serial** vs **OpenMP**: Shared-memory speedup
- **OpenMP** vs **MPI**: Local parallelism vs distributed
- **CPU** vs **CUDA**: CPU multi-core vs GPU massive parallelism

For meaningful benchmarks, use longer passwords (e.g., `max_len=4-6`) and measure execution time.

---

## Clean Build

```bash
make clean
```

Removes all binaries from `bin/`.

---

## Ethical & Safety Note

This repository is strictly for educational purposes. Use only test passwords/hashes you own. Do not target real or unauthorized accounts or systems. Include this statement when you demo or publish the repo.

---

## License

Licensed under the MIT License. See `LICENSE` for the full text.

While the license is permissive, please act responsibly and ethically—use only with test data and in permitted environments.
