# Makefile for bruteforce-parallel-demo
# Supports: Serial, OpenMP, MPI, CUDA implementations
# Platforms: macOS, Linux

# === Toolchain Configuration ===
CC ?= cc
MPICC ?= mpicc
NVCC ?= nvcc

# Detect OS
UNAME_S := $(shell uname -s)

# === Compiler Flags ===
CFLAGS := -O2 -Wall -Wextra -std=c11
OPENMP_FLAGS := -fopenmp
LDFLAGS :=

# Add POSIX flag for clock_gettime on Linux
ifneq ($(UNAME_S),Darwin)
CFLAGS += -D_POSIX_C_SOURCE=199309L
endif

# OpenSSL detection (for SHA-256 implementations)
OPENSSL_CFLAGS := $(shell pkg-config --cflags openssl 2>/dev/null)
OPENSSL_LIBS   := $(shell pkg-config --libs openssl 2>/dev/null)

# Fallback to Homebrew on macOS if pkg-config fails
ifeq ($(strip $(OPENSSL_LIBS)),)
ifeq ($(UNAME_S),Darwin)
BREW_OPENSSL := $(shell brew --prefix openssl@3 2>/dev/null)
ifneq ($(strip $(BREW_OPENSSL)),)
OPENSSL_CFLAGS := -I$(BREW_OPENSSL)/include
OPENSSL_LIBS   := -L$(BREW_OPENSSL)/lib -lcrypto
else
OPENSSL_LIBS := -lcrypto
endif
else
OPENSSL_LIBS := -lcrypto
endif
endif

# CUDA flags
NVCC_FLAGS := -O2

# === Directories ===
BIN_DIR := bin

# === Check for optional toolchains ===
HAS_OPENMP := $(shell echo "int main(){return 0;}" | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo yes || echo no)
HAS_MPI := $(shell command -v $(MPICC) >/dev/null 2>&1 && echo yes || echo no)
HAS_CUDA := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

# === Build Targets ===
PLAIN_SERIAL := $(BIN_DIR)/plain_serial
PLAIN_OPENMP := $(BIN_DIR)/plain_openmp
PLAIN_MPI := $(BIN_DIR)/plain_mpi
PLAIN_CUDA := $(BIN_DIR)/plain_cuda

SHA256_SERIAL := $(BIN_DIR)/sha256_serial
SHA256_OPENMP := $(BIN_DIR)/sha256_openmp
SHA256_MPI := $(BIN_DIR)/sha256_mpi
SHA256_CUDA := $(BIN_DIR)/sha256_cuda

# Default targets (only serial builds)
DEFAULT_TARGETS := $(PLAIN_SERIAL) $(SHA256_SERIAL)

# Optional targets
OPENMP_TARGETS :=
MPI_TARGETS :=
CUDA_TARGETS :=

ifeq ($(HAS_OPENMP),yes)
OPENMP_TARGETS := $(PLAIN_OPENMP) $(SHA256_OPENMP)
endif

ifeq ($(HAS_MPI),yes)
MPI_TARGETS := $(PLAIN_MPI) $(SHA256_MPI)
endif

ifeq ($(HAS_CUDA),yes)
CUDA_TARGETS := $(PLAIN_CUDA) $(SHA256_CUDA)
endif

# === Phony Targets ===
.PHONY: all serial openmp mpi cuda clean info
.DEFAULT_GOAL := all

all: serial $(OPENMP_TARGETS) $(MPI_TARGETS) $(CUDA_TARGETS)
	@echo ""
	@echo "=== Build Summary ==="
	@echo "Serial:  ✓ (always built)"
	@if [ "$(HAS_OPENMP)" = "yes" ]; then echo "OpenMP:  ✓"; else echo "OpenMP:  ✗ (compiler missing -fopenmp support)"; fi
	@if [ "$(HAS_MPI)" = "yes" ]; then echo "MPI:     ✓"; else echo "MPI:     ✗ (mpicc not found)"; fi
	@if [ "$(HAS_CUDA)" = "yes" ]; then echo "CUDA:    ✓"; else echo "CUDA:    ✗ (nvcc not found)"; fi
	@echo ""

serial: $(DEFAULT_TARGETS)

openmp: $(OPENMP_TARGETS)

mpi: $(MPI_TARGETS)

cuda: $(CUDA_TARGETS)

info:
	@echo "=== Toolchain Detection ==="
	@echo "OS:          $(UNAME_S)"
	@echo "CC:          $(CC)"
	@echo "MPICC:       $(MPICC) (available: $(HAS_MPI))"
	@echo "NVCC:        $(NVCC) (available: $(HAS_CUDA))"
	@echo "OpenMP:      $(HAS_OPENMP)"
	@echo ""
	@echo "OpenSSL:"
	@echo "  CFLAGS:    $(OPENSSL_CFLAGS)"
	@echo "  LIBS:      $(OPENSSL_LIBS)"
	@echo ""
	@echo "=== Available Targets ==="
	@echo "  make          - Build all available implementations"
	@echo "  make serial   - Build serial implementations only"
	@echo "  make openmp   - Build OpenMP implementations"
	@echo "  make mpi      - Build MPI implementations"
	@echo "  make cuda     - Build CUDA implementations"
	@echo "  make clean    - Remove all binaries"
	@echo "  make info     - Show this information"

# === Directory Creation ===
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# === Plain Serial ===
$(PLAIN_SERIAL): plain/serial/brute_plain.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS) -lgomp

# === Plain OpenMP ===
$(PLAIN_OPENMP): plain/openmp/brute_plain_openmp.c | $(BIN_DIR)
ifeq ($(HAS_OPENMP),yes)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS)
else
	@echo "Skipping $@ (OpenMP not available)"
endif

# === Plain MPI ===
$(PLAIN_MPI): plain/mpi/brute_plain_mpi.c | $(BIN_DIR)
ifeq ($(HAS_MPI),yes)
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)
else
	@echo "Skipping $@ (MPI not available)"
endif

# === Plain CUDA ===
$(PLAIN_CUDA): plain/cuda/brute_plain_cuda.cu | $(BIN_DIR)
ifeq ($(HAS_CUDA),yes)
	$(NVCC) $(NVCC_FLAGS) $< -o $@
else
	@echo "Skipping $@ (CUDA not available)"
endif

# === SHA-256 Serial ===
$(SHA256_SERIAL): sha256/serial/brute_force_serial.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENSSL_CFLAGS) $< -o $@ $(OPENSSL_LIBS) $(LDFLAGS)

# === SHA-256 OpenMP ===
$(SHA256_OPENMP): sha256/openmp/brute_sha256_openmp.c | $(BIN_DIR)
ifeq ($(HAS_OPENMP),yes)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(OPENSSL_CFLAGS) $< -o $@ $(OPENSSL_LIBS) $(LDFLAGS)
else
	@echo "Skipping $@ (OpenMP not available)"
endif

# === SHA-256 MPI ===
$(SHA256_MPI): sha256/mpi/brute_sha256_mpi.c | $(BIN_DIR)
ifeq ($(HAS_MPI),yes)
	$(MPICC) $(CFLAGS) $(OPENSSL_CFLAGS) $< -o $@ $(OPENSSL_LIBS) $(LDFLAGS)
else
	@echo "Skipping $@ (MPI not available)"
endif

# === SHA-256 CUDA ===
$(SHA256_CUDA): sha256/cuda/brute_sha256_cuda.cu | $(BIN_DIR)
ifeq ($(HAS_CUDA),yes)
	$(NVCC) $(NVCC_FLAGS) -I$(dir $(OPENSSL_CFLAGS)) $< -o $@ $(OPENSSL_LIBS)
else
	@echo "Skipping $@ (CUDA not available)"
endif

# === Run Helpers (with defaults) ===
.PHONY: run-plain-serial run-plain-openmp run-plain-mpi run-plain-cuda
.PHONY: run-sha256-serial run-sha256-openmp run-sha256-mpi run-sha256-cuda

# Plain defaults
PLAIN_DEFAULT_ARGS ?= a 1

run-plain-serial: $(PLAIN_SERIAL)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(PLAIN_DEFAULT_ARGS)"; \
	echo "Running: $(PLAIN_SERIAL) $$ARGS"; \
	$(PLAIN_SERIAL) $$ARGS

run-plain-openmp: $(PLAIN_OPENMP)
ifeq ($(HAS_OPENMP),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(PLAIN_DEFAULT_ARGS)"; \
	echo "Running: $(PLAIN_OPENMP) $$ARGS"; \
	$(PLAIN_OPENMP) $$ARGS
else
	@echo "OpenMP not available"
endif

run-plain-mpi: $(PLAIN_MPI)
ifeq ($(HAS_MPI),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(PLAIN_DEFAULT_ARGS)"; \
	NP=$${NP:-4}; \
	echo "Running: mpirun -np $$NP $(PLAIN_MPI) $$ARGS"; \
	mpirun -np $$NP $(PLAIN_MPI) $$ARGS
else
	@echo "MPI not available"
endif

run-plain-cuda: $(PLAIN_CUDA)
ifeq ($(HAS_CUDA),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(PLAIN_DEFAULT_ARGS)"; \
	echo "Running: $(PLAIN_CUDA) $$ARGS"; \
	$(PLAIN_CUDA) $$ARGS
else
	@echo "CUDA not available"
endif

# SHA-256 defaults (SHA-256 of "a", max_len=1)
SHA256_DEFAULT_ARGS ?= ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb 1

run-sha256-serial: $(SHA256_SERIAL)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(SHA256_DEFAULT_ARGS)"; \
	echo "Running: $(SHA256_SERIAL) $$ARGS"; \
	$(SHA256_SERIAL) $$ARGS

run-sha256-openmp: $(SHA256_OPENMP)
ifeq ($(HAS_OPENMP),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(SHA256_DEFAULT_ARGS)"; \
	echo "Running: $(SHA256_OPENMP) $$ARGS"; \
	$(SHA256_OPENMP) $$ARGS
else
	@echo "OpenMP not available"
endif

run-sha256-mpi: $(SHA256_MPI)
ifeq ($(HAS_MPI),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(SHA256_DEFAULT_ARGS)"; \
	NP=$${NP:-4}; \
	echo "Running: mpirun -np $$NP $(SHA256_MPI) $$ARGS"; \
	mpirun -np $$NP $(SHA256_MPI) $$ARGS
else
	@echo "MPI not available"
endif

run-sha256-cuda: $(SHA256_CUDA)
ifeq ($(HAS_CUDA),yes)
	@ARGS="$(ARGS)"; [ -z "$$ARGS" ] && ARGS="$(SHA256_DEFAULT_ARGS)"; \
	echo "Running: $(SHA256_CUDA) $$ARGS"; \
	$(SHA256_CUDA) $$ARGS
else
	@echo "CUDA not available"
endif

# === Clean ===
clean:
	@rm -rf $(BIN_DIR)
	@echo "Cleaned build directory"
