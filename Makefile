# Simple Makefile for bruteforce-demo
# Builds two binaries into ./bin and provides run helpers.

# --- Toolchain ---
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?=

# --- OpenSSL detection (for serial target) ---
# Try pkg-config first
OPENSSL_CFLAGS := $(shell pkg-config --cflags openssl 2>/dev/null)
OPENSSL_LIBS   := $(shell pkg-config --libs openssl 2>/dev/null)

# Fallback to Homebrew openssl@3 on macOS if pkg-config not available or empty
ifeq ($(strip $(OPENSSL_LIBS)),)
BREW_OPENSSL_PREFIX := $(shell brew --prefix openssl@3 2>/dev/null)
ifneq ($(strip $(BREW_OPENSSL_PREFIX)),)
OPENSSL_CFLAGS += -I$(BREW_OPENSSL_PREFIX)/include
OPENSSL_LIBS   += -L$(BREW_OPENSSL_PREFIX)/lib -lcrypto
else
# Last resort: rely on system linker path
OPENSSL_LIBS   += -lcrypto
endif
endif

# --- Paths ---
BIN_DIR := bin
PLAIN_SRC := plain/brute_plain.c
SERIAL_SRC := serial/brute_force_serial.c

PLAIN_BIN := $(BIN_DIR)/brute_plain
SERIAL_BIN := $(BIN_DIR)/brute_serial

# Default target builds both
.PHONY: all
all: $(PLAIN_BIN) $(SERIAL_BIN)

# Convenience aliases to build individually
.PHONY: plain serial
plain: $(PLAIN_BIN)
serial: $(SERIAL_BIN)

# Ensure bin dir exists
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build rules
$(PLAIN_BIN): $(PLAIN_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(SERIAL_BIN): $(SERIAL_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENSSL_CFLAGS) $< -o $@ $(OPENSSL_LIBS) $(LDFLAGS)

# Run helpers. Provide arguments via ARGS variable, or defaults will be used.
# Defaults:
#   run-plain  -> ARGS='a 1'
#   run-serial -> ARGS='ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb 1'  (sha256("a"), 1)
# Examples overriding defaults:
#   make run-plain ARGS='abc 4'
#   make run-serial ARGS='5d41402abc4b2a76b9719d911017c592 5'

# Default arguments if ARGS not supplied
RUN_PLAIN_DEFAULT_ARGS ?= nayantha03 10	
RUN_SERIAL_DEFAULT_ARGS ?= ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb 1
.PHONY: run-plain run-serial
run-plain: $(PLAIN_BIN)
	@ARGS="$(ARGS)"; \
	if [ -z "$$ARGS" ]; then \
		echo 'No ARGS provided; using default:' "$(RUN_PLAIN_DEFAULT_ARGS)"; \
		ARGS="$(RUN_PLAIN_DEFAULT_ARGS)"; \
	fi; \
	"$(PLAIN_BIN)" $$ARGS

run-serial: $(SERIAL_BIN)
	@ARGS="$(ARGS)"; \
	if [ -z "$$ARGS" ]; then \
		echo 'No ARGS provided; using default:' "$(RUN_SERIAL_DEFAULT_ARGS)"; \
		ARGS="$(RUN_SERIAL_DEFAULT_ARGS)"; \
	fi; \
	"$(SERIAL_BIN)" $$ARGS

# Housekeeping
.PHONY: clean distclean
clean:
	@rm -rf "$(BIN_DIR)"

distclean: clean
