# bruteforce-demo

Educational brute-force password cracking **simulation** — serial baseline implementations only (plaintext and SHA-256).  
This repository is for learning and demonstrating parallelism later. **Do not use on real/unauthorized accounts.** Use only test passwords/hashes you create.

---

## Repo layout

```
bruteforce-demo/
├─ README.md
├─ Makefile
├─ plain/
│  └─ brute_plain.c        # plaintext brute-force (no deps)
├─ serial/
│  └─ brute_force_serial.c # SHA-256 brute-force (requires OpenSSL)
└─ data/
    └─ test_hashes.txt      # optional test hashes
```

---

## Quick summary

- `plain/brute_plain.c` — minimal plaintext tester, no external libs. Good for quick checks.
- `serial/brute_force_serial.c` — uses OpenSSL EVP to compute SHA-256 and compare to a target hash (serial baseline for later parallel implementations).

---

## Requirements

- GCC (or compatible C compiler)
- Make
- For SHA-256 program: OpenSSL dev headers & library
  - Ubuntu/Debian: `sudo apt-get install build-essential libssl-dev`

---

## Build

From the repo root:

```bash
# build both programs
make

# or build just one:
make plain
make serial
```

Binaries are created in `bin/`:

- `bin/brute_plain` — plaintext testing binary
- `bin/brute_serial` — SHA-256 serial brute-force binary

---

## Usage

Plaintext tester (fast, no deps)

You can run via Make defaults (no args needed):

```bash
make run-plain
```

This uses default args `a 1`. To override:

```bash
make run-plain ARGS='ab1 3'
```

Or run the binary directly:

```bash
bin/brute_plain <target_password> <max_len>
```

Note: CHARSET is defined inside `plain/brute_plain.c`. Keep charset small for quick tests.

SHA-256 serial tester (requires OpenSSL)

1. Generate a test hash locally:

```bash
# produce SHA-256 hex digest (no newline)
echo -n "a1" | openssl dgst -sha256
# copy the hex digest from the output
```

2. Run the serial program:

Using Make with defaults (no args needed):

```bash
make run-serial
```

This uses the SHA-256 of "a" with `max_len=1` as a demo. To override:

```bash
make run-serial ARGS='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 4'
```

Or run the binary directly:

```bash
bin/brute_serial <hex-digest> <max_len>
```

Start with a small charset and small `max_len` for demos. The charset and brute logic live in `serial/brute_force_serial.c` and can be adjusted before compiling.

---

## Ethical & safety note

This repository is strictly for educational purposes. Use only test passwords/hashes you own. Do not target real or unauthorized accounts or systems. Include this statement when you demo or publish the repo.

---

## License

Licensed under the MIT License. See `LICENSE` for the full text.

While the license is permissive, please act responsibly and ethically—use only with test data and in permitted environments.
