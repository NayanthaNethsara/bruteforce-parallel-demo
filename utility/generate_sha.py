import hashlib
import sys

def main():
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = input("Enter password to hash: ")

    # Calculate SHA256
    sha256_hash = hashlib.sha256(password.encode()).hexdigest()
    length = len(password)

    # Output to file 'target.txt'
    with open("target.txt", "w") as f:
        f.write(f"{sha256_hash} {length}\n")

    print(f"Generated 'target.txt' with:")
    print(f"SHA256: {sha256_hash}")
    print(f"Length: {length}")
    print("-" * 60)
    print("You can now run the brute force programs using the content of target.txt.")
    print("Example run commands (copy-paste):")
    print(f"./brute_sha256_openmp {sha256_hash} {length}")
    print(f"mpirun -np 4 ./brute_sha256_mpi {sha256_hash} {length}")
    print(f"./brute_sha256_cuda {sha256_hash} {length}")

if __name__ == "__main__":
    main()
