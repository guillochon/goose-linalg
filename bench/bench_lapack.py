"""The same operations as bench.goose, through NumPy on OpenBLAS/LAPACK.

    python bench/bench_lapack.py [label]      -> bench_<label>.csv (default "lapack")

Set OPENBLAS_NUM_THREADS=1 before running for a single-threaded comparison.
The same timing rule as the Goose side: best of at least 3 runs and 0.3 s.
"""
import sys
import time
import numpy as np

SIZES = [50, 100, 200, 400]
label = sys.argv[1] if len(sys.argv) > 1 else "lapack"


def time_it(op, n, f):
    best, total, reps = float("inf"), 0.0, 0
    while (reps < 3 and total < 5.0) or total < 0.3:
        t0 = time.perf_counter()
        f()
        dt = time.perf_counter() - t0
        best, total, reps = min(best, dt), total + dt, reps + 1
    rows.append((op, n, best))
    print(f"{op} n={n}: {best * 1000:.3f} ms ({reps} runs)")


rows = []
rng = np.random.default_rng(42)
for n in SIZES:
    a = rng.uniform(-1, 1, (n, n))
    b = rng.uniform(-1, 1, (n, n))
    x = rng.uniform(-1, 1, n)
    spd = a @ a.T + n * np.eye(n)
    sym = a + a.T
    time_it("matmul", n, lambda: a @ b)                             # dgemm
    time_it("matvec", n, lambda: a @ x)                             # dgemv
    time_it("solve", n, lambda: np.linalg.solve(a, x))              # dgesv
    time_it("det", n, lambda: np.linalg.det(a))                     # dgetrf
    time_it("inverse", n, lambda: np.linalg.inv(a))                 # dgesv on I
    time_it("cholesky", n, lambda: np.linalg.cholesky(spd))         # dpotrf
    time_it("qr", n, lambda: np.linalg.qr(a, mode="complete"))      # dgeqrf + dorgqr
    time_it("eigh", n, lambda: np.linalg.eigh(sym))                 # dsyevd
    time_it("svd", n, lambda: np.linalg.svd(a, full_matrices=False))  # dgesdd

with open(f"bench_{label}.csv", "w") as f:
    f.write("op,n,seconds\n")
    for op, n, s in rows:
        f.write(f"{op},{n},{s}\n")
