# linalg

Dense real linear algebra for [Goose](https://aardappel.github.io/goose/), in pure Goose: no C, no
build step, nothing to install. An optional module, `linalg.parallel`, runs the large products on
several threads.

```goose
import linalg from "guillochon/goose-linalg@v0.2.0";   // or from "../goose-linalg" for a local checkout

fn main() {
    let a = linalg::from_rows(3, 3, [4, 1, 2, 1, 3, 0, 2, 0, 5]);
    let x, ok = linalg::solve(a, [1, 2, 3]);
    print(a, " det ", linalg::det(a), " x ", x);
}
```

Goose fetches the package the first time a program uses it. Pin a tag, as here, or a commit
(`"guillochon/goose-linalg@<commit>"`). A branch like `@main` is pinned at whatever was fetched
first, until `goose fetch --update`.

## Conventions

- A `linalg::matrix` is `{ rows, cols, data: f64[>..] }`, row-major. Vectors are plain f64
  arrays: functions take any array kind as `f64[:]` and return `f64[>..]`.
- Functions that only **read** a matrix take an untyped parameter, so both named matrices (passed
  by reference, never copied) and temporaries work: `linalg::matmul(a, linalg::transpose(a))`.
- Functions that **change** a matrix take `matrix&`; pass a variable.
- Results are built straight into the caller's variable. Decompositions with several outputs write
  into variables you pass, LAPACK style; their previous contents don't matter:
  `var q = linalg::zeros(0, 0); var r = linalg::zeros(0, 0); linalg::qr(a, q, r);`
- To update a matrix from itself, compute into a new variable and assign a copy: `let t =
  linalg::matmul(acc, a); acc = copy(t);` (Goose builds an array's new contents in place, so
  `acc = matmul(acc, a)` is rejected.)
- Shape mismatches and out-of-range indices abort. Numerical failure (singular, not positive
  definite, rank deficient, not converged) comes back as a trailing `bool`.
- Everything is in namespace `linalg`. Goose's UFCS doesn't cross namespaces, so write
  `linalg::trace(a)`, not `a.trace()`.
- A literal's first element sets its type: `[1, 2.5]` is an integer literal and won't compile as
  `f64[:]`. Write `[1.0, 2.5]`. Literals that are all integers, like `[1, 2, 3]`, adapt to `f64`
  fine.

## API

| Area | Functions |
|---|---|
| Construction | `zeros(r, c)`, `filled(r, c, v)`, `identity(n)`, `from_rows(r, c, values)`, `diag(values)`, `random(r, c, rng)`, `clone(a)` |
| Access | `at(a, i, j)`, `set(a&, i, j, v)`, `row(a, i)` (a writable view), `col(a, j)`, `diagonal(a)`, `submatrix(a, r0, r1, c0, c1)`, `set_block(a&, r0, c0, b)`, `is_square`, `same_shape`; `print(m)` shows `[1.0 2.0; 3.0 4.0]` |
| Elementwise | `add`, `sub`, `scale(a, s)`, `hadamard`; in place: `add_assign`, `sub_assign`, `scale_assign`, `add_scaled(a&, s, b)` |
| Products | `matmul`, `matvec(a, x)`, `outer(x, y)`, `transpose`, `power(a, k)` |
| Vectors | `dot`, `vnorm` (overflow-safe), `vnorm1`, `vnorm_inf`, `vadd`, `vsub`, `vscale`, `axpy(alpha, x, y)` |
| Matrix scalars | `trace`, `norm_fro`, `norm1`, `norm_inf`, `norm2`, `max_abs`, `max_abs_diff`, `approx_equal(a, b, tol)`, `is_symmetric(a, tol)`, `cond` |
| Triangular | `solve_lower(l, b, unit_diag)`, `solve_upper(u, b)` |
| LU | `lu_factor(a&, piv&) -> sign, ok`, `lu_solve(lu, piv, b)`, `det`, `solve(a, b) -> x, ok`, `solve_matrix(a, b) -> x, ok`, `inverse(a) -> inv, ok` |
| Cholesky | `cholesky(a) -> l, ok`, `cholesky_solve(l, b)` |
| QR | `qr(a, q&, r&)` (Householder, any shape), `lstsq(a, b) -> x, ok` |
| Symmetric eigen | `eigh(a, values&, vectors&) -> converged` (tridiagonal QL, ascending; reads the upper triangle), `eigvals_sym(a)` |
| SVD | `svd(a, u&, s&, v&) -> converged` (thin, Golub-Kahan, descending), `singular_values`, `rank(a, tol)`, `pinv(a, tol)` |
| Multithreaded (`import linalg.parallel`) | `matmul_mt`, `lu_factor_mt`, `det_mt`, `solve_mt`, `solve_matrix_mt`, `inverse_mt`, `cholesky_mt`; `set_threads(n)`, `threads()`, `stop_threads()`, `parallel_jobs()` |

## Algorithms and accuracy

- `lu_factor` uses partial pivoting, blocked twice as LAPACK's `getrf`. Within an outer block of
  256 columns it factors 32-column panels, each updating the rest of the outer block with one
  matrix multiply; then the outer block updates the whole trailing matrix, 256 deep. Up to 64
  columns it runs unblocked. `det`, `solve`, `solve_matrix` and `inverse` are built on it. The
  triangular solves behind `solve_matrix` and `inverse` are blocked the same two ways. `inverse`
  also skips the zeros of L⁻¹, so it costs about 2n³ flops rather than 2⅔n³.
- `cholesky` is blocked the same way, as `potrf`: 32-column panels within 256-column outer
  blocks, each level updating its trailing lower triangle with matrix multiplies.
- `qr` uses Householder reflections, blocked as `geqrf`/`orgqr` with the compact WY form. The 32
  reflectors of a panel combine into I − V·T·Vᵀ, so the rest of R, and later Q, take each block
  as two matrix multiplies. `lstsq` solves through QR and refuses rank-deficient input rather
  than returning garbage.
- `eigh` reduces the matrix to tridiagonal form with Householder reflections, then diagonalizes it
  with implicit QL steps and Wilkinson shifts. These are EISPACK's `tred2` and `tql2`, ported from
  their public-domain JAMA form and run on the transposed matrix so their inner loops follow rows.
  The eigenvector basis is built from the reflectors in compact WY blocks, like `qr`'s Q.
- `svd` uses Golub-Kahan-Reinsch. Householder reflections reduce the matrix to bidiagonal form,
  then implicit-shift QR steps diagonalize it. This is LINPACK's `dsvdc` by way of JAMA, and
  LAPACK's SVD routines build on the same reduction. Like `eigh`, it works on transposed
  storage, and builds U and V from their reflectors in compact WY blocks. Every singular value is accurate to about ε·σ_max, so a value far below σ_max is
  only accurate in absolute terms (a one-sided Jacobi SVD would also get those right relative
  to their own size).
- `svd` always returns orthonormal `u` and `v`, rank-deficient matrices included.
- Every sum goes through `dot`, which keeps 16 partial sums in a fixed array of lanes. Results are
  deterministic for a given build: the summation order is fixed in the source, not chosen by the
  compiler.

## Building for speed

`goose build` compiles with plain `/O2` (MSVC) or `-O2`, which targets baseline x86-64 (SSE2). For
production, enable the CPU's vector units. With MSVC, set `CL` before building:

```bash
CL="/arch:AVX2 /fp:contract" goose build yourprogram.goose
```

With gcc or clang, the equivalent flags are `-march=x86-64-v3 -ffp-contract=fast`.

Don't use `/fp:fast` or `-ffast-math`. They gain nothing here, because `linalg`'s own kernels
already vectorize their sums. And they break IEEE semantics: under `/fp:fast`, MSVC folds
`1.0 / 0.0` to `1e300`, and checks for NaN and infinity stop working. One check in the test
suite fails under it. It also ends `linalg.parallel`'s bit-for-bit match with the serial results:
fast-math lets the compiler build the worker threads' copy of the kernel with a different
summation order.

## Multithreading

```goose
import linalg from "guillochon/goose-linalg@v0.2.0";
import linalg.parallel;

fn main() {
    linalg::set_threads(6);                     // optional: hardware_threads() by default
    var r = rng { 1 };
    let a = linalg::random(1600, 1600, r);
    let c = linalg::matmul_mt(a, a);
    let inv, ok = linalg::inverse_mt(a);
}
```

A program that imports `linalg.parallel` contains a `thread_fn`, so it must be built with
`goose build`: the in-process `goose run` can't run threads. Programs that import only `linalg`
are unaffected.

Each `_mt` function is its serial namesake with one change: the kernel of its large matrix
products, `gemm_mt`, shares a product's rows (or columns) out between a pool of worker threads and
the calling thread. The results are **bit for bit the serial ones** under IEEE floating point (not
under `/fp:fast`).

Goose threads share no memory: a worker is a separate program, and its operands and results are
copied through typed queues. On the test machine a number costs about 1.6 ns to cross a queue
(each message is a fresh allocation), so sharing a product pays only when it does enough work per
number copied. `gemm_mt` estimates both sides, and keeps any product that wouldn't pay on the
calling thread. In practice:

- `matmul_mt` pays from about n = 500.
- The LU family (`lu_factor_mt`, `det_mt`, `solve_mt`, `solve_matrix_mt`, `inverse_mt`) and
  `cholesky_mt` share only their 256-deep outer updates, from about n = 1000. Up to n = 256 they
  are the serial code.
- `qr`, `lstsq`, `eigh` and `svd` have no `_mt` versions. Their products are blocks of 32
  reflectors, too narrow ever to pay for the copies.

So an `_mt` function is never slower than its serial version (measured from n = 100 to 1600 on 2
and 6 threads, the worst ratio was 0.98×, within noise), and faster where there is enough work to
share.

On a CPU with performance and efficiency cores, set the thread count to the number of performance
cores. Call the `_mt` functions from main's thread: they share one pool and its queues, which live
in globals no worker can reach, so a `thread_fn` that calls them is a compile error rather than a
race.

## Performance

### Test setup

| | |
|---|---|
| CPU | Intel Core i5-12600K (Alder Lake). 6 performance cores with Hyper-Threading: logical CPUs 0–11, 3.7 GHz base, 4.9 GHz turbo, 48 KB L1d and 1.25 MB L2 each. 4 efficiency cores: logical CPUs 12–15, 3.6 GHz turbo, 2 MB shared L2. 20 MB shared L3. |
| Memory | 32 GB DDR5-6200 (2 × 16 GB Corsair Vengeance, CMK32GX5M2B6200C36) |
| Board | ASUS ROG STRIX Z690-I GAMING WIFI |
| OS | Windows 11 Home 23H2 (build 22631), High performance power plan |
| Goose | v2026.3, compiled by MSVC 19.44 (Visual Studio 2022, toolset 14.44) with `/O2 /arch:AVX2 /fp:contract` |
| Reference | NumPy 2.2.6 on Python 3.12.0, linked to OpenBLAS 0.3.29 (`DYNAMIC_ARCH`, which picks AVX2/FMA kernels on this CPU) |
| Pinning | Single-threaded runs on logical CPU 4, a performance core. Multithreaded runs use 6 threads pinned to logical CPUs 0, 2, 4, 6, 8 and 10, one per performance core. Unpinned, or on all 16 logical CPUs, OpenBLAS's LU ran more than 100× slower than on one thread. |

Every time is the best of at least three runs and 0.3 s.

### Single-threaded

![Goose linalg against OpenBLAS](docs/benchmark.png)

On the left is the time ratio to OpenBLAS at each size; 1x means parity. On the right are
absolute times at n = 400:

| | matmul | solve | inverse | cholesky | qr | eigh | svd |
|---|---|---|---|---|---|---|---|
| Goose | 2.6 ms | 1.7 ms | 4.5 ms | 1.1 ms | 6.9 ms | 22 ms | 48 ms |
| OpenBLAS | 2.0 ms | 1.3 ms | 4.4 ms | 1.1 ms | 6.1 ms | 11 ms | 22 ms |
| ratio | 1.25× | 1.3× | 1.0× | 1.0× | 1.1× | 1.9× | 2.1× |

`matmul` runs at 50 GFLOP/s against OpenBLAS's 62. At n = 50, `matvec`, `solve`, `det`,
`inverse`, `cholesky` and `eigh` are as fast as NumPy or faster.

What makes the kernels fast:
- Inner loops run over row slices with `assert`ed lengths, so Goose drops their bounds checks.
- Update loops (`ri[j] -= f * rk[j]`) then vectorize as written.
- Sums use `dot`'s lane array, the one form MSVC vectorizes without fast-math.
- One matrix-multiply kernel, `gemm_acc`, works on blocks of matrices. It's register-blocked four
  rows of the result by four steps of the inner dimension. `matmul` is a single call to it, and
  the blocked LU, Cholesky, QR and the eigenvector/singular-vector bases do most of their work in
  it.
- Large products go through the right-hand operand in cache blocks of 256 × 512, so a product
  bigger than the caches runs nearly as fast as one that fits.

`eigh` and `svd` remain about 2× behind. Their reductions to tridiagonal/bidiagonal form are
still one reflector at a time, and their QR iterations spend most of their time on plane
rotations. LAPACK replaces both with blocked reductions and divide-and-conquer solvers.

### Multithreaded

![linalg.parallel against OpenBLAS](docs/benchmark_mt.png)

Times at n = 1600, and the speedups on 6 threads over 1:

| n = 1600 | matmul | det | solve | solve_matrix | inverse | cholesky |
|---|---|---|---|---|---|---|
| Goose, 1 thread | 184 ms | 88 ms | 82 ms | 287 ms | 258 ms | 51 ms |
| Goose, 6 threads | 68 ms | 67 ms | 64 ms | 219 ms | 195 ms | 49 ms |
| speedup | **2.7×** | 1.3× | 1.3× | 1.3× | 1.3× | 1.0× |
| OpenBLAS, 1 thread | 121 ms | 53 ms | 54 ms | 199 ms | 194 ms | 40 ms |
| OpenBLAS, 6 threads | 24 ms | 18 ms | 54 ms | 62 ms | 57 ms | 26 ms |
| speedup | 5.0× | 2.9× | 1.0× | 3.2× | 3.4× | 1.6× |

On one thread Goose is 1.3–1.7× behind OpenBLAS at this size. On six it is 1.2–3.7× behind,
because OpenBLAS's threads share the matrix in memory while Goose's copy their share of it. At
n ≤ 800 only `matmul` gains (1.8× at n = 800); everything else runs serially there, at serial
speed.

### Reproducing

In `bench/` on Windows, with Goose on `PATH` (or `$env:GOOSE` set):

```bash
powershell -File run_pinned.ps1
```
```bash
python compare.py
```
```bash
python compare_mt.py
```

`run_pinned.ps1` builds the single-threaded benchmark three ways (plain `/O2`, production flags,
and `/fp:fast` for reference) and the multithreaded one with production flags. It runs each,
plus OpenBLAS on one and on six threads, pinned as above; `-Core`, `-Threads` and `-ThreadMask`
change the pinning for another machine.

## Tests

```bash
goose test/test_linalg.goose
```

The suite runs 1,230 checks and exits 1 on any failure. It has these kinds of checks:
- **Exact cases worked by hand:** the Hilbert inverse, a textbook Cholesky factor, Fibonacci
  through `power`, the path-graph Laplacian spectrum, and Wilkinson's W₂₁⁺, whose eigenvalues come in
  pairs that agree to about 10⁻¹⁴.
- **SVD edge cases:** matrices that force the bidiagonal QR to split and to deflate, graded
  singular values from 1 down to 10⁻¹⁵, and clustered singular values that only converge with
  the Wilkinson shift.
- **Blocked sizes:** n = 33, 64, 97 and 300, past the 32-column panels and the 256-column
  outer blocks, for every blocked routine, with singular and indefinite matrices whose trouble
  lies past the first block; and a product past the matrix-multiply cache blocks.
- **Identities on random matrices:** checked at sizes 1–29, square, tall and wide. These cover
  `p a = l u`, `q r = a` with `q` orthogonal, `a v = v diag(w)`, `u s vᵀ = a`, the four
  Moore-Penrose conditions, `det(ab) = det(a) det(b)`, and more.

The parallel module has its own suite, which needs a native build:

```bash
goose build test/test_parallel.goose && test/test_parallel
```

Its 237 checks require every `_mt` result to equal the serial one bit for bit, at 1, 2, 3, 5 and 6
threads, through both ways of sharing a product (by rows and by columns), across a pool restart
and a resize. They also check that large products go to the pool and small ones don't.

Both suites pass as native MSVC builds, plain and with production flags, and the serial one also
in-process. They were also mutation-tested: 99 bugs were planted one at a time, 78 in the library
and 21 in the parallel module, and every one was caught. They covered `gemm_acc` and its cache
blocks, the two-level LU, solves and Cholesky, the compact-WY helpers, every branch of the
Golub-Kahan iteration, `tred2`/`tql2`, the `dot` lanes, and, in the parallel module, the row and
column shares, copying results back, the cost model and the pool.

`examples/fit.goose` fits a polynomial by least squares and finds a point cloud's principal axes.

## License

Apache-2.0, the same license as Goose itself (see `LICENSE`). The symmetric eigensolver and the SVD
are ports of the public-domain JAMA library's versions of EISPACK's `tred2`/`tql2` and LINPACK's
`dsvdc`.
