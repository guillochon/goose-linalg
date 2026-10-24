# linalg

Dense real linear algebra for [Goose](https://aardappel.github.io/goose/), in pure Goose: no C, no
build step, nothing to install.

```goose
import linalg from "guillochon/goose-linalg@main";     // or from "../goose-linalg" for a local checkout

fn main() {
    let a = linalg::from_rows(3, 3, [4, 1, 2, 1, 3, 0, 2, 0, 5]);
    let x, ok = linalg::solve(a, [1, 2, 3]);
    print(a, " det ", linalg::det(a), " x ", x);
}
```

Goose fetches the package the first time a program uses it. A branch like `@main` is pinned at
whatever was fetched first, until `goose fetch --update`. For reproducible builds, pin a commit
instead: `"guillochon/goose-linalg@<commit>"`.

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

## Algorithms and accuracy

- `lu_factor` uses partial pivoting. `det`, `solve` and `inverse` are built on it, and
  `solve_matrix` / `inverse` work on whole rows of right-hand sides at once.
- `qr` uses Householder reflections. `lstsq` solves through QR and refuses rank-deficient input
  rather than returning garbage.
- `eigh` reduces the matrix to tridiagonal form with Householder reflections, then diagonalizes it
  with implicit QL steps and Wilkinson shifts. These are EISPACK's `tred2` and `tql2`, ported from
  their public-domain JAMA form and run on the transposed matrix so their inner loops follow rows.
- `svd` uses Golub-Kahan-Reinsch. Householder reflections reduce the matrix to bidiagonal form,
  then implicit-shift QR steps diagonalize it. This is LINPACK's `dsvdc` by way of JAMA, and
  LAPACK's SVD routines build on the same reduction. Like `eigh`, it works on transposed
  storage. Every singular value is accurate to about ε·σ_max, so a value far below σ_max is
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
suite fails under it.

## Performance

The `bench/` folder times every operation against NumPy on OpenBLAS 0.3.29, which dispatches to
AVX2/FMA kernels at run time. The setup is production on both sides: Goose built with
`/arch:AVX2 /fp:contract`, OpenBLAS on one thread, both pinned to the same performance core of an
i5-12600K.

![Goose linalg against OpenBLAS](docs/benchmark.png)

On the left is the time ratio to OpenBLAS at each size; 1x means parity. On the right are
absolute times at n = 400:

| | matmul | solve | inverse | cholesky | qr | eigh | svd |
|---|---|---|---|---|---|---|---|
| Goose | 2.6 ms | 3.0 ms | 7.9 ms | 2.0 ms | 18 ms | 26 ms | 54 ms |
| OpenBLAS | 2.1 ms | 1.3 ms | 4.4 ms | 1.1 ms | 6.2 ms | 11 ms | 22 ms |
| ratio | 1.2× | 2.3× | 1.8× | 1.8× | 2.9× | 2.3× | 2.4× |

`matmul` runs at 49 GFLOP/s against OpenBLAS's 61. At n = 50, `matvec`, `solve`, `det`,
`cholesky`, `eigh` and `svd` are as fast as NumPy or faster.

What makes the kernels fast:
- Inner loops run over row slices with `assert`ed lengths, so Goose drops their bounds checks.
- Update loops (`ri[j] -= f * rk[j]`) then vectorize as written.
- Sums use `dot`'s lane array, the one form MSVC vectorizes without fast-math.
- `matmul` is register-blocked, four rows of the result by four steps of the inner dimension.

Everything is single-threaded. On this hybrid CPU, multi-threaded OpenBLAS was more than 100× *slower* for
LU at n = 400 than single-threaded OpenBLAS.

To reproduce, in `bench/` on Windows with Goose on `PATH` (or `$env:GOOSE` set):

```bash
powershell -File run_pinned.ps1
```
```bash
python compare.py
```

`run_pinned.ps1` builds the benchmark three ways (plain `/O2`, production flags and `/fp:fast` for
reference) and runs each, plus OpenBLAS, pinned to one core.

## Tests

```bash
goose test/test_linalg.goose
```

The suite runs 1,117 checks and exits 1 on any failure. It has two kinds of checks:
- **Exact cases worked by hand:** the Hilbert inverse, a textbook Cholesky factor, Fibonacci
  through `power`, the path-graph Laplacian spectrum, and Wilkinson's W₂₁⁺, whose eigenvalues come in
  pairs that agree to about 10⁻¹⁴.
- **SVD edge cases:** matrices that force the bidiagonal QR to split and to deflate, graded
  singular values from 1 down to 10⁻¹⁵, and clustered singular values that only converge with
  the Wilkinson shift.
- **Identities on random matrices:** checked at sizes 1–29, square, tall and wide. These cover
  `p a = l u`, `q r = a` with `q` orthogonal, `a v = v diag(w)`, `u s vᵀ = a`, the four
  Moore-Penrose conditions, `det(ab) = det(a) det(b)`, and more.

The suite passes in-process, as a native MSVC build, and with production flags. It was also
mutation-tested: 48 bugs were planted in the library one at a time. Every one that changes the
library's behavior was caught; the one survivor swaps the Wilkinson shift for a simpler shift that
also converges. The mutants covered the blocked `matmul`, the `dot` lanes, `tred2`/`tql2`, every
branch of the Golub-Kahan iteration, and the multi-RHS solve.

`examples/fit.goose` fits a polynomial by least squares and finds a point cloud's principal axes.

## License

Apache-2.0, the same license as Goose itself (see `LICENSE`). The symmetric eigensolver and the SVD
are ports of the public-domain JAMA library's versions of EISPACK's `tred2`/`tql2` and LINPACK's
`dsvdc`.
