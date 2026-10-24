// linalg: dense real linear algebra in pure Goose.
//
// A `matrix` is row-major f64 storage with its shape. Vectors are plain f64
// arrays: functions take any of them as `f64[:]` and return `f64[>..]`.
//
// Functions that only read a matrix take it as an untyped parameter, so a
// named matrix is passed by reference and a temporary, `matmul(a,
// transpose(a))`, works too. Functions that change one take `matrix&`.
// Results come back by value, built straight into the caller's variable.
// Decompositions with several outputs write them into the caller's
// variables, LAPACK style: `qr(a, q, r)`, `eigh(a, values, vectors)`,
// `svd(a, u, s, v)`.
//
// Everything here is in namespace linalg: `linalg::matmul(a, b)`. Shape
// mismatches and out-of-range indices are programming errors and abort;
// numerical failure (a singular matrix, one that is not positive definite)
// comes back as a trailing bool.

import std;
import math;

namespace linalg;

struct matrix { rows: i64, cols: i64, data: f64[>..] }

// Machine epsilon for f64.
let EPS = 2.220446049250313e-16;

// ---------------------------------------------------------------------------
// Construction

fn zeros(rows: i64, cols: i64) -> matrix { filled(rows, cols, 0.0) }

fn filled(rows: i64, cols: i64, v: f64) -> matrix {
    assert(rows >= 0 && cols >= 0);
    var m = matrix { rows: rows, cols: cols, data: [] };
    m.data.resize(rows * cols, v);
    return m;
}

fn identity(n: i64) -> matrix {
    var m = zeros(n, n);
    for i in n { m.data[i * n + i] = 1.0; }
    return m;
}

// From row-major values: from_rows(2, 2, [1, 2, 3, 4]) is [1 2; 3 4].
fn from_rows(rows: i64, cols: i64, values: f64[:]) -> matrix {
    assert(rows >= 0 && cols >= 0 && values.len == rows * cols);
    var m = matrix { rows: rows, cols: cols, data: [] };
    m.data.append(values);
    return m;
}

// A square matrix with `values` on its diagonal.
fn diag(values: f64[:]) -> matrix {
    let n = values.len;
    var m = zeros(n, n);
    for v, i in values { m.data[i * n + i] = v; }
    return m;
}

// Entries uniform in [-1, 1).
fn random(rows: i64, cols: i64, r: rng&) -> matrix {
    var m = zeros(rows, cols);
    for &x in m.data { x = 2.0 * r.rand_flt() - 1.0; }
    return m;
}

fn clone(a) -> matrix {
    var m = matrix { rows: a.rows, cols: a.cols, data: [] };
    m.data.append(a.data);
    return m;
}

// ---------------------------------------------------------------------------
// Access and shape

fn at(a, i: i64, j: i64) -> f64 {
    assert(i >= 0 && i < a.rows && j >= 0 && j < a.cols);
    return a.data[i * a.cols + j];
}

fn set(a: matrix&, i: i64, j: i64, v: f64) {
    assert(i >= 0 && i < a.rows && j >= 0 && j < a.cols);
    a.data[i * a.cols + j] = v;
}

// Row i as a view into the matrix: writable when the matrix is.
fn row(a, i: i64) -> f64[:] {
    assert(i >= 0 && i < a.rows);
    return a.data[i * a.cols..(i + 1) * a.cols];
}

fn col(a, j: i64) -> f64[>..] {
    assert(j >= 0 && j < a.cols);
    var out: f64[>..] = [];
    for i in a.rows { out.push(a.data[i * a.cols + j]); }
    return out;
}

fn diagonal(a) -> f64[>..] {
    var out: f64[>..] = [];
    for i in min(a.rows, a.cols) { out.push(a.data[i * a.cols + i]); }
    return out;
}

// Rows r0..r1 and columns c0..c1, half-open, as a new matrix.
fn submatrix(a, r0: i64, r1: i64, c0: i64, c1: i64) -> matrix {
    assert(0 <= r0 && r0 <= r1 && r1 <= a.rows && 0 <= c0 && c0 <= c1 && c1 <= a.cols);
    var m = zeros(r1 - r0, c1 - c0);
    for i in r0..r1 { for j in c0..c1 { m.data[(i - r0) * m.cols + (j - c0)] = a.data[i * a.cols + j]; } }
    return m;
}

// Writes b into a with its top-left corner at (r0, c0).
fn set_block(a: matrix&, r0: i64, c0: i64, b) {
    assert(r0 >= 0 && c0 >= 0 && r0 + b.rows <= a.rows && c0 + b.cols <= a.cols);
    for i in b.rows { for j in b.cols { a.data[(r0 + i) * a.cols + c0 + j] = b.data[i * b.cols + j]; } }
}

fn is_square(a) -> bool { a.rows == a.cols }
fn same_shape(a, b) -> bool { a.rows == b.rows && a.cols == b.cols }

// Renders as [1.0 2.0; 3.0 4.0], so print(m) works.
fn format(out: u8[>..]&, m: matrix&) {
    out.push('[');
    for i in m.rows {
        if i > 0 { out.append("; "); }
        for j in m.cols {
            if j > 0 { out.push(' '); }
            format(out, m.data[i * m.cols + j]);
        }
    }
    out.push(']');
}

// ---------------------------------------------------------------------------
// Elementwise arithmetic

fn add(a, b) -> matrix {
    assert(same_shape(a, b));
    var c = clone(a);
    for x, i in b.data { c.data[i] += x; }
    return c;
}

fn sub(a, b) -> matrix {
    assert(same_shape(a, b));
    var c = clone(a);
    for x, i in b.data { c.data[i] -= x; }
    return c;
}

fn scale(a, s: f64) -> matrix {
    var c = clone(a);
    for &x in c.data { x *= s; }
    return c;
}

// The elementwise product.
fn hadamard(a, b) -> matrix {
    assert(same_shape(a, b));
    var c = clone(a);
    for x, i in b.data { c.data[i] *= x; }
    return c;
}

// In place: a += b, a -= b, a *= s, a += s * b.
fn add_assign(a: matrix&, b) {
    assert(same_shape(a, b));
    for x, i in b.data { a.data[i] += x; }
}

fn sub_assign(a: matrix&, b) {
    assert(same_shape(a, b));
    for x, i in b.data { a.data[i] -= x; }
}

fn scale_assign(a: matrix&, s: f64) {
    for &x in a.data { x *= s; }
}

fn add_scaled(a: matrix&, s: f64, b) {
    assert(same_shape(a, b));
    for x, i in b.data { a.data[i] += s * x; }
}

// ---------------------------------------------------------------------------
// Products

fn transpose(a) -> matrix {
    var t = zeros(a.cols, a.rows);
    for i in a.rows { for j in a.cols { t.data[j * a.rows + i] = a.data[i * a.cols + j]; } }
    return t;
}

// a * b.
fn matmul_with<G>(a, b) -> matrix {
    assert(a.cols == b.rows);
    var c = zeros(a.rows, b.cols);
    G(c, 0, 0, a, 0, 0, false, b, 0, 0, a.rows, b.cols, a.cols, 1.0);
    return c;
}

// matmul_with on this thread: the kernel is gemm_acc.
fn matmul(a, b) -> matrix { return matmul_with(a, b, gemm_acc); }

// The one matrix-multiply kernel, on blocks of matrices:
//   c[ci + i][cj + j] += alpha * sum over t of A(i, t) * b[bi + t][bj + j]
// for i < m, j < n, t < k, where A(i, t) is a[ai + i][aj + t], or
// a[ai + t][aj + i] with trans_a. The blocks may lie in one matrix as long
// as c's block overlaps neither of the others.
//
// Large products go through b in KC x NC blocks that stay in cache, each
// block feeding every row of c before the next is loaded. KC is a multiple
// of four, so the blocks change no sum's order: the result is bit for bit
// what one pass over all of b gives.
fn gemm_acc(c: matrix&, ci: i64, cj: i64, a, ai: i64, aj: i64, trans_a: bool,
            b, bi: i64, bj: i64, m: i64, n: i64, k: i64, alpha: f64) {
    var j0 = 0;
    while j0 < n || j0 == 0 {
        let nb = min(NC, n - j0);
        var t0 = 0;
        while t0 < k || t0 == 0 {
            let kb = min(KC, k - t0);
            if trans_a {
                gemm_block(c, ci, cj + j0, a, ai + t0, aj, true, b, bi + t0, bj + j0, m, nb, kb, alpha);
            } else {
                gemm_block(c, ci, cj + j0, a, ai, aj + t0, false, b, bi + t0, bj + j0, m, nb, kb, alpha);
            }
            t0 += KC;
        }
        j0 += NC;
    }
}

// The cache blocks of gemm_acc: KC steps of the inner dimension by NC
// columns, 256 x 512 doubles, a megabyte.
let KC = 256;
let NC = 512;

// gemm_acc on one cache block. Loops run i-k-j, so the innermost runs along
// rows of c and b. Four rows of c and four rows of b at a time: each element
// of b loaded feeds four rows of c, and each pass over a row of c folds in
// four rows of b. The entries of a are scalars here, so a transposed a
// costs nothing extra.
fn gemm_block(c: matrix&, ci: i64, cj: i64, a, ai: i64, aj: i64, trans_a: bool,
              b, bi: i64, bj: i64, m: i64, n: i64, k: i64, alpha: f64) {
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(ci + m <= c.rows && cj + n <= c.cols && bi + k <= b.rows && bj + n <= b.cols);
    if trans_a { assert(ai + k <= a.rows && aj + m <= a.cols); } else { assert(ai + m <= a.rows && aj + k <= a.cols); }
    let lda = a.cols;
    fn av(i: i64, t: i64) -> f64 {
        alpha * (if trans_a { a.data[(ai + t) * lda + aj + i] } else { a.data[(ai + i) * lda + aj + t] })
    }
    fn crow(i: i64) -> f64[:] { row(c, ci + i)[cj..cj + n] }
    fn brow(t: i64) -> f64[:] { row(b, bi + t)[bj..bj + n] }
    var i = 0;
    while i + 4 <= m {
        let c0 = crow(i);
        let c1 = crow(i + 1);
        let c2 = crow(i + 2);
        let c3 = crow(i + 3);
        assert(c0.len == n && c1.len == n && c2.len == n && c3.len == n);
        var t = 0;
        while t + 4 <= k {
            let b0 = brow(t);
            let b1 = brow(t + 1);
            let b2 = brow(t + 2);
            let b3 = brow(t + 3);
            assert(b0.len == n && b1.len == n && b2.len == n && b3.len == n);
            let p0 = av(i, t);
            let p1 = av(i, t + 1);
            let p2 = av(i, t + 2);
            let p3 = av(i, t + 3);
            let q0 = av(i + 1, t);
            let q1 = av(i + 1, t + 1);
            let q2 = av(i + 1, t + 2);
            let q3 = av(i + 1, t + 3);
            let r0 = av(i + 2, t);
            let r1 = av(i + 2, t + 1);
            let r2 = av(i + 2, t + 2);
            let r3 = av(i + 2, t + 3);
            let s0 = av(i + 3, t);
            let s1 = av(i + 3, t + 1);
            let s2 = av(i + 3, t + 2);
            let s3 = av(i + 3, t + 3);
            for j in n {
                let v0 = b0[j];
                let v1 = b1[j];
                let v2 = b2[j];
                let v3 = b3[j];
                c0[j] += p0 * v0 + p1 * v1 + p2 * v2 + p3 * v3;
                c1[j] += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
                c2[j] += r0 * v0 + r1 * v1 + r2 * v2 + r3 * v3;
                c3[j] += s0 * v0 + s1 * v1 + s2 * v2 + s3 * v3;
            }
            t += 4;
        }
        while t < k {
            let bt = brow(t);
            assert(bt.len == n);
            let p = av(i, t);
            let q = av(i + 1, t);
            let r = av(i + 2, t);
            let s = av(i + 3, t);
            for j in n {
                let v = bt[j];
                c0[j] += p * v;
                c1[j] += q * v;
                c2[j] += r * v;
                c3[j] += s * v;
            }
            t++;
        }
        i += 4;
    }
    // The last rows, one at a time.
    while i < m {
        let ck = crow(i);
        for t in k { axpy(av(i, t), brow(t), ck); }
        i++;
    }
}

// a * x.
fn matvec(a, x: f64[:]) -> f64[>..] {
    assert(a.cols == x.len);
    var y: f64[>..] = [];
    for i in a.rows { y.push(dot(row(a, i), x)); }
    return y;
}

// x * y^T.
fn outer(x: f64[:], y: f64[:]) -> matrix {
    var m = zeros(x.len, y.len);
    for a, i in x { for b, j in y { m.data[i * y.len + j] = a * b; } }
    return m;
}

// a^k for k >= 0, by repeated squaring.
fn power(a, k: i64) -> matrix {
    assert(a.rows == a.cols && k >= 0);
    var result = identity(a.rows);
    var base = clone(a);
    var e = k;
    while e > 0 {
        if e % 2 == 1 {
            var t = matmul(result, base);
            result = copy(t);
        }
        e /= 2;
        if e > 0 {
            var t = matmul(base, base);
            base = copy(t);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Vectors

// The partial sums a reduction keeps. A fixed array of lanes, updated by an
// inner loop over them, is a summation order the C compiler vectorizes as
// written; a single running sum it may not reorder without fast-math.
let LANES = 16;

fn dot(x: f64[:], y: f64[:]) -> f64 {
    let n = x.len;
    assert(y.len == n);
    // Short vectors don't fill the lanes: setting them up and adding them
    // back would cost more than the loop.
    if n < 2 * LANES {
        var s = 0.0;
        for i in n { s += x[i] * y[i]; }
        return s;
    }
    var acc: f64[LANES] = [0.0; LANES];
    let m = n - n % LANES;
    var i = 0;
    while i < m {
        let xs = x[i..i + LANES];
        let ys = y[i..i + LANES];
        assert(xs.len == LANES && ys.len == LANES);
        for k in LANES { acc[k] += xs[k] * ys[k]; }
        i += LANES;
    }
    var s = 0.0;
    for k in LANES { s += acc[k]; }
    for j in m..n { s += x[j] * y[j]; }
    return s;
}

// The Euclidean norm, scaled so it neither overflows nor underflows.
fn vnorm(x: f64[:]) -> f64 {
    var big = 0.0;
    for v in x { big = max(big, abs(v)); }
    if big == 0.0 || big != big { return big; }
    var s = 0.0;
    for v in x { let t = v / big; s += t * t; }
    return big * sqrt(s);
}

fn vnorm1(x: f64[:]) -> f64 {
    var s = 0.0;
    for v in x { s += abs(v); }
    return s;
}

fn vnorm_inf(x: f64[:]) -> f64 {
    var s = 0.0;
    for v in x { s = max(s, abs(v)); }
    return s;
}

fn vadd(x: f64[:], y: f64[:]) -> f64[>..] {
    assert(x.len == y.len);
    var out: f64[>..] = [];
    for v, i in x { out.push(v + y[i]); }
    return out;
}

fn vsub(x: f64[:], y: f64[:]) -> f64[>..] {
    assert(x.len == y.len);
    var out: f64[>..] = [];
    for v, i in x { out.push(v - y[i]); }
    return out;
}

fn vscale(x: f64[:], s: f64) -> f64[>..] {
    var out: f64[>..] = [];
    for v in x { out.push(v * s); }
    return out;
}

// y += alpha * x, in place.
fn axpy(alpha: f64, x: f64[:], y: f64[:]) {
    assert(x.len == y.len);
    for v, i in x { y[i] += alpha * v; }
}

// ---------------------------------------------------------------------------
// Matrix scalars and comparisons

fn trace(a) -> f64 {
    assert(a.rows == a.cols);
    var s = 0.0;
    for i in a.rows { s += a.data[i * a.cols + i]; }
    return s;
}

// The Frobenius norm: the Euclidean norm of all the entries.
fn norm_fro(a) -> f64 { vnorm(a.data) }

// The largest absolute column sum.
fn norm1(a) -> f64 {
    var best = 0.0;
    for j in a.cols {
        var s = 0.0;
        for i in a.rows { s += abs(a.data[i * a.cols + j]); }
        best = max(best, s);
    }
    return best;
}

// The largest absolute row sum.
fn norm_inf(a) -> f64 {
    var best = 0.0;
    for i in a.rows { best = max(best, vnorm1(row(a, i))); }
    return best;
}

fn max_abs(a) -> f64 { vnorm_inf(a.data) }

fn max_abs_diff(a, b) -> f64 {
    assert(same_shape(a, b));
    var d = 0.0;
    for x, i in a.data { d = max(d, abs(x - b.data[i])); }
    return d;
}

// Same shape and every entry within tol.
fn approx_equal(a, b, tol: f64) -> bool { same_shape(a, b) && max_abs_diff(a, b) <= tol }

fn is_symmetric(a, tol: f64 = 0.0) -> bool {
    guard a.rows == a.cols else { return false; }
    for i in a.rows {
        for j in i + 1..a.cols {
            if abs(a.data[i * a.cols + j] - a.data[j * a.cols + i]) > tol { return false; }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Triangular systems

// Solves l x = b by forward substitution, reading only the lower triangle;
// with unit_diag the diagonal is taken to be ones.
fn solve_lower(l, b: f64[:], unit_diag: bool = false) -> f64[>..] {
    assert(l.rows == l.cols && l.rows == b.len);
    let n = b.len;
    var x: f64[>..] = [];
    x.resize(n, 0.0);
    for i in n {
        let li = row(l, i);
        assert(li.len == n && x.len == n);
        let s = b[i] - dot(li[..i], x[..i]);
        x[i] = if unit_diag { s } else { s / li[i] };
    }
    return x;
}

// Solves u x = b by back substitution, reading only the upper triangle.
fn solve_upper(u, b: f64[:]) -> f64[>..] {
    assert(u.rows == u.cols && u.rows == b.len);
    let n = b.len;
    var x: f64[>..] = [];
    x.resize(n, 0.0);
    for ii in n {
        let i = n - 1 - ii;
        let ui = row(u, i);
        assert(ui.len == n && x.len == n);
        x[i] = (b[i] - dot(ui[i + 1..], x[i + 1..])) / ui[i];
    }
    return x;
}

// ---------------------------------------------------------------------------
// LU decomposition with partial pivoting

// The panel width of the blocked factorizations.
let NB = 32;

// Factors square `a` in place into P a = L U: U on and above the diagonal,
// L's multipliers below it (L's diagonal is ones), and row i of P a is row
// piv[i] of the original. Returns the permutation's sign, and false when a
// is singular (a zero pivot column); the factors are then incomplete.
//
// Blocked twice, as LAPACK's getrf: an outer block of OB columns is
// factored a panel of NB columns at a time, each panel solving for its
// block row of U and updating the rest of the outer block with gemm_acc;
// then the outer block solves for its own block row of U and updates the
// whole trailing matrix with one G product, where most of the work is.
fn lu_factor_with<G>(a: matrix&, piv: i64[>..]&) -> f64, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    piv.clear();
    for i in n { piv.push(i); }
    var sign = 1.0;
    var ok = true;
    // Below a few panels' width, blocking costs more than it saves: one
    // panel covering the matrix is the unblocked algorithm.
    let nb = if n <= 2 * NB { max(n, 1) } else { NB };
    var swaps: i64[>..] = [];
    var o0 = 0;
    while o0 < n {
        let o1 = min(o0 + OB, n);
        let s, pok = lu_outer_panel(a, piv, o0, o1, nb, swaps);
        sign *= s;
        if !pok { ok = false; }
        if o1 < n {
            lu_swap_rows(a, o1, n, swaps, o0, o1);
            lu_outer_update(a, o1, n - o1, a, 0, o0, o0, o1, nb, G);
        }
        o0 = o1;
    }
    return sign, ok;
}

// Swaps each row k in k0..k1 of c with row swaps[k - k0], over columns
// c0..c1, in order: what the row swaps of a panel do to columns it doesn't
// hold.
fn lu_swap_rows(c: matrix&, c0: i64, c1: i64, swaps: i64[:], k0: i64, k1: i64) {
    let ld = c.cols;
    for k in k0..k1 {
        let p = swaps[k - k0];
        if p == k { continue; }
        let rk = c.data[k * ld + c0..k * ld + c1];
        let rp = c.data[p * ld + c0..p * ld + c1];
        assert(rk.len == c1 - c0 && rp.len == c1 - c0);
        for j in c1 - c0 {
            let t = rk[j];
            rk[j] = rp[j];
            rp[j] = t;
        }
    }
}

// Factors the outer block of columns o0..o1, rows o0.. of a, a panel of nb
// columns at a time: each panel picks its pivots, swaps rows, solves for its
// block row of U within the outer block and updates the rest of the outer
// block. Returns the swaps' sign and false if a pivot column was zero, and
// leaves in `swaps` the row swapped with each row o0..o1 (itself when none
// was). The swaps reach columns 0..o1 of a only: the caller applies them to
// the columns from o1 on (lu_swap_rows) before lu_outer_update.
fn lu_outer_panel(a: matrix&, piv: i64[>..]&, o0: i64, o1: i64, nb: i64, swaps: i64[>..]&) -> f64, bool {
    let n = a.rows;
    var sign = 1.0;
    var ok = true;
    swaps.clear();
    var b: f64[>..] = [];
    var k0 = o0;
    while k0 < o1 {
        let k1 = min(k0 + nb, o1);
        let w = k1 - k0;
        let m = n - k0;
        // The panel, rows k0.. of columns k0..k1, copied out column-major:
        // each step below then divides one long contiguous column and folds
        // it into the columns to its right, loops the C compiler vectorizes.
        // Every entry sees the same operations in the same order as it would
        // in a, so the factors are the same bit for bit.
        b.resize(w * m, 0.0);
        for i in m {
            let ri = a.data[(k0 + i) * n + k0..(k0 + i) * n + k1];
            assert(ri.len == w);
            for j in w { b[j * m + i] = ri[j]; }
        }
        for j in w {
            let cj = b[j * m..(j + 1) * m];
            assert(cj.len == m);
            var p = j;
            var big = abs(cj[j]);
            for i in j + 1..m {
                let v = abs(cj[i]);
                if v > big { big = v; p = i; }
            }
            swaps.push(k0 + p);
            if big == 0.0 { ok = false; continue; }
            if p != j {
                for t in w {
                    let x = b[t * m + j];
                    b[t * m + j] = b[t * m + p];
                    b[t * m + p] = x;
                }
                let x = piv[k0 + j];
                piv[k0 + j] = piv[k0 + p];
                piv[k0 + p] = x;
                sign = -sign;
            }
            let pivot = cj[j];
            let l = cj[j + 1..];
            for i in l.len { l[i] = l[i] / pivot; }
            for t in j + 1..w {
                let ct = b[t * m + j..(t + 1) * m];
                assert(ct.len == l.len + 1);
                let u = ct[0];
                let below = ct[1..];
                assert(below.len == l.len);
                for i in l.len { below[i] += -l[i] * u; }
            }
        }
        for i in m {
            let ri = a.data[(k0 + i) * n + k0..(k0 + i) * n + k1];
            assert(ri.len == w);
            for j in w { ri[j] = b[j * m + i]; }
        }
        // The panel's swaps, on the L to its left and the rest of the outer block.
        lu_swap_rows(a, 0, k0, swaps[k0 - o0..], k0, k1);
        lu_swap_rows(a, k1, o1, swaps[k0 - o0..], k0, k1);
        if k1 < o1 {
            // The panel's block row of U, within the outer block: the unit
            // lower panel's inverse applied to columns k1..o1 of its rows.
            for i in k0..k1 {
                let ri = row(a, i)[k1..o1];
                for t in k0..i { axpy(-a.data[i * n + t], row(a, t)[k1..o1], ri); }
            }
            // The rest of the outer block: a22 -= l21 u12.
            gemm_acc(a, k1, k1, a, k1, k0, false, a, k0, k1, n - k1, o1 - k1, k1 - k0, -1.0);
        }
        k0 = k1;
    }
    return sign, ok;
}

// After lu_outer_panel factored the outer block o0..o1, brings w columns of
// c, from column cj, up to date: their block row of U, rows o0..o1, then
// the trailing rows below, OB deep with one G product. c's rows are a's
// rows; l holds the factored outer block, a's row i and column o0 at l's
// row i - lr and column lj. Every column is computed on its own, so the
// result is the same bit for bit whichever columns, and in whatever groups,
// c holds.
fn lu_outer_update<G>(c: matrix&, cj: i64, w: i64, l, lr: i64, lj: i64, o0: i64, o1: i64, nb: i64) {
    let n = c.rows;
    let ld = l.cols;
    // The outer block row of U: the unit lower triangle of the outer
    // block's inverse, nb rows at a time.
    var i0 = o0;
    while i0 < o1 {
        let i1 = min(i0 + nb, o1);
        gemm_acc(c, i0, cj, l, i0 - lr, lj, false, c, o0, cj, i1 - i0, w, i0 - o0, -1.0);
        for i in i0..i1 {
            let ri = row(c, i)[cj..cj + w];
            for t in i0..i { axpy(-l.data[(i - lr) * ld + lj + t - o0], row(c, t)[cj..cj + w], ri); }
        }
        i0 = i1;
    }
    // The trailing matrix: a22 -= l21 u12, OB deep.
    G(c, o1, cj, l, o1 - lr, lj, false, c, o0, cj, n - o1, w, o1 - o0, -1.0);
}

// lu_factor_with on this thread: the kernel is gemm_acc.
fn lu_factor(a: matrix&, piv: i64[>..]&) -> f64, bool { return lu_factor_with(a, piv, gemm_acc); }

// Solves a x = b given lu_factor's output for a.
fn lu_solve(lu, piv: i64[:], b: f64[:]) -> f64[>..] {
    assert(lu.rows == lu.cols && lu.rows == b.len && piv.len == b.len);
    var pb: f64[>..] = [];
    for p in piv { pb.push(b[p]); }
    let y = solve_lower(lu, pb, true);
    return solve_upper(lu, y);
}

// The LU family below factors with F, lu_factor or another function of its
// shape (linalg.parallel passes its own), and the solves' products go
// through G.
fn det_with<F>(a) -> f64 {
    assert(a.rows == a.cols);
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = F(lu, piv);
    guard ok else { return 0.0; }
    var d = sign;
    for i in a.rows { d *= lu.data[i * a.rows + i]; }
    return d;
}

// det_with on this thread: the factorization is lu_factor.
fn det(a) -> f64 { return det_with(a, lu_factor); }

// Solves a x = b for square a; false when a is singular.
fn solve_with<F>(a, b: f64[:]) -> f64[>..], bool {
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = F(lu, piv);
    guard ok else { return [], false; }
    return lu_solve(lu, piv, b), true;
}

// solve_with on this thread: the factorization is lu_factor.
fn solve(a, b: f64[:]) -> f64[>..], bool { return solve_with(a, b, lu_factor); }

// The outer block of the two-level factorizations and solves: products
// this wide in their inner dimension go through G, the kernel the caller
// chose; the NB-wide work inside an outer block stays on gemm_acc.
let OB = 256;

// x = l^-1 x for the unit lower triangle of lu, all of x's columns. Rows go
// in outer blocks of OB, each first taking every row solved above it in one
// G product, then in inner blocks of NB, each taking the rows above it in
// its outer block in one gemm_acc product, then row by row. With tri, x
// starts out as the identity, and row i of the result is zero past column i
// (the forward half of inverse), which every step skips.
fn forward_unit_with<G>(lu, x: matrix&, tri: bool) {
    var below: i64[>..] = [];
    if tri { for r in x.rows + 1 { below.push(r); } }
    forward_unit_cols_with(lu, x, tri, below, G);
}

// forward_unit_with on some of the columns: with tri, x's first below[r]
// columns are the identity's columns before r, and the rest come after it.
// Every column is computed on its own, so each comes out the same bit for
// bit whichever others x holds.
fn forward_unit_cols_with<G>(lu, x: matrix&, tri: bool, below: i64[:]) {
    let n = x.rows;
    let m = x.cols;
    fn width(r: i64) -> i64 { if tri { below[r] } else { m } }
    var o0 = 0;
    while o0 < n {
        let o1 = min(o0 + OB, n);
        G(x, o0, 0, lu, o0, 0, false, x, 0, 0, o1 - o0, width(o0), o0, -1.0);
        var i0 = o0;
        while i0 < o1 {
            let i1 = min(i0 + NB, o1);
            gemm_acc(x, i0, 0, lu, i0, o0, false, x, o0, 0, i1 - i0, width(i0), i0 - o0, -1.0);
            for i in i0..i1 {
                let li = row(lu, i);
                let xi = row(x, i);
                for k in i0..i {
                    let w = width(k + 1);
                    axpy(-li[k], row(x, k)[..w], xi[..w]);
                }
            }
            i0 = i1;
        }
        o0 = o1;
    }
}

// x = u^-1 x for the upper triangle of lu, blocked as forward_unit_with
// from the bottom up.
fn backward_with<G>(lu, x: matrix&) {
    let n = x.rows;
    let m = x.cols;
    var o1 = n;
    while o1 > 0 {
        let o0 = max(o1 - OB, 0);
        G(x, o0, 0, lu, o0, o1, false, x, o1, 0, o1 - o0, m, n - o1, -1.0);
        var j1 = o1;
        while j1 > o0 {
            let j0 = max(j1 - NB, o0);
            gemm_acc(x, j0, 0, lu, j0, j1, false, x, j1, 0, j1 - j0, m, o1 - j1, -1.0);
            for ii in j1 - j0 {
                let i = j1 - 1 - ii;
                let ui = row(lu, i);
                let xi = row(x, i);
                assert(ui.len == n);
                for k in i + 1..j1 { axpy(-ui[k], row(x, k), xi); }
                let d = ui[i];
                for &v in xi { v /= d; }
            }
            j1 = j0;
        }
        o1 = o0;
    }
}

// Solves a x = b for every column of b at once. The substitutions work on
// whole rows of right-hand sides: row i of x takes a multiple of each row
// solved before it, so every inner loop runs along a row.
fn solve_matrix_with<F, G>(a, b) -> matrix, bool {
    assert(a.rows == a.cols && a.rows == b.rows);
    let n = a.rows;
    let m = b.cols;
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = F(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    var x = zeros(n, m);
    for i in n { set_block(x, i, 0, submatrix(b, piv[i], piv[i] + 1, 0, m)); }
    forward_unit_with(lu, x, false, G);    // l y = p b
    backward_with(lu, x, G);               // u x = y
    return x, true;
}

// solve_matrix_with on this thread: lu_factor, and gemm_acc as the kernel.
fn solve_matrix(a, b) -> matrix, bool { return solve_matrix_with(a, b, lu_factor, gemm_acc); }

// With p a = l u, a^-1 = u^-1 l^-1 p. l^-1 is lower triangular, so the
// forward pass skips the columns it knows are zero: about 2n^3 flops in all
// where solving against the identity would take 8n^3/3.
fn inverse_with<F, G>(a) -> matrix, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = F(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    var x = identity(n);
    forward_unit_with(lu, x, true, G);     // x = l^-1
    backward_with(lu, x, G);               // x = u^-1 l^-1
    // Times p: column i of x is column piv[i] of the inverse.
    var inv = zeros(n, n);
    for r in n {
        let xr = row(x, r);
        let out = row(inv, r);
        assert(xr.len == n && out.len == n);
        for i in n { out[piv[i]] = xr[i]; }
    }
    return inv, true;
}

// inverse_with on this thread: lu_factor, and gemm_acc as the kernel.
fn inverse(a) -> matrix, bool { return inverse_with(a, lu_factor, gemm_acc); }

// ---------------------------------------------------------------------------
// Cholesky decomposition

// The lower-triangular l with l l^T = a, for symmetric positive definite a
// (only its lower triangle is read); false when a is not positive definite.
//
// Blocked twice, as LAPACK's potrf: within an outer block of OB columns,
// NB columns at a time, factor the diagonal block and the panel below it,
// then subtract the panel's outer product from the rest of the outer block
// with gemm_acc; then subtract the outer block's outer product from the
// whole trailing matrix with G.
fn cholesky_with<G>(a) -> matrix, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    var l = zeros(n, n);
    for i in n {
        let li = row(l, i);
        let ai = row(a, i);
        assert(li.len == n && ai.len == n);
        for j in i + 1 { li[j] = ai[j]; }
    }
    var ok = true;
    var o0 = 0;
    while o0 < n && ok {
        let o1 = min(o0 + OB, n);
        var k0 = o0;
        while k0 < o1 && ok {
            let k1 = min(k0 + NB, o1);
            // The panel, columns k0..k1. Entry (i, j) is the trailing value
            // less a dot product of rows i and j of l, from column k0 up to j.
            for j in k0..k1 {
                let lj = row(l, j);
                assert(lj.len == n);
                let head = lj[k0..j];
                let d = lj[j] - dot(head, head);
                if d <= 0.0 || d != d { ok = false; break; }
                let ljj = sqrt(d);
                lj[j] = ljj;
                for i in j + 1..n {
                    let li = row(l, i);
                    assert(li.len == n);
                    li[j] = (li[j] - dot(li[k0..j], head)) / ljj;
                }
            }
            // The rest of the outer block's columns: a -= l21 l21^T.
            if ok && k1 < o1 { cholesky_update(l, k0, k1, k1, o1, NB, gemm_acc); }
            k0 = k1;
        }
        // The whole trailing matrix, OB deep, in strips tall enough to share.
        if ok && o1 < n { cholesky_update(l, o0, o1, o1, n, OB, G); }
        o0 = o1;
    }
    for i in n {
        let li = row(l, i);
        for j in i + 1..n { li[j] = 0.0; }
    }
    return l, ok;
}

// Subtracts l[r.., k0..k1] l[r.., k0..k1]^T from the lower triangle of
// rows r.. and columns r..c1 of l, for rows r..l.rows, with kernel G.
// gemm_acc wants its right factor by rows, so the panel's transpose is
// copied out once. Strips of `strip` rows each stop at their last row's
// diagonal (or c1); cholesky_with clears the few entries above the diagonal
// they touch.
fn cholesky_update<G>(l: matrix&, k0: i64, k1: i64, r: i64, c1: i64, strip: i64) {
    let n = l.rows;
    let m = n - r;
    var lt = zeros(k1 - k0, m);
    for i in m {
        let li = row(l, r + i);
        for t in k1 - k0 { lt.data[t * m + i] = li[k0 + t]; }
    }
    var s0 = r;
    while s0 < n {
        let s1 = min(s0 + strip, n);
        G(l, s0, r, l, s0, k0, false, lt, 0, 0, s1 - s0, min(s1, c1) - r, k1 - k0, -1.0);
        s0 = s1;
    }
}

// cholesky_with on this thread: the kernel is gemm_acc.
fn cholesky(a) -> matrix, bool { return cholesky_with(a, gemm_acc); }

// Solves a x = b given a's Cholesky factor l.
fn cholesky_solve(l, b: f64[:]) -> f64[>..] {
    assert(l.rows == l.cols && l.rows == b.len);
    let n = b.len;
    // l y = b, then l^T x = y column by column: column i of l^T is row i of l,
    // so once x[i] is known it is subtracted from the y above it along that row.
    var x = solve_lower(l, b);
    for ii in n {
        let i = n - 1 - ii;
        let li = row(l, i);
        assert(li.len == n && x.len == n);
        let xi = x[i] / li[i];
        x[i] = xi;
        for k in i { x[k] -= li[k] * xi; }
    }
    return x;
}

// ---------------------------------------------------------------------------
// QR decomposition and least squares

// a = q r by Householder reflections, for any shape: q is rows x rows and
// orthogonal, r is rows x cols and upper triangular.
//
// Blocked with the compact WY form, as LAPACK's geqrf and orgqr: the NB
// reflectors H = I - tau v v^T of a panel multiply out to I - V T V^T, with
// V their vectors side by side and T upper triangular. The panel is
// factored one reflector at a time; the rest of r takes the whole block
// at once, as two gemm_acc calls. q is formed afterwards from the
// identity, blocks last first, each touching only the part of q that is
// no longer the identity.
fn qr(a, q: matrix&, r: matrix&) {
    let m = a.rows;
    let n = a.cols;
    r = clone(a);
    q = identity(m);
    let kmax = max(min(m - 1, n), 0);
    // Reflector k: unit vector in column k of vs, from row k down, and
    // tau[k] = 2 (0 where column k needed none). Each block's T sits in
    // rows k0..k1 of tt.
    var vs = zeros(m, kmax);
    var tau: f64[>..] = [];
    tau.resize(kmax, 0.0);
    var tt = zeros(kmax, NB);
    var v: f64[>..] = [];
    var w: f64[>..] = [];
    w.resize(n, 0.0);

    var k0 = 0;
    while k0 < kmax {
        let k1 = min(k0 + NB, kmax);
        let kb = k1 - k0;
        // The panel, one reflector at a time.
        for k in k0..k1 {
            // The reflection that takes r[k.., k] to a multiple of e_k.
            v.clear();
            for i in k..m { v.push(r.data[i * n + k]); }
            let alpha = if v[0] >= 0.0 { -vnorm(v) } else { vnorm(v) };
            v[0] -= alpha;
            let vn = vnorm(v);
            if vn == 0.0 { continue; }
            for &x in v { x /= vn; }
            tau[k] = 2.0;
            for i in k..m { vs.data[i * kmax + k] = v[i - k]; }
            // r[k.., k..k1] -= 2 v (v^T r[k.., k..k1]), both passes along rows.
            assert(w.len == n);
            for &x in w[k..k1] { x = 0.0; }
            for i in k..m {
                let ri = row(r, i);
                assert(ri.len == n);
                axpy(v[i - k], ri[k..k1], w[k..k1]);
            }
            for i in k..m {
                let ri = row(r, i);
                assert(ri.len == n);
                let vi = 2.0 * v[i - k];
                axpy(-vi, w[k..k1], ri[k..k1]);
            }
            for i in k + 1..m { r.data[i * n + k] = 0.0; }
        }
        householder_t(vs, tau, k0, k1, 0, tt);
        // The rest of r takes the block's transpose: c -= V (T^T (V^T c)).
        if k1 < n {
            let nc = n - k1;
            var wb = zeros(kb, nc);
            gemm_acc(wb, 0, 0, vs, k0, k0, true, r, k0, k1, kb, nc, m - k0, 1.0);
            apply_t(wb, tt, k0, kb, true);
            gemm_acc(r, k0, k1, vs, k0, k0, false, wb, 0, 0, m - k0, nc, kb, -1.0);
        }
        k0 = k1;
    }

    form_from_reflectors(q, vs, tau, 0, tt);
}

// The T of the block of reflectors k0..k1 into rows k0..k1 of tt, so that
// H_k0 ... H_k1-1 = I - V T V^T: column by column,
// T[..j, j] = -tau_j T[..j, ..j] (V[.., ..j]^T v_j). Reflector k is
// H_k = I - tau[k] v_k v_k^T with v_k column k of vs, zero above row k + off.
fn householder_t(vs: matrix&, tau: f64[:], k0: i64, k1: i64, off: i64, tt: matrix&) {
    let m = vs.rows;
    let kmax = vs.cols;
    var z: f64[NB] = [0.0; NB];
    for j in k1 - k0 {
        let kj = k0 + j;
        let tj = row(tt, kj);
        assert(tj.len == NB);
        for &x in tj { x = 0.0; }
        tj[j] = tau[kj];
        if tau[kj] == 0.0 { continue; }
        for l in j { z[l] = 0.0; }
        for i in kj + off..m {
            let vi = row(vs, i);
            assert(vi.len == kmax);
            let x = vi[kj];
            axpy(x, vi[k0..k0 + j], z[..j]);
        }
        for i in j {
            let ti = row(tt, k0 + i);
            assert(ti.len == NB);
            var acc = 0.0;
            for l in i..j { acc += ti[l] * z[l]; }
            ti[j] = -tau[kj] * acc;
        }
    }
}

// q = H_0 H_1 ... H_kmax-1 q, for q starting out as the first q.cols
// columns of the identity, with each block's T already in tt. Blocks go
// last first, and each multiplies only rows and columns b0 + off on: the
// rest of q is still the identity there, which the block leaves alone.
fn form_from_reflectors(q: matrix&, vs: matrix&, tau: f64[:], off: i64, tt: matrix&) {
    let m = q.rows;
    let kmax = vs.cols;
    var blocks = (kmax + NB - 1) / NB;
    while blocks > 0 {
        blocks--;
        let b0 = blocks * NB;
        let kb = min(b0 + NB, kmax) - b0;
        let r0 = b0 + off;
        let nc = q.cols - r0;
        if nc <= 0 { continue; }
        var wb = zeros(kb, nc);
        gemm_acc(wb, 0, 0, vs, r0, b0, true, q, r0, r0, kb, nc, m - r0, 1.0);
        apply_t(wb, tt, b0, kb, false);
        gemm_acc(q, r0, r0, vs, r0, b0, false, wb, 0, 0, m - r0, nc, kb, -1.0);
    }
}


// wb = T wb, or T^T wb with trans, in place, for the kb x kb upper
// triangular T in rows k0..k0 + kb of tt. Rows are combined in the order
// that reads each source row before it is overwritten.
fn apply_t(wb: matrix&, tt: matrix&, k0: i64, kb: i64, trans: bool) {
    fn t(i: i64, l: i64) -> f64 { tt.data[(k0 + i) * tt.cols + l] }
    if trans {
        // Row i of T^T wb is the sum over l <= i of T[l][i] wb[l]: bottom up.
        for ii in kb {
            let i = kb - 1 - ii;
            let wi = row(wb, i);
            let d = t(i, i);
            for &x in wi { x *= d; }
            for l in i { axpy(t(l, i), row(wb, l), wi); }
        }
    } else {
        // Row i of T wb is the sum over l >= i of T[i][l] wb[l]: top down.
        for i in kb {
            let wi = row(wb, i);
            let d = t(i, i);
            for &x in wi { x *= d; }
            for l in i + 1..kb { axpy(t(i, l), row(wb, l), wi); }
        }
    }
}

// The x minimizing |a x - b| for a with rows >= cols and full column rank;
// false when a is rank deficient.
fn lstsq(a, b: f64[:]) -> f64[>..], bool {
    assert(a.rows == b.len && a.rows >= a.cols);
    let n = a.cols;
    var q = zeros(0, 0);
    var r = zeros(0, 0);
    qr(a, q, r);
    var big = 0.0;
    for i in n { big = max(big, abs(r.data[i * n + i])); }
    let tol = big * EPS * max(a.rows, n);
    for i in n {
        guard abs(r.data[i * n + i]) > tol else { return [], false; }
    }
    // x = R1^-1 (Q^T b)[..n]
    var qtb: f64[>..] = [];
    for j in n {
        var s = 0.0;
        for i in a.rows { s += q.data[i * a.rows + j] * b[i]; }
        qtb.push(s);
    }
    return solve_upper(submatrix(r, 0, n, 0, n), qtb), true;
}

// ---------------------------------------------------------------------------
// Symmetric eigenproblem

// Swaps rows p and q of m.
fn swap_rows(m: matrix&, p: i64, q: i64) {
    let rp = row(m, p);
    let rq = row(m, q);
    assert(rq.len == rp.len);
    for k in rp.len {
        let t = rp[k];
        rp[k] = rq[k];
        rq[k] = t;
    }
}

// Rows p and q of m become c p - s q and s p + c q: the rotation every
// Jacobi step applies, along two contiguous rows.
fn rotate_rows(m: matrix&, p: i64, q: i64, c: f64, s: f64) {
    let rp = row(m, p);
    let rq = row(m, q);
    assert(rq.len == rp.len);
    for k in rp.len {
        let x = rp[k];
        let y = rq[k];
        rp[k] = c * x - s * y;
        rq[k] = s * x + c * y;
    }
}

// sqrt(a^2 + b^2) without overflow or underflow on the way.
fn hypot2(a: f64, b: f64) -> f64 {
    let x = abs(a);
    let y = abs(b);
    let big = max(x, y);
    guard big > 0.0 else { return 0.0; }
    let small = min(x, y) / big;
    return big * sqrt(1.0 + small * small);
}

// Eigenvalues and eigenvectors of symmetric a: a = vectors diag(values)
// vectors^T with values ascending and vectors orthogonal, column j belonging
// to values[j]. Reads the upper triangle of a. Returns false if the QL
// iteration did not converge.
//
// Householder reduction to tridiagonal form, then the implicit QL method
// with Wilkinson shifts: EISPACK's tred2 and tql2, by way of JAMA. Both
// classically walk the columns of the working matrix; here it is kept
// transposed, w[c][r] for their V[r][c], so the walks run along rows.
fn eigh(a, values: f64[>..]&, vectors: matrix&) -> bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    values.clear();
    guard n > 0 else {
        vectors = zeros(0, 0);
        return true;
    }
    var w = clone(a);
    var e: f64[>..] = [];
    e.resize(n, 0.0);
    values.resize(n, 0.0);
    let d = values[..];

    // tred2: reduce to tridiagonal form, d the diagonal and e the
    // subdiagonal, accumulating the transformation in w.
    for j in n { d[j] = w.data[j * n + n - 1]; }
    for ii in n - 1 {
        let i = n - 1 - ii;
        // Scale to avoid under- and overflow.
        var sc = 0.0;
        var h = 0.0;
        for k in i { sc += abs(d[k]); }
        if sc == 0.0 {
            e[i] = d[i - 1];
            for j in i {
                d[j] = w.data[j * n + i - 1];
                w.data[j * n + i] = 0.0;
                w.data[i * n + j] = 0.0;
            }
        } else {
            // The Householder vector.
            for k in i { d[k] /= sc; }
            h = dot(d[..i], d[..i]);
            var f = d[i - 1];
            var g = sqrt(h);
            if f > 0.0 { g = -g; }
            e[i] = sc * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in i { e[j] = 0.0; }
            // The similarity transformation on the remaining columns.
            for j in i {
                let wj = row(w, j);
                assert(wj.len == n);
                f = d[j];
                w.data[i * n + j] = f;
                e[j] += wj[j] * f + dot(wj[j + 1..i], d[j + 1..i]);
                axpy(f, wj[j + 1..i], e[j + 1..i]);
            }
            f = 0.0;
            for j in i { e[j] /= h; }
            f = dot(e[..i], d[..i]);
            let hh = f / (h + h);
            axpy(-hh, d[..i], e[..i]);
            for j in i {
                let wj = row(w, j);
                assert(wj.len == n);
                f = d[j];
                g = e[j];
                for k in j..i { wj[k] -= f * e[k] + g * d[k]; }
                d[j] = wj[i - 1];
                wj[i] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate the transformations: q = H_n-2 ... H_0, where
    // H_i = I - u_i u_i^T / h_i acts on coordinates 0..=i, u_i is row i + 1
    // of w over columns 0..=i, and h_i = d[i + 1]. With the coordinates
    // reversed, c to n - 1 - c, that is G_0 ... G_n-2 with G_k = H_n-2-k
    // acting from coordinate k + 1 on: form_from_reflectors' shape, blocked.
    // The tridiagonal's diagonal is w's own.
    let nr = n - 1;
    var vs = zeros(n, nr);
    var tau: f64[>..] = [];
    tau.resize(nr, 0.0);
    for k in nr {
        let i = n - 2 - k;
        let h = d[i + 1];
        if h == 0.0 { continue; }
        tau[k] = 1.0 / h;
        let ui = row(w, i + 1);
        for c in i + 1 { vs.data[(n - 1 - c) * nr + k] = ui[c]; }
    }
    for j in n { d[j] = w.data[j * n + j]; }
    var tt = zeros(nr, NB);
    var k0 = 0;
    while k0 < nr {
        householder_t(vs, tau, k0, min(k0 + NB, nr), 1, tt);
        k0 += NB;
    }
    var qrev = identity(n);
    form_from_reflectors(qrev, vs, tau, 1, tt);
    // w = q^T, back in the original coordinates: w[a][b] = q[b][a].
    for a in n {
        let wa = row(w, a);
        assert(wa.len == n);
        for b in n { wa[b] = qrev.data[(n - 1 - b) * n + n - 1 - a]; }
    }
    e[0] = 0.0;

    // tql2: diagonalize the tridiagonal matrix by implicit QL steps.
    for i in 1..n { e[i - 1] = e[i]; }
    e[n - 1] = 0.0;
    var f = 0.0;
    var tst1 = 0.0;
    var converged = true;
    for l in n {
        // The first negligible subdiagonal element at or after l.
        tst1 = max(tst1, abs(d[l]) + abs(e[l]));
        var m = l;
        while m < n {
            if abs(e[m]) <= EPS * tst1 { break; }
            m++;
        }
        // m == l: d[l] is an eigenvalue. Otherwise iterate.
        if m > l {
            var iter = 0;
            loop {
                iter++;
                if iter > 60 { converged = false; break; }
                // The implicit shift.
                var g = d[l];
                var p = (d[l + 1] - g) / (2.0 * e[l]);
                var r = hypot2(p, 1.0);
                if p < 0.0 { r = -r; }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                var h = g - d[l];
                for i in l + 2..n { d[i] -= h; }
                f += h;
                // The implicit QL transformation.
                p = d[m];
                var c = 1.0;
                var c2 = c;
                var c3 = c;
                let el1 = e[l + 1];
                var s = 0.0;
                var s2 = 0.0;
                var i = m - 1;
                while i >= l {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot2(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    // Accumulate: columns i and i + 1 of V are rows of w.
                    rotate_rows(w, i, i + 1, c, s);
                    i--;
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if abs(e[l]) <= EPS * tst1 { break; }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }

    // Ascending, moving the vectors with their values.
    for i in n {
        var best = i;
        for j in i + 1..n { if d[j] < d[best] { best = j; } }
        if best != i {
            let t = d[i];
            d[i] = d[best];
            d[best] = t;
            swap_rows(w, i, best);
        }
    }
    vectors = transpose(w);
    return converged;
}

fn eigvals_sym(a) -> f64[>..] {
    var values: f64[>..] = [];
    var vectors = zeros(0, 0);
    eigh(a, values, vectors);
    return values;
}

// ---------------------------------------------------------------------------
// Singular value decomposition

// The thin SVD of a rows x cols matrix, k = min(rows, cols):
// a = u diag(s) v^T with u rows x k and v cols x k, both with orthonormal
// columns, and s descending and non-negative. Returns false if the QR
// iteration did not converge.
fn svd(a, u: matrix&, s: f64[>..]&, v: matrix&) -> bool {
    if a.rows >= a.cols { return svd_tall(a, u, s, v); }
    return svd_tall(transpose(a), v, s, u);     // a^T = v diag(s) u^T
}

// svd for rows >= cols, by Golub-Kahan-Reinsch: Householder reflections
// reduce a to bidiagonal form, then implicit-shift QR steps diagonalize the
// bidiagonal. This is LINPACK's dsvdc by way of JAMA. The classical
// algorithm walks the columns of a, u and v; here all three are kept
// transposed, so every walk is along a row.
fn svd_tall(a, u: matrix&, s: f64[>..]&, v: matrix&) -> bool {
    assert(a.rows >= a.cols);
    let m = a.rows;
    let n = a.cols;
    s.clear();
    guard n > 0 else {
        u = zeros(m, 0);
        v = zeros(0, 0);
        return true;
    }
    var at = transpose(a);      // row j is column j of a
    var ut = zeros(n, m);       // row j is column j of u
    var vt = zeros(n, n);       // row j is column j of v
    s.resize(n, 0.0);
    var e: f64[>..] = [];
    e.resize(n, 0.0);
    var work: f64[>..] = [];
    work.resize(m, 0.0);

    // Bidiagonalize: the diagonal into s, the superdiagonal into e.
    let nct = min(m - 1, n);
    let nrt = max(0, min(n - 2, m));
    for k in max(nct, nrt) {
        let ak = row(at, k);
        assert(ak.len == m);
        if k < nct {
            // The reflection for column k; its diagonal entry into s[k].
            var sk = vnorm(ak[k..]);
            if sk != 0.0 {
                if ak[k] < 0.0 { sk = -sk; }
                for i in k..m { ak[i] /= sk; }
                ak[k] += 1.0;
            }
            s[k] = -sk;
        }
        for j in k + 1..n {
            let aj = row(at, j);
            assert(aj.len == m);
            if k < nct && s[k] != 0.0 {
                axpy(-dot(ak[k..], aj[k..]) / ak[k], ak[k..], aj[k..]);
            }
            // Row k of a, for the row reflection.
            e[j] = aj[k];
        }
        if k < nct {
            let uk = row(ut, k);
            for i in k..m { uk[i] = ak[i]; }
        }
        if k < nrt {
            // The reflection for row k; its superdiagonal entry into e[k].
            var ek = vnorm(e[k + 1..]);
            if ek != 0.0 {
                if e[k + 1] < 0.0 { ek = -ek; }
                for i in k + 1..n { e[i] /= ek; }
                e[k + 1] += 1.0;
            }
            e[k] = -ek;
            if k + 1 < m && e[k] != 0.0 {
                let wk = work[k + 1..];
                for &x in wk { x = 0.0; }
                for j in k + 1..n { axpy(e[j], row(at, j)[k + 1..], wk); }
                for j in k + 1..n { axpy(-e[j] / e[k + 1], wk, row(at, j)[k + 1..]); }
            }
            let vk = row(vt, k);
            for i in k + 1..n { vk[i] = e[i]; }
        }
    }

    // The bidiagonal matrix, of order p.
    var p = min(n, m + 1);
    if nct < n { s[nct] = at.data[nct * m + nct]; }
    if m < p { s[p - 1] = 0.0; }
    if nrt + 1 < p { e[nrt] = at.data[(p - 1) * m + nrt]; }
    e[p - 1] = 0.0;

    // Generate u and v from their reflectors, blocked as qr's q. Reflector
    // k of u is H_k = I - u_k u_k^T / u_k[k], u_k row k of ut from column k
    // on, where s[k] != 0; v's is G_k = I - v_k v_k^T / v_k[k + 1], v_k row k
    // of vt from column k + 1 on, where k < nrt and e[k] != 0.
    {
        var vs = zeros(m, nct);
        var tau: f64[>..] = [];
        tau.resize(nct, 0.0);
        for k in nct {
            let uk = row(ut, k);
            assert(uk.len == m);
            if s[k] == 0.0 { continue; }
            tau[k] = 1.0 / uk[k];
            for i in k..m { vs.data[i * nct + k] = uk[i]; }
        }
        var tt = zeros(nct, NB);
        var k0 = 0;
        while k0 < nct {
            householder_t(vs, tau, k0, min(k0 + NB, nct), 0, tt);
            k0 += NB;
        }
        var um = zeros(m, n);
        for i in n { um.data[i * n + i] = 1.0; }
        form_from_reflectors(um, vs, tau, 0, tt);
        ut = transpose(um);
    }
    {
        var vs = zeros(n, nrt);
        var tau: f64[>..] = [];
        tau.resize(nrt, 0.0);
        for k in nrt {
            let vk = row(vt, k);
            assert(vk.len == n);
            if e[k] == 0.0 { continue; }
            tau[k] = 1.0 / vk[k + 1];
            for i in k + 1..n { vs.data[i * nrt + k] = vk[i]; }
        }
        var tt = zeros(nrt, NB);
        var k0 = 0;
        while k0 < nrt {
            householder_t(vs, tau, k0, min(k0 + NB, nrt), 1, tt);
            k0 += NB;
        }
        var vm = identity(n);
        form_from_reflectors(vm, vs, tau, 1, tt);
        vt = transpose(vm);
    }

    // Diagonalize the bidiagonal. Rotations of columns j and q of u or v
    // are rotations of rows of ut or vt: x' = c x + s y, y' = -s x + c y
    // is rotate_rows with -s.
    let pp = p - 1;
    var iter = 0;
    let tiny = pow(2.0, -966.0);
    var converged = true;
    while p > 0 {
        if iter > 75 { converged = false; break; }
        // k: the last negligible superdiagonal entry before p - 1, or -1.
        var k = p - 2;
        while k >= 0 {
            if abs(e[k]) <= tiny + EPS * (abs(s[k]) + abs(s[k + 1])) {
                e[k] = 0.0;
                break;
            }
            k--;
        }
        // 1: s[p - 1] and e[k] negligible, k < p: deflate s[p - 1].
        // 2: s[k] negligible, k < p: split at k.
        // 3: e[k] negligible, k < p, s[k..p] not: a QR step.
        // 4: e[p - 2] negligible: s[p - 1] has converged.
        var kase = 0;
        if k == p - 2 {
            kase = 4;
        } else {
            var ks = p - 1;
            while ks > k {
                let t = abs(e[ks]) + (if ks != k + 1 { abs(e[ks - 1]) } else { 0.0 });
                if abs(s[ks]) <= tiny + EPS * t {
                    s[ks] = 0.0;
                    break;
                }
                ks--;
            }
            if ks == k {
                kase = 3;
            } else if ks == p - 1 {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        k++;

        if kase == 1 {
            var f = e[p - 2];
            e[p - 2] = 0.0;
            var j = p - 2;
            while j >= k {
                let t = hypot2(s[j], f);
                let cs = s[j] / t;
                let sn = f / t;
                s[j] = t;
                if j != k {
                    f = -sn * e[j - 1];
                    e[j - 1] = cs * e[j - 1];
                }
                rotate_rows(vt, j, p - 1, cs, -sn);
                j--;
            }
        } else if kase == 2 {
            var f = e[k - 1];
            e[k - 1] = 0.0;
            for j in k..p {
                let t = hypot2(s[j], f);
                let cs = s[j] / t;
                let sn = f / t;
                s[j] = t;
                f = -sn * e[j];
                e[j] = cs * e[j];
                rotate_rows(ut, j, k - 1, cs, -sn);
            }
        } else if kase == 3 {
            // The shift, from the trailing 2x2 of b^T b.
            let sc = max(max(max(max(abs(s[p - 1]), abs(s[p - 2])), abs(e[p - 2])), abs(s[k])), abs(e[k]));
            let sp = s[p - 1] / sc;
            let spm1 = s[p - 2] / sc;
            let epm1 = e[p - 2] / sc;
            let sk = s[k] / sc;
            let ek = e[k] / sc;
            let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
            let c = (sp * epm1) * (sp * epm1);
            var shift = 0.0;
            if b != 0.0 || c != 0.0 {
                shift = sqrt(b * b + c);
                if b < 0.0 { shift = -shift; }
                shift = c / (b + shift);
            }
            var f = (sk + sp) * (sk - sp) + shift;
            var g = sk * ek;
            // Chase the bulge down the bidiagonal.
            for j in k..p - 1 {
                var t = hypot2(f, g);
                var cs = f / t;
                var sn = g / t;
                if j != k { e[j - 1] = t; }
                f = cs * s[j] + sn * e[j];
                e[j] = cs * e[j] - sn * s[j];
                g = sn * s[j + 1];
                s[j + 1] = cs * s[j + 1];
                rotate_rows(vt, j, j + 1, cs, -sn);
                t = hypot2(f, g);
                cs = f / t;
                sn = g / t;
                s[j] = t;
                f = cs * e[j] + sn * s[j + 1];
                s[j + 1] = -sn * e[j] + cs * s[j + 1];
                g = sn * e[j + 1];
                e[j + 1] = cs * e[j + 1];
                if j < m - 1 { rotate_rows(ut, j, j + 1, cs, -sn); }
            }
            e[p - 2] = f;
            iter++;
        } else {
            // Converged: make s[k] non-negative, then move it into order.
            if s[k] <= 0.0 {
                s[k] = if s[k] < 0.0 { -s[k] } else { 0.0 };
                let vk = row(vt, k);
                for i in pp + 1 { vk[i] = -vk[i]; }
            }
            while k < pp {
                if s[k] >= s[k + 1] { break; }
                let t = s[k];
                s[k] = s[k + 1];
                s[k + 1] = t;
                if k < n - 1 { swap_rows(vt, k, k + 1); }
                if k < m - 1 { swap_rows(ut, k, k + 1); }
                k++;
            }
            iter = 0;
            p--;
        }
    }
    u = transpose(ut);
    v = transpose(vt);
    return converged;
}


fn singular_values(a) -> f64[>..] {
    var u = zeros(0, 0);
    var v = zeros(0, 0);
    var s: f64[>..] = [];
    svd(a, u, s, v);
    return s;
}

// The spectral norm: the largest singular value.
fn norm2(a) -> f64 {
    let s = singular_values(a);
    return if s.len > 0 { s[0] } else { 0.0 };
}

// The number of singular values above tol; by default
// max(rows, cols) * eps * the largest.
fn rank(a, tol: f64 = -1.0) -> i64 {
    let s = singular_values(a);
    guard s.len > 0 else { return 0; }
    let t = if tol >= 0.0 { tol } else { max(a.rows, a.cols) * EPS * s[0] };
    var r = 0;
    for x in s { if x > t { r++; } }
    return r;
}

// The 2-norm condition number; infinite for a singular matrix.
fn cond(a) -> f64 {
    let s = singular_values(a);
    guard s.len > 0 && s[s.len - 1] > 0.0 else { return 1.0 / 0.0; }
    return s[0] / s[s.len - 1];
}

// The Moore-Penrose pseudoinverse, v diag(1/s) u^T, dropping the singular
// values at or below tol (default as for rank).
fn pinv(a, tol: f64 = -1.0) -> matrix {
    var u = zeros(0, 0);
    var v = zeros(0, 0);
    var s: f64[>..] = [];
    svd(a, u, s, v);
    var p = zeros(a.cols, a.rows);
    guard s.len > 0 else { return p; }
    let t = if tol >= 0.0 { tol } else { max(a.rows, a.cols) * EPS * s[0] };
    // p = sum over j of v_j u_j^T / s_j: with u and v transposed, every
    // update is a scaled row of u^T added to a row of p.
    let ut = transpose(u);
    let vt = transpose(v);
    for j in s.len {
        if s[j] <= t { continue; }
        let inv = 1.0 / s[j];
        let uj = row(ut, j);
        let vj = row(vt, j);
        for r in a.cols { axpy(vj[r] * inv, uj, row(p, r)); }
    }
    return p;
}
