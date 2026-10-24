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

// a * b, in i-k-j order so the inner loop runs along rows of both. Four rows
// of c and four rows of b at a time: each element of b loaded feeds four
// rows of c, and each pass over a row of c folds in four rows of b.
fn matmul(a, b) -> matrix {
    assert(a.cols == b.rows);
    let n = b.cols;
    let kk = a.cols;
    var c = zeros(a.rows, n);
    var i = 0;
    while i + 4 <= a.rows {
        let c0 = row(c, i);
        let c1 = row(c, i + 1);
        let c2 = row(c, i + 2);
        let c3 = row(c, i + 3);
        let a0 = row(a, i);
        let a1 = row(a, i + 1);
        let a2 = row(a, i + 2);
        let a3 = row(a, i + 3);
        assert(c0.len == n && c1.len == n && c2.len == n && c3.len == n);
        assert(a0.len == kk && a1.len == kk && a2.len == kk && a3.len == kk);
        var k = 0;
        while k + 4 <= kk {
            let b0 = row(b, k);
            let b1 = row(b, k + 1);
            let b2 = row(b, k + 2);
            let b3 = row(b, k + 3);
            assert(b0.len == n && b1.len == n && b2.len == n && b3.len == n);
            let p0 = a0[k];
            let p1 = a0[k + 1];
            let p2 = a0[k + 2];
            let p3 = a0[k + 3];
            let q0 = a1[k];
            let q1 = a1[k + 1];
            let q2 = a1[k + 2];
            let q3 = a1[k + 3];
            let r0 = a2[k];
            let r1 = a2[k + 1];
            let r2 = a2[k + 2];
            let r3 = a2[k + 3];
            let s0 = a3[k];
            let s1 = a3[k + 1];
            let s2 = a3[k + 2];
            let s3 = a3[k + 3];
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
            k += 4;
        }
        while k < kk {
            let bk = row(b, k);
            assert(bk.len == n);
            let p = a0[k];
            let q = a1[k];
            let r = a2[k];
            let s = a3[k];
            for j in n {
                let v = bk[j];
                c0[j] += p * v;
                c1[j] += q * v;
                c2[j] += r * v;
                c3[j] += s * v;
            }
            k++;
        }
        i += 4;
    }
    // The last rows, one at a time.
    while i < a.rows {
        let ci = row(c, i);
        let ai = row(a, i);
        assert(ci.len == n && ai.len == kk);
        for k in kk { axpy(ai[k], row(b, k), ci); }
        i++;
    }
    return c;
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

// Factors square `a` in place into P a = L U: U on and above the diagonal,
// L's multipliers below it (L's diagonal is ones), and row i of P a is row
// piv[i] of the original. Returns the permutation's sign, and false when a
// is singular (a zero pivot column); the factors are then incomplete.
fn lu_factor(a: matrix&, piv: i64[>..]&) -> f64, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    piv.clear();
    for i in n { piv.push(i); }
    var sign = 1.0;
    var ok = true;
    for k in n {
        var p = k;
        var big = abs(a.data[k * n + k]);
        for i in k + 1..n {
            let v = abs(a.data[i * n + k]);
            if v > big { big = v; p = i; }
        }
        if big == 0.0 { ok = false; continue; }
        // Row slices with their lengths asserted: the inner loops then run
        // without bounds checks, and the C compiler can vectorize them.
        let rk = a.data[k * n..(k + 1) * n];
        assert(rk.len == n);
        if p != k {
            let rp = a.data[p * n..(p + 1) * n];
            assert(rp.len == n);
            for j in n {
                let t = rk[j];
                rk[j] = rp[j];
                rp[j] = t;
            }
            let t = piv[k];
            piv[k] = piv[p];
            piv[p] = t;
            sign = -sign;
        }
        let pivot = rk[k];
        for i in k + 1..n {
            let ri = a.data[i * n..(i + 1) * n];
            assert(ri.len == n);
            let f = ri[k] / pivot;
            ri[k] = f;
            for j in k + 1..n { ri[j] -= f * rk[j]; }
        }
    }
    return sign, ok;
}

// Solves a x = b given lu_factor's output for a.
fn lu_solve(lu, piv: i64[:], b: f64[:]) -> f64[>..] {
    assert(lu.rows == lu.cols && lu.rows == b.len && piv.len == b.len);
    var pb: f64[>..] = [];
    for p in piv { pb.push(b[p]); }
    let y = solve_lower(lu, pb, true);
    return solve_upper(lu, y);
}

fn det(a) -> f64 {
    assert(a.rows == a.cols);
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor(lu, piv);
    guard ok else { return 0.0; }
    var d = sign;
    for i in a.rows { d *= lu.data[i * a.rows + i]; }
    return d;
}

// Solves a x = b for square a; false when a is singular.
fn solve(a, b: f64[:]) -> f64[>..], bool {
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor(lu, piv);
    guard ok else { return [], false; }
    return lu_solve(lu, piv, b), true;
}

// Solves a x = b for every column of b at once. The substitutions work on
// whole rows of right-hand sides: row i of x takes a multiple of each row
// solved before it, so every inner loop runs along a row.
fn solve_matrix(a, b) -> matrix, bool {
    assert(a.rows == a.cols && a.rows == b.rows);
    let n = a.rows;
    let m = b.cols;
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    var x = zeros(n, m);
    for i in n { set_block(x, i, 0, submatrix(b, piv[i], piv[i] + 1, 0, m)); }
    // l y = p b: l has a unit diagonal.
    for i in n {
        let li = row(lu, i);
        let xi = row(x, i);
        for k in i { axpy(-li[k], row(x, k), xi); }
    }
    // u x = y, from the bottom row up.
    for ii in n {
        let i = n - 1 - ii;
        let ui = row(lu, i);
        let xi = row(x, i);
        assert(ui.len == n);
        for k in i + 1..n { axpy(-ui[k], row(x, k), xi); }
        let d = ui[i];
        for &v in xi { v /= d; }
    }
    return x, true;
}

fn inverse(a) -> matrix, bool {
    assert(a.rows == a.cols);
    return solve_matrix(a, identity(a.rows));
}

// ---------------------------------------------------------------------------
// Cholesky decomposition

// The lower-triangular l with l l^T = a, for symmetric positive definite a
// (only its lower triangle is read); false when a is not positive definite.
fn cholesky(a) -> matrix, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    var l = zeros(n, n);
    var ok = true;
    for j in n {
        // Entry (i, j) is a dot product of rows i and j of l, up to column j.
        let lj = l.data[j * n..(j + 1) * n];
        assert(lj.len == n);
        let head = lj[..j];
        let d = a.data[j * n + j] - dot(head, head);
        if d <= 0.0 || d != d { ok = false; break; }
        let ljj = sqrt(d);
        lj[j] = ljj;
        for i in j + 1..n {
            let li = l.data[i * n..(i + 1) * n];
            assert(li.len == n);
            li[j] = (a.data[i * n + j] - dot(li[..j], head)) / ljj;
        }
    }
    return l, ok;
}

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
fn qr(a, q: matrix&, r: matrix&) {
    let m = a.rows;
    let n = a.cols;
    r = clone(a);
    q = identity(m);
    var v: f64[>..] = [];
    var w: f64[>..] = [];
    w.resize(n, 0.0);
    for k in min(m - 1, n) {
        // The reflection that takes r[k.., k] to a multiple of e_k.
        v.clear();
        for i in k..m { v.push(r.data[i * n + k]); }
        let alpha = if v[0] >= 0.0 { -vnorm(v) } else { vnorm(v) };
        v[0] -= alpha;
        let vn = vnorm(v);
        if vn == 0.0 { continue; }
        for &x in v { x /= vn; }
        // r[k.., ..] -= 2 v w^T with w^T = v^T r[k.., ..], both passes along rows.
        assert(w.len == n);
        for j in k..n { w[j] = 0.0; }
        for i in k..m {
            let ri = row(r, i);
            assert(ri.len == n);
            let vi = v[i - k];
            for j in k..n { w[j] += vi * ri[j]; }
        }
        for i in k..m {
            let ri = row(r, i);
            assert(ri.len == n);
            let vi = 2.0 * v[i - k];
            for j in k..n { ri[j] -= vi * w[j]; }
        }
        // q[.., k..] -= 2 (q[.., k..] v) v^T, accumulating q = H_1 H_2 ...
        for i in m {
            let qi = q.data[i * m + k..(i + 1) * m];
            axpy(-2.0 * dot(qi, v), v, qi);
        }
        for i in k + 1..m { r.data[i * n + k] = 0.0; }
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
    // Accumulate the transformations.
    for i in n - 1 {
        let wi = row(w, i);
        let wi1 = row(w, i + 1);
        assert(wi.len == n && wi1.len == n);
        wi[n - 1] = wi[i];
        wi[i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in i + 1 { d[k] = wi1[k] / h; }
            for j in i + 1 {
                let wj = row(w, j);
                axpy(-dot(wi1[..i + 1], wj[..i + 1]), d[..i + 1], wj[..i + 1]);
            }
        }
        for k in i + 1 { wi1[k] = 0.0; }
    }
    for j in n {
        d[j] = w.data[j * n + n - 1];
        w.data[j * n + n - 1] = 0.0;
    }
    w.data[(n - 1) * n + n - 1] = 1.0;
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

// Makes the rows of m from `from` on orthonormal to all the rows before
// them, by Gram-Schmidt on unit vectors.
fn complete_rows(m: matrix&, from: i64) {
    let len = m.cols;
    var e = 0;
    for j in from..m.rows {
        let rj = row(m, j);
        loop {
            assert(e < len);
            for x, i in rj { rj[i] = if i == e { 1.0 } else { 0.0 }; }
            e++;
            for pass in 2 {
                for k in j {
                    let rk = row(m, k);
                    axpy(-dot(rk, rj), rk, rj);
                }
            }
            let nrm = vnorm(rj);
            if nrm > 0.5 {
                for &x in rj { x /= nrm; }
                break;
            }
        }
    }
}

// The thin SVD of a rows x cols matrix, k = min(rows, cols):
// a = u diag(s) v^T with u rows x k and v cols x k, both with orthonormal
// columns, and s descending and non-negative. One-sided Jacobi (Hestenes),
// accurate to working precision in the small singular values too. Returns
// false if it did not converge.
fn svd(a, u: matrix&, s: f64[>..]&, v: matrix&) -> bool {
    if a.rows >= a.cols { return svd_tall(a, u, s, v); }
    return svd_tall(transpose(a), v, s, u);     // a^T = v diag(s) u^T
}

// x.x, y.y and x.y in one pass, with dot's lanes.
fn gram2(x: f64[:], y: f64[:]) -> f64, f64, f64 {
    let n = x.len;
    assert(y.len == n);
    var xx: f64[LANES] = [0.0; LANES];
    var yy: f64[LANES] = [0.0; LANES];
    var xy: f64[LANES] = [0.0; LANES];
    let m = n - n % LANES;
    var i = 0;
    while i < m {
        let xs = x[i..i + LANES];
        let ys = y[i..i + LANES];
        assert(xs.len == LANES && ys.len == LANES);
        for k in LANES {
            let a = xs[k];
            let b = ys[k];
            xx[k] += a * a;
            yy[k] += b * b;
            xy[k] += a * b;
        }
        i += LANES;
    }
    var sxx = 0.0;
    var syy = 0.0;
    var sxy = 0.0;
    for k in LANES {
        sxx += xx[k];
        syy += yy[k];
        sxy += xy[k];
    }
    for j in m..n {
        sxx += x[j] * x[j];
        syy += y[j] * y[j];
        sxy += x[j] * y[j];
    }
    return sxx, syy, sxy;
}

// svd for rows >= cols. One-sided Jacobi rotates pairs of columns of a until
// they are orthogonal; it works on a^T here, so each column is a contiguous
// row, and so does v.
fn svd_tall(a, u: matrix&, s: f64[>..]&, v: matrix&) -> bool {
    assert(a.rows >= a.cols);
    let m = a.rows;
    let n = a.cols;
    var ut = transpose(a);
    var vt = identity(n);
    var converged = false;
    for sweep in 100 {
        var rotated = false;
        for p in n {
            for q in p + 1..n {
                let alpha, beta, gamma = gram2(row(ut, p), row(ut, q));
                if abs(gamma) <= EPS * sqrt(alpha * beta) { continue; }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                var t = 1.0 / (abs(zeta) + sqrt(1.0 + zeta * zeta));
                if zeta < 0.0 { t = -t; }
                let c = 1.0 / sqrt(1.0 + t * t);
                let sn = c * t;
                rotate_rows(ut, p, q, c, sn);
                rotate_rows(vt, p, q, c, sn);
            }
        }
        if !rotated { converged = true; break; }
    }
    // The singular values are the row norms; sort, then normalize.
    s.clear();
    for j in n { s.push(vnorm(row(ut, j))); }
    for i in n {
        var best = i;
        for j in i + 1..n { if s[j] > s[best] { best = j; } }
        if best != i {
            let t = s[i];
            s[i] = s[best];
            s[best] = t;
            swap_rows(ut, i, best);
            swap_rows(vt, i, best);
        }
    }
    let tiny = if n > 0 { s[0] * EPS * m } else { 0.0 };
    var nonzero = 0;
    for j in n {
        if s[j] > tiny && s[j] > 0.0 {
            for &x in row(ut, j) { x /= s[j]; }
            nonzero = j + 1;
        }
    }
    // Rows for zero singular values: any orthonormal completion.
    complete_rows(ut, nonzero);
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
