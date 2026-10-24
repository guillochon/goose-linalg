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

// a * b, in i-k-j order so the inner loop runs along rows of both.
fn matmul(a, b) -> matrix {
    assert(a.cols == b.rows);
    let n = b.cols;
    var c = zeros(a.rows, n);
    for i in a.rows {
        let crow = c.data[i * n..(i + 1) * n];
        for k in a.cols {
            let x = a.data[i * a.cols + k];
            let brow = b.data[k * n..(k + 1) * n];
            assert(crow.len == n && brow.len == n);
            for j in n { crow[j] += x * brow[j]; }
        }
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

fn dot(x: f64[:], y: f64[:]) -> f64 {
    assert(x.len == y.len);
    var s = 0.0;
    for v, i in x { s += v * y[i]; }
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
        var s = b[i];
        for k in i { s -= li[k] * x[k]; }
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
        var s = b[i];
        for k in i + 1..n { s -= ui[k] * x[k]; }
        x[i] = s / ui[i];
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

// Solves a x = b for every column of b at once.
fn solve_matrix(a, b) -> matrix, bool {
    assert(a.rows == b.rows);
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    var x = zeros(a.cols, b.cols);
    for j in b.cols {
        let xj = lu_solve(lu, piv, col(b, j));
        for i in xj.len { x.data[i * x.cols + j] = xj[i]; }
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
        var d = a.data[j * n + j];
        for k in j { d -= lj[k] * lj[k]; }
        if d <= 0.0 || d != d { ok = false; break; }
        let ljj = sqrt(d);
        lj[j] = ljj;
        for i in j + 1..n {
            let li = l.data[i * n..(i + 1) * n];
            assert(li.len == n);
            var s = a.data[i * n + j];
            for k in j { s -= li[k] * lj[k]; }
            li[j] = s / ljj;
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
            assert(qi.len == v.len);
            var s = 0.0;
            for t in qi.len { s += qi[t] * v[t]; }
            s *= 2.0;
            for t in qi.len { qi[t] -= s * v[t]; }
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

// Eigenvalues and eigenvectors of symmetric a, by cyclic Jacobi rotations:
// a = vectors diag(values) vectors^T with values ascending and vectors
// orthogonal, column j belonging to values[j]. Reads the whole of a, which
// should be symmetric. Returns false if it did not converge.
fn eigh(a, values: f64[>..]&, vectors: matrix&) -> bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    var w = clone(a);
    // The eigenvectors are kept as rows, so their rotations run along rows
    // too, and transposed once at the end.
    var vt = identity(n);
    let size = max(norm_fro(a), 1.0e-300);
    var converged = false;
    for sweep in 100 {
        var off = 0.0;
        for i in n {
            // Summed directly: all minus diagonal would cancel catastrophically
            // just when the diagonal dominates, and stop the sweeps too early.
            let wi = row(w, i);
            for x, j in wi { if j != i { off += x * x; } }
        }
        // Off-diagonal mass at rounding level is as converged as it gets.
        if sqrt(off) <= EPS * size { converged = true; break; }
        for p in n {
            for q in p + 1..n {
                let apq = w.data[p * n + q];
                if abs(apq) <= EPS * 1.0e-3 * size {
                    w.data[p * n + q] = 0.0;
                    w.data[q * n + p] = 0.0;
                    continue;
                }
                let app = w.data[p * n + p];
                let aqq = w.data[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                var t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0));
                if theta < 0.0 { t = -t; }
                let c = 1.0 / sqrt(t * t + 1.0);
                let s = t * c;
                // w = J^T w J. Outside the (p, q) block, rows p and q of the
                // result are rows p and q of J^T w, rotated along the rows;
                // columns p and q follow by symmetry.
                let wp = row(w, p);
                let wq = row(w, q);
                assert(wp.len == n && wq.len == n);
                for k in n {
                    let x = wp[k];
                    let y = wq[k];
                    wp[k] = c * x - s * y;
                    wq[k] = s * x + c * y;
                }
                for k in n {
                    w.data[k * n + p] = wp[k];
                    w.data[k * n + q] = wq[k];
                }
                // The block itself, where t was chosen to zero apq.
                wp[p] = app - t * apq;
                wq[q] = aqq + t * apq;
                wp[q] = 0.0;
                wq[p] = 0.0;
                rotate_rows(vt, p, q, c, s);
            }
        }
    }
    values.clear();
    for i in n { values.push(w.data[i * n + i]); }
    // Ascending, moving the vectors with their values.
    for i in n {
        var best = i;
        for j in i + 1..n { if values[j] < values[best] { best = j; } }
        if best != i {
            let t = values[i];
            values[i] = values[best];
            values[best] = t;
            swap_rows(vt, i, best);
        }
    }
    vectors = transpose(vt);
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
                let up = row(ut, p);
                let uq = row(ut, q);
                assert(up.len == m && uq.len == m);
                var alpha = 0.0;
                var beta = 0.0;
                var gamma = 0.0;
                for i in m {
                    alpha += up[i] * up[i];
                    beta += uq[i] * uq[i];
                    gamma += up[i] * uq[i];
                }
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
