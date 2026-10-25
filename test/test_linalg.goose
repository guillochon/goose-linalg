// Tests for the linalg package.
//
//   goose test/test_linalg.goose                                 (in-process)
//   goose build test/test_linalg.goose && test/test_linalg        (native)
//
// Each check that fails prints its group and what it checked; the run ends
// with a summary and exits 1 if anything failed. The tests pin down exact
// results on small hand-checked cases, and on random matrices of several
// sizes and shapes the identities each operation must satisfy (a = q r with
// q orthogonal, a v = v diag(w), p a = l u, ...).

import std;
import os;
import math;
import linalg from "..";

var passed = 0;
var failed = 0;
var group_name: u8[>..] = [];

fn group(name: const u8[:]) {
    group_name.clear();
    group_name.append(name);
}

fn check(ok: bool, what: const u8[:]) {
    if ok { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what);
}

// |got - want| <= tol * max(1, |want|).
fn check_close(got: f64, want: f64, tol: f64, what: const u8[:]) {
    let ok = abs(got - want) <= tol * max(1.0, abs(want));
    if ok { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": got ", got, ", want ", want);
}

fn check_mat(got, want, tol: f64, what: const u8[:]) {
    guard linalg::same_shape(got, want) else {
        failed++;
        print("FAIL [", group_name, "] ", what, ": shape ", got.rows, "x", got.cols, ", want ", want.rows, "x", want.cols);
        return;
    }
    let d = linalg::max_abs_diff(got, want);
    if d <= tol { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": max abs diff ", d, " > ", tol);
}

fn check_vec(got: f64[:], want: f64[:], tol: f64, what: const u8[:]) {
    guard got.len == want.len else {
        failed++;
        print("FAIL [", group_name, "] ", what, ": length ", got.len, ", want ", want.len);
        return;
    }
    var d = 0.0;
    for x, i in got { d = max(d, abs(x - want[i])); }
    if d <= tol { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": got ", got, ", want ", want);
}

// Sizes the property tests run over.
let SIZES = [1, 2, 3, 4, 7, 12, 25];

// A tolerance for an n-sized computation on O(1) entries.
fn tol(n: i64) -> f64 { 1.0e-12 * (n + 1) * (n + 1) }

fn random_symmetric(n: i64, r: rng&) -> linalg::matrix {
    let a = linalg::random(n, n, r);
    return linalg::add(a, linalg::transpose(a));
}

// b b^T + n I: symmetric positive definite and well conditioned.
fn random_spd(n: i64, r: rng&) -> linalg::matrix {
    let b = linalg::random(n, n, r);
    var a = linalg::matmul(b, linalg::transpose(b));
    linalg::add_scaled(a, n * 1.0, linalg::identity(n));
    return a;
}

// q^T q == I.
fn orthonormal_cols(q, tolerance: f64) -> bool {
    let g = linalg::matmul(linalg::transpose(q), q);
    return linalg::approx_equal(g, linalg::identity(q.cols), tolerance);
}

fn is_upper(a) -> bool {
    for i in a.rows { for j in min(i, a.cols) { if linalg::at(a, i, j) != 0.0 { return false; } } }
    return true;
}

fn is_lower(a) -> bool {
    for i in a.rows { for j in i + 1..a.cols { if linalg::at(a, i, j) != 0.0 { return false; } } }
    return true;
}

// ---------------------------------------------------------------------------

fn test_construction() {
    group("construction");
    let z = linalg::zeros(2, 3);
    check(z.rows == 2 && z.cols == 3 && z.data.len == 6, "zeros shape");
    check(linalg::max_abs(z) == 0.0, "zeros are zero");
    let f = linalg::filled(3, 2, 7.5);
    check(f.rows == 3 && f.cols == 2, "filled shape");
    var all = true;
    for x in f.data { all = all && x == 7.5; }
    check(all, "filled value");
    let i3 = linalg::identity(3);
    check(i3.data == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "identity");
    let a = linalg::from_rows(2, 3, [1, 2, 3, 4, 5, 6]);
    check(linalg::at(a, 0, 2) == 3.0 && linalg::at(a, 1, 0) == 4.0, "from_rows is row-major");
    let d = linalg::diag([1, -2, 3]);
    check(d.data == [1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0], "diag");

    var c = linalg::clone(a);
    linalg::set(c, 0, 0, 100.0);
    check(linalg::at(a, 0, 0) == 1.0 && linalg::at(c, 0, 0) == 100.0, "clone is independent");
    var cp = copy(a);
    linalg::set(cp, 1, 1, -1.0);
    check(linalg::at(a, 1, 1) == 5.0, "copy() is independent");

    var r1 = rng { 7 };
    var r2 = rng { 7 };
    let x = linalg::random(4, 5, r1);
    let y = linalg::random(4, 5, r2);
    check(linalg::max_abs_diff(x, y) == 0.0, "random is deterministic per seed");
    var inrange = true;
    for v in x.data { inrange = inrange && v >= -1.0 && v < 1.0; }
    check(inrange, "random entries in [-1, 1)");

    let e = linalg::zeros(0, 0);
    check(e.rows == 0 && e.cols == 0 && e.data.len == 0, "empty matrix");
    let thin = linalg::zeros(0, 4);
    check(thin.data.len == 0 && thin.cols == 4, "0 x 4 matrix");
    check(linalg::identity(0).data.len == 0, "identity(0)");
}

fn test_access() {
    group("access");
    var a = linalg::from_rows(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    check(linalg::at(a, 2, 1) == 8.0, "at");
    linalg::set(a, 2, 1, -8.0);
    check(a.data[7] == -8.0, "set");
    check(linalg::row(a, 1) == [4.0, 5.0, 6.0], "row");
    let rv = linalg::row(a, 0);
    rv[2] = 30.0;
    check(linalg::at(a, 0, 2) == 30.0, "row is a writable view");
    check(linalg::col(a, 1) == [2.0, 5.0, -8.0], "col");
    check(linalg::diagonal(a) == [1.0, 5.0, 9.0], "diagonal square");
    let w = linalg::from_rows(2, 3, [1, 2, 3, 4, 5, 6]);
    check(linalg::diagonal(w) == [1.0, 5.0], "diagonal wide");
    check(linalg::diagonal(linalg::transpose(w)) == [1.0, 5.0], "diagonal tall");

    let s = linalg::submatrix(a, 1, 3, 0, 2);
    check(s.rows == 2 && s.cols == 2 && s.data == [4.0, 5.0, 7.0, -8.0], "submatrix");
    check(linalg::submatrix(a, 1, 1, 0, 3).rows == 0, "empty submatrix");
    check_mat(linalg::submatrix(a, 0, 3, 0, 3), a, 0.0, "whole submatrix");

    var big = linalg::zeros(4, 4);
    linalg::set_block(big, 1, 2, linalg::from_rows(2, 2, [1, 2, 3, 4]));
    check(linalg::at(big, 1, 2) == 1.0 && linalg::at(big, 2, 3) == 4.0 && linalg::at(big, 0, 0) == 0.0, "set_block");
    check(linalg::norm_fro(big) == sqrt(30.0), "set_block writes nothing else");

    check(linalg::is_square(a) && !linalg::is_square(w), "is_square");
    check(linalg::same_shape(w, linalg::zeros(2, 3)) && !linalg::same_shape(w, linalg::zeros(3, 2)), "same_shape");

    let m = linalg::from_rows(2, 2, [1.0, 2.5, -3, 4]);
    check(str(m) == "[1.0 2.5; -3.0 4.0]", "format");
    check(str(linalg::zeros(0, 0)) == "[]", "format empty");
}

fn test_elementwise() {
    group("elementwise");
    let a = linalg::from_rows(2, 2, [1, 2, 3, 4]);
    let b = linalg::from_rows(2, 2, [10, 20, 30, 40]);
    check(linalg::add(a, b).data == [11.0, 22.0, 33.0, 44.0], "add");
    check(linalg::sub(b, a).data == [9.0, 18.0, 27.0, 36.0], "sub");
    check(linalg::scale(a, -2.0).data == [-2.0, -4.0, -6.0, -8.0], "scale");
    check(linalg::hadamard(a, b).data == [10.0, 40.0, 90.0, 160.0], "hadamard");
    check(linalg::max_abs(linalg::sub(a, a)) == 0.0, "a - a = 0");

    var c = linalg::clone(a);
    linalg::add_assign(c, b);
    check_mat(c, linalg::add(a, b), 0.0, "add_assign matches add");
    linalg::sub_assign(c, b);
    check_mat(c, a, 0.0, "sub_assign undoes add_assign");
    linalg::scale_assign(c, 3.0);
    check_mat(c, linalg::scale(a, 3.0), 0.0, "scale_assign matches scale");
    var d = linalg::clone(a);
    linalg::add_scaled(d, 0.5, b);
    check(d.data == [6.0, 12.0, 18.0, 24.0], "add_scaled");

    var r = rng { 11 };
    let x = linalg::random(5, 3, r);
    let y = linalg::random(5, 3, r);
    let z = linalg::random(5, 3, r);
    check_mat(linalg::add(x, y), linalg::add(y, x), 0.0, "add commutes");
    check_mat(linalg::add(linalg::add(x, y), z), linalg::add(x, linalg::add(y, z)), 1.0e-15, "add associates");
    check_mat(linalg::scale(linalg::add(x, y), 2.5), linalg::add(linalg::scale(x, 2.5), linalg::scale(y, 2.5)), 1.0e-15,
              "scale distributes");
}

fn test_products() {
    group("products");
    let a = linalg::from_rows(2, 3, [1, 2, 3, 4, 5, 6]);
    let b = linalg::from_rows(3, 2, [7, 8, 9, 10, 11, 12]);
    let ab = linalg::matmul(a, b);
    check(ab.rows == 2 && ab.cols == 2 && ab.data == [58.0, 64.0, 139.0, 154.0], "matmul 2x3 * 3x2");
    let ba = linalg::matmul(b, a);
    check(ba.rows == 3 && ba.cols == 3 && linalg::at(ba, 0, 0) == 39.0 && linalg::at(ba, 2, 2) == 105.0, "matmul 3x2 * 2x3");
    check(linalg::transpose(a).data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], "transpose");
    check(linalg::matvec(a, [1, 0, -1]) == [-2.0, -2.0], "matvec");
    let o = linalg::outer([1, 2], [3, 4, 5]);
    check(o.rows == 2 && o.cols == 3 && o.data == [3.0, 4.0, 5.0, 6.0, 8.0, 10.0], "outer");
    let row_times_col = linalg::matmul(linalg::from_rows(1, 3, [1, 2, 3]), linalg::from_rows(3, 1, [4, 5, 6]));
    check(row_times_col.rows == 1 && row_times_col.cols == 1 && row_times_col.data[0] == 32.0, "1x3 * 3x1");
    let empty = linalg::matmul(linalg::zeros(2, 0), linalg::zeros(0, 3));
    check(empty.rows == 2 && empty.cols == 3 && linalg::max_abs(empty) == 0.0, "inner dimension 0 gives zeros");

    // matmul works on blocks of four rows and four inner steps: every
    // remainder of both, against the naive triple loop.
    var g0 = rng { 99 };
    for rows in 1..10 {
        for inner in [1, 3, 4, 5, 8, 11] {
            let x = linalg::random(rows, inner, g0);
            let y = linalg::random(inner, 6, g0);
            var naive = linalg::zeros(rows, 6);
            for i in rows { for j in 6 {
                var s = 0.0;
                for k in inner { s += linalg::at(x, i, k) * linalg::at(y, k, j); }
                linalg::set(naive, i, j, s);
            } }
            check_mat(linalg::matmul(x, y), naive, 1.0e-14, str("blocked matmul remainders ", rows, "x", inner));
        }
    }
    // gemm_acc goes through b in cache blocks of 256 rows by 512 columns: a
    // product past both, against the naive triple loop.
    let wide_a = linalg::random(5, 600, g0);
    let wide_b = linalg::random(600, 1100, g0);
    var wide_naive = linalg::zeros(5, 1100);
    for i in 5 { for j in 1100 {
        var s = 0.0;
        for k in 600 { s += linalg::at(wide_a, i, k) * linalg::at(wide_b, k, j); }
        linalg::set(wide_naive, i, j, s);
    } }
    check_mat(linalg::matmul(wide_a, wide_b), wide_naive, 1.0e-12, "matmul past the cache blocks, 5x600 * 600x1100");

    // dot keeps 16 partial sums: every remainder, against a plain loop.
    for len in 0..40 {
        let x = linalg::col(linalg::random(len, 1, g0), 0);
        let y = linalg::col(linalg::random(len, 1, g0), 0);
        var s = 0.0;
        for i in len { s += x[i] * y[i]; }
        check_close(linalg::dot(x, y), s, 1.0e-14, str("dot lanes and tail, len=", len));
    }

    let fib = linalg::power(linalg::from_rows(2, 2, [1, 1, 1, 0]), 10);
    check(fib.data == [89.0, 55.0, 55.0, 34.0], "power: Fibonacci");
    let m = linalg::from_rows(2, 2, [2.0, -1, 0.5, 3]);
    check_mat(linalg::power(m, 0), linalg::identity(2), 0.0, "power 0 is identity");
    check_mat(linalg::power(m, 1), m, 0.0, "power 1");
    let m5 = linalg::matmul(linalg::matmul(linalg::matmul(m, m), linalg::matmul(m, m)), m);
    check_mat(linalg::power(m, 5), m5, 1.0e-12, "power 5");

    var r = rng { 12 };
    for n in SIZES {
        let x = linalg::random(n, n + 1, r);
        let y = linalg::random(n + 1, n + 2, r);
        let z = linalg::random(n + 2, n, r);
        let t = tol(n);
        check_mat(linalg::matmul(x, linalg::identity(n + 1)), x, 0.0, str("x I = x, n=", n));
        check_mat(linalg::matmul(linalg::identity(n), x), x, 0.0, str("I x = x, n=", n));
        check_mat(linalg::matmul(linalg::matmul(x, y), z), linalg::matmul(x, linalg::matmul(y, z)), t, str("associativity, n=", n));
        let y2 = linalg::random(n + 1, n + 2, r);
        check_mat(linalg::matmul(x, linalg::add(y, y2)), linalg::add(linalg::matmul(x, y), linalg::matmul(x, y2)), t,
                  str("distributivity, n=", n));
        check_mat(linalg::transpose(linalg::transpose(x)), x, 0.0, str("transpose twice, n=", n));
        check_mat(linalg::transpose(linalg::matmul(x, y)), linalg::matmul(linalg::transpose(y), linalg::transpose(x)), t,
                  str("(xy)^T = y^T x^T, n=", n));
        // matvec agrees with matmul by a one-column matrix.
        let v = linalg::col(linalg::random(n + 1, 1, r), 0);
        let mv = linalg::matmul(x, linalg::from_rows(n + 1, 1, v));
        check_vec(linalg::matvec(x, v), mv.data, t, str("matvec = matmul, n=", n));
        // A naive triple loop as the reference for the optimized matmul.
        var naive = linalg::zeros(n, n + 2);
        for i in n { for j in n + 2 {
            var s = 0.0;
            for k in n + 1 { s += linalg::at(x, i, k) * linalg::at(y, k, j); }
            linalg::set(naive, i, j, s);
        } }
        check_mat(linalg::matmul(x, y), naive, t, str("matmul = naive, n=", n));
    }
}

fn test_vectors() {
    group("vectors");
    check(linalg::dot([1, 2, 3], [4, -5, 6]) == 12.0, "dot");
    check(linalg::dot(linalg::zeros(0, 0).data, linalg::zeros(0, 0).data) == 0.0, "dot of empty");
    check(linalg::vnorm([3, 4]) == 5.0, "vnorm 3-4-5");
    check(linalg::vnorm([0, 0, 0]) == 0.0, "vnorm of zero");
    check_close(linalg::vnorm([3.0e200, 4.0e200]), 5.0e200, 1.0e-15, "vnorm does not overflow");
    check_close(linalg::vnorm([3.0e-200, 4.0e-200]), 5.0e-200, 1.0e-15, "vnorm does not underflow");
    check(linalg::vnorm1([1, -2, 3]) == 6.0, "vnorm1");
    check(linalg::vnorm_inf([1, -7, 3]) == 7.0, "vnorm_inf");
    check(linalg::vadd([1, 2], [3, 4]) == [4.0, 6.0], "vadd");
    check(linalg::vsub([1, 2], [3, 5]) == [-2.0, -3.0], "vsub");
    check(linalg::vscale([1, -2], 3.0) == [3.0, -6.0], "vscale");
    var y: f64[>..] = [1.0, 1.0, 1.0];
    linalg::axpy(2.0, [1, 2, 3], y);
    check(y == [3.0, 5.0, 7.0], "axpy");
    var fixed: f64[3] = [1.0, 2.0, 2.0];
    check(linalg::vnorm(fixed) == 3.0, "fixed arrays pass as slices");

    var r = rng { 13 };
    for n in SIZES {
        let a = linalg::col(linalg::random(n, 1, r), 0);
        let b = linalg::col(linalg::random(n, 1, r), 0);
        check(abs(linalg::dot(a, b)) <= linalg::vnorm(a) * linalg::vnorm(b) * (1.0 + 1.0e-14), str("Cauchy-Schwarz, n=", n));
        check(linalg::vnorm(linalg::vadd(a, b)) <= (linalg::vnorm(a) + linalg::vnorm(b)) * (1.0 + 1.0e-14), str("triangle inequality, n=", n));
        check_close(linalg::vnorm(a) * linalg::vnorm(a), linalg::dot(a, a), 1.0e-13, str("vnorm^2 = dot, n=", n));
        check(linalg::vnorm_inf(a) <= linalg::vnorm(a) && linalg::vnorm(a) <= linalg::vnorm1(a) * (1.0 + 1.0e-14), str("norm ordering, n=", n));
    }
}

fn test_scalars() {
    group("matrix scalars");
    let a = linalg::from_rows(2, 3, [1, -2, 3, -4, 5, -6]);
    check(linalg::norm1(a) == 9.0, "norm1 is max column sum");
    check(linalg::norm_inf(a) == 15.0, "norm_inf is max row sum");
    check_close(linalg::norm_fro(a), sqrt(91.0), 1.0e-15, "norm_fro");
    check(linalg::max_abs(a) == 6.0, "max_abs");
    check(linalg::trace(linalg::from_rows(2, 2, [1, 9, 9, 2])) == 3.0, "trace");
    check(linalg::trace(linalg::identity(5)) == 5.0, "trace of identity");
    check(linalg::is_symmetric(linalg::from_rows(2, 2, [1, 2, 2, 1])), "is_symmetric yes");
    check(!linalg::is_symmetric(linalg::from_rows(2, 2, [1, 2, 3, 1])), "is_symmetric no");
    check(!linalg::is_symmetric(a), "a wide matrix is not symmetric");
    check(linalg::is_symmetric(linalg::from_rows(2, 2, [1.0, 2, 2.001, 1]), 0.01), "is_symmetric with tolerance");
    check(!linalg::approx_equal(a, linalg::transpose(a), 1.0e9), "approx_equal needs the same shape");
    check(linalg::approx_equal(a, linalg::add(a, linalg::filled(2, 3, 1.0e-10)), 1.0e-9), "approx_equal within tol");
    check(linalg::max_abs_diff(a, linalg::zeros(2, 3)) == 6.0, "max_abs_diff");

    var r = rng { 14 };
    for n in SIZES {
        let x = linalg::random(n, n, r);
        let y = linalg::random(n, n, r);
        check_close(linalg::trace(linalg::matmul(x, y)), linalg::trace(linalg::matmul(y, x)), tol(n), str("trace(xy) = trace(yx), n=", n));
        check_close(linalg::norm1(x), linalg::norm_inf(linalg::transpose(x)), 0.0, str("norm1(x) = norm_inf(x^T), n=", n));
        check(linalg::norm_fro(linalg::matmul(x, y)) <= linalg::norm_fro(x) * linalg::norm_fro(y) * (1.0 + 1.0e-13),
              str("Frobenius is submultiplicative, n=", n));
        check(linalg::is_symmetric(random_symmetric(n, r)), str("x + x^T is symmetric, n=", n));
    }
}

fn test_triangular() {
    group("triangular");
    let l = linalg::from_rows(3, 3, [2, 0, 0, 1, 3, 0, -1, 2, 4]);
    check_vec(linalg::solve_lower(l, [2, 7, 11]), [1, 2, 2], 1.0e-15, "solve_lower");
    let u = linalg::transpose(l);
    let x = linalg::solve_upper(u, [3, 8, 8]);
    check_vec(linalg::matvec(u, x), [3, 8, 8], 1.0e-14, "solve_upper residual");
    check_vec(linalg::solve_lower(l, [1, 1, 1], true), [1.0, 0.0, 2.0], 1.0e-15, "solve_lower with unit diagonal");
    // Entries above the diagonal are not read.
    let junk = linalg::from_rows(2, 2, [1, 99, 2, 1]);
    check_vec(linalg::solve_lower(junk, [1, 4]), [1, 2], 0.0, "solve_lower ignores the upper triangle");

    var r = rng { 15 };
    for n in SIZES {
        var t = linalg::random(n, n, r);
        for i in n {
            for j in i + 1..n { linalg::set(t, i, j, 0.0); }
            linalg::set(t, i, i, 2.0 + linalg::at(t, i, i));    // well away from singular
        }
        let b = linalg::col(linalg::random(n, 1, r), 0);
        check_vec(linalg::matvec(t, linalg::solve_lower(t, b)), b, tol(n), str("lower residual, n=", n));
        let tu = linalg::transpose(t);
        check_vec(linalg::matvec(tu, linalg::solve_upper(tu, b)), b, tol(n), str("upper residual, n=", n));
    }
}

fn test_lu() {
    group("lu");
    check(linalg::det(linalg::from_rows(2, 2, [3, 8, 4, 6])) == -14.0, "det 2x2");
    check_close(linalg::det(linalg::from_rows(3, 3, [6, 1, 1, 4, -2, 5, 2, 8, 7])), -306.0, 1.0e-14, "det 3x3");
    check(linalg::det(linalg::identity(6)) == 1.0, "det identity");
    check(linalg::det(linalg::zeros(0, 0)) == 1.0, "det of 0x0 is 1");
    check(linalg::det(linalg::from_rows(1, 1, [-2.5])) == -2.5, "det 1x1");
    check(linalg::det(linalg::from_rows(2, 2, [0, 1, 1, 0])) == -1.0, "det of a swap is -1");
    check_close(linalg::det(linalg::from_rows(3, 3, [2, 5, 7, 0, 3, 1, 0, 0, -4])), -24.0, 0.0, "det triangular");

    // Singular matrices.
    let s = linalg::from_rows(3, 3, [1, 2, 3, 2, 4, 6, 1, 0, 1]);
    check(linalg::det(s) == 0.0, "det singular");
    let sx, sok = linalg::solve(s, [1, 2, 3]);
    check(!sok, "solve reports singular");
    let si, siok = linalg::inverse(s);
    check(!siok, "inverse reports singular");
    var zero = linalg::zeros(3, 3);
    var zpiv: i64[>..] = [];
    let zsign, zok = linalg::lu_factor(zero, zpiv);
    check(!zok, "lu_factor of zero matrix");

    // A zero in the first pivot position needs the row swap.
    let p = linalg::from_rows(2, 2, [0, 2, 3, 0]);
    let px, pok = linalg::solve(p, [4, 9]);
    check(pok && px == [3.0, 2.0], "solve needs pivoting");

    // The 4x4 Hilbert matrix, whose inverse has integer entries.
    var h = linalg::zeros(4, 4);
    for i in 4 { for j in 4 { linalg::set(h, i, j, 1.0 / (i + j + 1)); } }
    let hinv_want = linalg::from_rows(4, 4, [16, -120, 240, -140, -120, 1200, -2700, 1680,
                                              240, -2700, 6480, -4200, -140, 1680, -4200, 2800]);
    let hinv, hok = linalg::inverse(h);
    check(hok, "Hilbert is invertible");
    check_mat(hinv, hinv_want, 1.0e-8, "Hilbert inverse");
    check_close(linalg::det(h), 1.0 / 6048000.0, 1.0e-10, "Hilbert determinant");

    var r = rng { 16 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let t = tol(n) * 10.0;
        // p a = l u.
        var lu = linalg::clone(a);
        var piv: i64[>..] = [];
        let sign, ok = linalg::lu_factor(lu, piv);
        check(ok, str("random matrix factors, n=", n));
        var pm = linalg::zeros(n, n);
        var l = linalg::identity(n);
        var u = linalg::zeros(n, n);
        for i in n {
            linalg::set(pm, i, piv[i], 1.0);
            for j in n {
                if j < i { linalg::set(l, i, j, linalg::at(lu, i, j)); } else { linalg::set(u, i, j, linalg::at(lu, i, j)); }
            }
        }
        check_mat(linalg::matmul(pm, a), linalg::matmul(l, u), t, str("p a = l u, n=", n));
        check_close(linalg::det(pm), sign, 0.0, str("sign is det(p), n=", n));
        var bounded = true;
        for i in n { for j in i { bounded = bounded && abs(linalg::at(l, i, j)) <= 1.0; } }
        check(bounded, str("partial pivoting keeps |l| <= 1, n=", n));

        // Determinant identities.
        let b = linalg::random(n, n, r);
        let da = linalg::det(a);
        let db = linalg::det(b);
        check_close(linalg::det(linalg::matmul(a, b)), da * db, t, str("det(ab) = det(a) det(b), n=", n));
        check_close(linalg::det(linalg::transpose(a)), da, t, str("det(a^T) = det(a), n=", n));
        check_close(linalg::det(linalg::scale(a, 2.0)), pow(2.0, n * 1.0) * da, t, str("det(2a) = 2^n det(a), n=", n));

        // Solves and inverses.
        let rhs = linalg::col(linalg::random(n, 1, r), 0);
        let x, sok2 = linalg::solve(a, rhs);
        check(sok2, str("solve ok, n=", n));
        let resid = linalg::vnorm(linalg::vsub(linalg::matvec(a, x), rhs));
        check(resid <= t * linalg::norm_fro(a) * linalg::vnorm(x) + t, str("solve residual, n=", n));
        check_vec(linalg::lu_solve(lu, piv, rhs), x, t, str("lu_solve = solve, n=", n));
        let inv, iok = linalg::inverse(a);
        check(iok, str("inverse ok, n=", n));
        let c = linalg::cond(a);
        check_mat(linalg::matmul(a, inv), linalg::identity(n), t * c, str("a inv = I, n=", n));
        check_mat(linalg::matmul(inv, a), linalg::identity(n), t * c, str("inv a = I, n=", n));
        check_close(linalg::det(inv) * da, 1.0, t * c, str("det(inv) = 1/det, n=", n));
        let bm = linalg::random(n, 3, r);
        let xm, mok = linalg::solve_matrix(a, bm);
        check(mok && xm.rows == n && xm.cols == 3, str("solve_matrix shape, n=", n));
        check_mat(linalg::matmul(a, xm), bm, t * c, str("solve_matrix residual, n=", n));
    }
}

fn test_cholesky() {
    group("cholesky");
    let a = linalg::from_rows(3, 3, [4, 12, -16, 12, 37, -43, -16, -43, 98]);
    let l, ok = linalg::cholesky(a);
    check(ok, "textbook example is positive definite");
    check_mat(l, linalg::from_rows(3, 3, [2, 0, 0, 6, 1, 0, -8, 5, 3]), 1.0e-14, "textbook factor");
    let nl, nok = linalg::cholesky(linalg::from_rows(2, 2, [1, 2, 2, 1]));
    check(!nok, "indefinite matrix rejected");
    let zl, zok = linalg::cholesky(linalg::zeros(2, 2));
    check(!zok, "zero matrix rejected");
    let psd, psdok = linalg::cholesky(linalg::from_rows(2, 2, [1, 1, 1, 1]));
    check(!psdok, "semidefinite (singular) matrix rejected");
    let z1, z1ok = linalg::cholesky(linalg::from_rows(1, 1, [0]));
    check(!z1ok, "1x1 zero rejected");
    let nd, ndok = linalg::cholesky(linalg::from_rows(1, 1, [-4]));
    check(!ndok, "negative 1x1 rejected");
    let one, oneok = linalg::cholesky(linalg::from_rows(1, 1, [9]));
    check(oneok && one.data == [3.0], "1x1");

    var r = rng { 17 };
    for n in SIZES {
        let s = random_spd(n, r);
        let f, fok = linalg::cholesky(s);
        check(fok, str("spd factors, n=", n));
        check(is_lower(f), str("factor is lower triangular, n=", n));
        var pos = true;
        for i in n { pos = pos && linalg::at(f, i, i) > 0.0; }
        check(pos, str("positive diagonal, n=", n));
        check_mat(linalg::matmul(f, linalg::transpose(f)), s, tol(n) * n, str("l l^T = a, n=", n));
        let b = linalg::col(linalg::random(n, 1, r), 0);
        let x = linalg::cholesky_solve(f, b);
        check_vec(linalg::matvec(s, x), b, tol(n) * n, str("cholesky_solve residual, n=", n));
        let xl, lok = linalg::solve(s, b);
        check_vec(x, xl, tol(n), str("cholesky_solve = solve, n=", n));
        check_close(linalg::det(s), pow(linalg::det(f), 2.0), tol(n) * 10.0, str("det(a) = det(l)^2, n=", n));
    }
}

fn test_qr() {
    group("qr");
    var q = linalg::zeros(0, 0);
    var r = linalg::zeros(0, 0);
    let a = linalg::from_rows(3, 2, [3, 0, 4, 0, 0, 5]);
    linalg::qr(a, q, r);
    check(q.rows == 3 && q.cols == 3 && r.rows == 3 && r.cols == 2, "shapes for 3x2");
    check_close(abs(linalg::at(r, 0, 0)), 5.0, 1.0e-15, "|r00| is the first column's norm");
    check_close(abs(linalg::at(r, 1, 1)), 5.0, 1.0e-15, "|r11|");

    // Least squares: an exact line, then a noisy one against the normal equations.
    var design = linalg::zeros(5, 2);
    var ys: f64[>..] = [];
    for i in 5 {
        linalg::set(design, i, 0, i * 1.0);
        linalg::set(design, i, 1, 1.0);
        ys.push(2.0 * i + 1.0);
    }
    let fit, fok = linalg::lstsq(design, ys);
    check(fok, "line fit ok");
    check_vec(fit, [2, 1], 1.0e-13, "line fit recovers slope and intercept");
    let noisy = [1.1, 2.9, 5.2, 6.8, 9.1];
    let nfit, nok = linalg::lstsq(design, noisy);
    let dt = linalg::transpose(design);
    let normal, normok = linalg::solve(linalg::matmul(dt, design), linalg::matvec(dt, noisy));
    check(nok && normok, "noisy fit ok");
    check_vec(nfit, normal, 1.0e-12, "lstsq = normal equations");
    let rd, rdok = linalg::lstsq(linalg::from_rows(3, 2, [1, 2, 2, 4, 3, 6]), [1, 2, 3]);
    check(!rdok, "rank-deficient lstsq rejected");
    let sq = linalg::from_rows(2, 2, [2, 1, 1, 3]);
    let sqx, sqok = linalg::lstsq(sq, [3, 5]);
    let sqs, sqsok = linalg::solve(sq, [3, 5]);
    check(sqok && sqsok, "square lstsq ok");
    check_vec(sqx, sqs, 1.0e-14, "square lstsq = solve");

    var g = rng { 18 };
    for n in SIZES {
        for shape in 3 {
            let rows = if shape == 0 { n } else if shape == 1 { n + 3 } else { n };
            let cols = if shape == 2 { n + 3 } else { n };
            let m = linalg::random(rows, cols, g);
            let t = tol(rows + cols);
            linalg::qr(m, q, r);
            let what = str(rows, "x", cols);
            check(q.rows == rows && q.cols == rows && r.rows == rows && r.cols == cols, str("shapes ", what));
            check(orthonormal_cols(q, t), str("q^T q = I ", what));
            check(orthonormal_cols(linalg::transpose(q), t), str("q q^T = I ", what));
            check(is_upper(r), str("r upper triangular ", what));
            check_mat(linalg::matmul(q, r), m, t, str("q r = a ", what));
            check_close(abs(linalg::det(q)), 1.0, t, str("|det q| = 1 ", what));
            if rows >= cols {
                let b = linalg::col(linalg::random(rows, 1, g), 0);
                let x, ok = linalg::lstsq(m, b);
                check(ok, str("lstsq ok ", what));
                // The residual is orthogonal to the columns: a^T (a x - b) = 0.
                let res = linalg::vsub(linalg::matvec(m, x), b);
                check(linalg::vnorm(linalg::matvec(linalg::transpose(m), res)) <= t * 10.0, str("lstsq residual orthogonal ", what));
            }
        }
    }
}

fn test_eigh() {
    group("eigh");
    var w: f64[>..] = [];
    var v = linalg::zeros(0, 0);
    check(linalg::eigh(linalg::from_rows(2, 2, [2, 1, 1, 2]), w, v), "2x2 converges");
    check_vec(w, [1, 3], 1.0e-15, "eigenvalues of [2 1; 1 2]");
    check_close(abs(linalg::at(v, 0, 0)), sqrt(0.5), 1.0e-15, "eigenvector entries are 1/sqrt 2");
    check(linalg::eigh(linalg::diag([3, -1, 2]), w, v), "diagonal converges");
    check(w == [-1.0, 2.0, 3.0], "diagonal eigenvalues, sorted");
    check(linalg::eigh(linalg::identity(4), w, v), "identity converges");
    check(w == [1.0, 1.0, 1.0, 1.0] && orthonormal_cols(v, 0.0), "repeated eigenvalues");
    check(linalg::eigh(linalg::from_rows(1, 1, [-7]), w, v) && w == [-7.0] && v.data == [1.0], "1x1");
    check(linalg::eigh(linalg::zeros(0, 0), w, v) && w.len == 0, "0x0");
    check(linalg::eigh(linalg::zeros(3, 3), w, v) && w == [0.0, 0.0, 0.0], "zero matrix");
    // The path graph's Laplacian: eigenvalues 2 - 2 cos(k pi / (n + 1)).
    let n = 6;
    var lap = linalg::zeros(n, n);
    for i in n {
        linalg::set(lap, i, i, 2.0);
        if i > 0 { linalg::set(lap, i, i - 1, -1.0); linalg::set(lap, i - 1, i, -1.0); }
    }
    var want: f64[>..] = [];
    for k in 1..n + 1 { want.push(2.0 - 2.0 * cos(k * PI / (n + 1))); }
    // Wilkinson's W21+: eigenvalues in pairs agreeing to about 1e-14, which
    // still need orthogonal eigenvectors.
    var wil = linalg::zeros(21, 21);
    for i in 21 {
        linalg::set(wil, i, i, abs(10 - i) * 1.0);
        if i > 0 { linalg::set(wil, i, i - 1, 1.0); linalg::set(wil, i - 1, i, 1.0); }
    }
    check(linalg::eigh(wil, w, v), "W21+ converges");
    check_close(w[20], 10.746194182903322, 1.0e-14, "W21+ largest eigenvalue");
    check_close(w[20] - w[19], 0.0, 1.0e-13, "W21+ top pair nearly equal");
    check(orthonormal_cols(v, 1.0e-13), "W21+ eigenvectors orthonormal");
    check_mat(linalg::matmul(wil, v), linalg::matmul(v, linalg::diag(w)), 1.0e-12, "W21+ a v = v diag(w)");
    // Only the upper triangle is read.
    var junk = linalg::clone(lap);
    for i in n { for j in i { linalg::set(junk, i, j, 99.0); } }
    check_vec(linalg::eigvals_sym(junk), linalg::eigvals_sym(lap), 0.0, "eigh reads only the upper triangle");
    check(linalg::eigh(lap, w, v), "Laplacian converges");
    check_vec(w, want, 1.0e-14, "Laplacian spectrum");
    check_vec(linalg::eigvals_sym(lap), want, 1.0e-14, "eigvals_sym");

    var g = rng { 19 };
    for n2 in SIZES {
        let a = random_symmetric(n2, g);
        let t = tol(n2) * 10.0;
        check(linalg::eigh(a, w, v), str("converges, n=", n2));
        check(w.len == n2 && v.rows == n2 && v.cols == n2, str("shapes, n=", n2));
        var sorted = true;
        for i in 1..n2 { sorted = sorted && w[i - 1] <= w[i]; }
        check(sorted, str("ascending, n=", n2));
        check(orthonormal_cols(v, t), str("v^T v = I, n=", n2));
        check_mat(linalg::matmul(a, v), linalg::matmul(v, linalg::diag(w)), t, str("a v = v diag(w), n=", n2));
        check_mat(linalg::matmul(linalg::matmul(v, linalg::diag(w)), linalg::transpose(v)), a, t, str("v diag(w) v^T = a, n=", n2));
        var sum = 0.0;
        var prod = 1.0;
        for x in w { sum += x; prod *= x; }
        check_close(sum, linalg::trace(a), t, str("sum of eigenvalues = trace, n=", n2));
        check_close(prod, linalg::det(a), t * 10.0, str("product of eigenvalues = det, n=", n2));
        let s = random_spd(n2, g);
        var all_pos = true;
        for x in linalg::eigvals_sym(s) { all_pos = all_pos && x > 0.0; }
        check(all_pos, str("spd has positive eigenvalues, n=", n2));
    }
}

fn test_svd() {
    group("svd");
    var u = linalg::zeros(0, 0);
    var v = linalg::zeros(0, 0);
    var s: f64[>..] = [];
    check(linalg::svd(linalg::diag([3, -2]), u, s, v), "diagonal converges");
    check_vec(s, [3, 2], 1.0e-15, "singular values of diag(3, -2)");
    check_vec(linalg::singular_values(linalg::from_rows(2, 3, [3, 2, 2, 2, 3, -2])), [5, 3], 1.0e-14,
              "textbook 2x3 singular values");
    check(linalg::singular_values(linalg::zeros(0, 0)).len == 0, "0x0");
    check(linalg::svd(linalg::from_rows(1, 1, [-4]), u, s, v) && s == [4.0], "1x1");
    check_vec(linalg::singular_values(linalg::zeros(3, 2)), [0, 0], 0.0, "zero matrix");
    check(linalg::svd(linalg::zeros(3, 2), u, s, v) && orthonormal_cols(u, 1.0e-15), "zero matrix still gives orthonormal u");

    check(linalg::rank(linalg::identity(5)) == 5, "rank of identity");
    check(linalg::rank(linalg::zeros(3, 4)) == 0, "rank of zero");
    check(linalg::rank(linalg::outer([1, 2, 3], [4, 5])) == 1, "rank of an outer product");
    let r2 = linalg::add(linalg::outer([1, 0, 1, 2], [1, 1, 0]), linalg::outer([0, 1, 1, -1], [2, -1, 1]));
    check(linalg::rank(r2) == 2, "rank 2");
    check(linalg::rank(linalg::transpose(r2)) == 2, "rank of the transpose");
    check(linalg::rank(linalg::from_rows(2, 2, [1.0, 0, 0, 1.0e-20])) == 1, "default tolerance");
    check(linalg::rank(linalg::from_rows(2, 2, [1.0, 0, 0, 1.0e-20]), 0.0) == 2, "zero tolerance");
    check_close(linalg::cond(linalg::identity(3)), 1.0, 1.0e-15, "cond of identity");
    check_close(linalg::cond(linalg::diag([10.0, 0.1])), 100.0, 1.0e-14, "cond of diag");
    check(linalg::cond(linalg::outer([1, 2], [3, 4])) > 1.0e15, "cond of a singular matrix is huge");
    check(linalg::cond(linalg::zeros(2, 2)) == 1.0 / 0.0, "cond of zero is inf");
    check_close(linalg::norm2(linalg::from_rows(2, 2, [3, 0, 4, 5])), sqrt(45.0), 1.0e-14, "norm2");

    // Matrices that drive the bidiagonal QR into its rarer branches: a zero
    // singular value mid-diagonal (a split), a zero at the end (a
    // deflation), a zero column, and all singular values equal.
    var odd = linalg::zeros(0, 0);
    for case in 4 {
        odd = if case == 0 { linalg::diag([3.0, 0, 2]) }
              else if case == 1 { linalg::from_rows(3, 3, [1, 1, 0, 0, 1, 1, 0, 0, 0]) }
              else if case == 2 { linalg::from_rows(3, 3, [1, 0, 2, 3, 0, 4, 5, 0, 6]) }
              else { linalg::from_rows(2, 2, [0.6, -0.8, 0.8, 0.6]) };
        let what = str("special case ", case);
        check(linalg::svd(odd, u, s, v), str(what, " converges"));
        check(orthonormal_cols(u, 1.0e-14) && orthonormal_cols(v, 1.0e-14), str(what, " orthonormal factors"));
        check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), odd, 1.0e-14,
                  str(what, " u s v^T = a"));
        var desc = true;
        for i in 1..s.len { desc = desc && s[i - 1] >= s[i]; }
        check(desc && s[s.len - 1] >= 0.0, str(what, " descending, non-negative"));
    }
    check_vec(linalg::singular_values(linalg::diag([3.0, 0, 2])), [3, 2, 0], 1.0e-15, "split at a zero");
    check_vec(linalg::singular_values(linalg::from_rows(2, 2, [0.6, -0.8, 0.8, 0.6])), [1, 1], 1.0e-15,
              "orthogonal matrix");

    // Graded singular values, 1 down to 1e-15, behind random orthogonal
    // factors: each comes back within eps * s_max (Golub-Kahan's guarantee).
    var gq = rng { 21 };
    var q1 = linalg::zeros(0, 0);
    var r1 = linalg::zeros(0, 0);
    var q2 = linalg::zeros(0, 0);
    linalg::qr(linalg::random(6, 6, gq), q1, r1);
    linalg::qr(linalg::random(4, 4, gq), q2, r1);
    let graded = [1.0, 1.0e-5, 1.0e-10, 1.0e-15];
    var sig = linalg::zeros(6, 4);
    for i in 4 { linalg::set(sig, i, i, graded[i]); }
    let gm = linalg::matmul(linalg::matmul(q1, sig), linalg::transpose(q2));
    check_vec(linalg::singular_values(gm), graded, 1.0e-15, "graded singular values");
    check(linalg::rank(gm) == 3, "graded: the 1e-15 value is below the rank tolerance");

    // A zero on the diagonal of an upper bidiagonal matrix survives the
    // bidiagonalization, so the QR phase must split there.
    let split = linalg::from_rows(4, 4, [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]);
    check(linalg::svd(split, u, s, v), "split converges");
    check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), split, 1.0e-14,
              "split u s v^T = a");
    check(orthonormal_cols(u, 1.0e-14) && orthonormal_cols(v, 1.0e-14), "split orthonormal factors");
    check_close(s[3], 0.0, 1.0e-15, "split: the matrix is singular");

    // Tightly clustered singular values converge only with the Wilkinson
    // shift: unshifted QR would need on the order of 1e5 iterations.
    var clustered: f64[>..] = [];
    for i in 10 { clustered.push(1.0 - 1.0e-4 * i); }
    var qa = linalg::zeros(0, 0);
    var qb = linalg::zeros(0, 0);
    linalg::qr(linalg::random(10, 10, gq), qa, r1);
    linalg::qr(linalg::random(10, 10, gq), qb, r1);
    let cm = linalg::matmul(linalg::matmul(qa, linalg::diag(clustered)), linalg::transpose(qb));
    check(linalg::svd(cm, u, s, v), "clustered singular values converge");
    check_vec(s, clustered, 1.0e-14, "clustered singular values");

    // A larger tall matrix.
    let big = linalg::random(60, 40, gq);
    check(linalg::svd(big, u, s, v), "60x40 converges");
    check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), big, 1.0e-12, "60x40 u s v^T = a");
    check(orthonormal_cols(u, 1.0e-12) && orthonormal_cols(v, 1.0e-12), "60x40 orthonormal factors");

    // The pseudoinverse of an invertible matrix is its inverse.
    let sq = linalg::from_rows(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2]);
    let sqinv, sqok = linalg::inverse(sq);
    check_mat(linalg::pinv(sq), sqinv, 1.0e-14, "pinv of invertible = inverse");
    check_mat(linalg::pinv(linalg::zeros(2, 3)), linalg::zeros(3, 2), 0.0, "pinv of zero");

    var g = rng { 20 };
    for n in SIZES {
        for shape in 3 {
            let rows = if shape == 1 { n + 4 } else { n };
            let cols = if shape == 2 { n + 4 } else { n };
            let a = linalg::random(rows, cols, g);
            let k = min(rows, cols);
            let t = tol(rows + cols) * 10.0;
            let what = str(rows, "x", cols);
            check(linalg::svd(a, u, s, v), str("converges ", what));
            check(u.rows == rows && u.cols == k && v.rows == cols && v.cols == k && s.len == k, str("thin shapes ", what));
            var ordered = true;
            for i in k {
                ordered = ordered && s[i] >= 0.0;
                if i > 0 { ordered = ordered && s[i - 1] >= s[i]; }
            }
            check(ordered, str("descending and non-negative ", what));
            check(orthonormal_cols(u, t), str("u^T u = I ", what));
            check(orthonormal_cols(v, t), str("v^T v = I ", what));
            check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), a, t, str("u s v^T = a ", what));
            // Singular values squared are the eigenvalues of a^T a (or a a^T).
            let gram = if rows >= cols { linalg::matmul(linalg::transpose(a), a) } else { linalg::matmul(a, linalg::transpose(a)) };
            let ev = linalg::eigvals_sym(gram);
            var sq2: f64[>..] = [];
            for i in k { sq2.push(s[k - 1 - i] * s[k - 1 - i]); }
            check_vec(sq2, ev, t * 10.0, str("s^2 = eig(a^T a) ", what));
            check_close(linalg::norm_fro(a), linalg::vnorm(s), t, str("norm_fro = |s| ", what));
            check(linalg::rank(a) == k, str("random matrices have full rank ", what));

            // The Moore-Penrose conditions.
            let p = linalg::pinv(a);
            check(p.rows == cols && p.cols == rows, str("pinv shape ", what));
            let ap = linalg::matmul(a, p);
            let pa = linalg::matmul(p, a);
            let c = s[0] / s[k - 1];
            check_mat(linalg::matmul(ap, a), a, t * c, str("a p a = a ", what));
            check_mat(linalg::matmul(pa, p), p, t * c * c, str("p a p = p ", what));
            check(linalg::is_symmetric(ap, t * c), str("a p symmetric ", what));
            check(linalg::is_symmetric(pa, t * c), str("p a symmetric ", what));
        }
        // Rank-deficient: two outer products in an n+2 square.
        if n >= 2 {
            let m = n + 2;
            let x1 = linalg::col(linalg::random(m, 1, g), 0);
            let y1 = linalg::col(linalg::random(m, 1, g), 0);
            let x2 = linalg::col(linalg::random(m, 1, g), 0);
            let y2 = linalg::col(linalg::random(m, 1, g), 0);
            let d = linalg::add(linalg::outer(x1, y1), linalg::outer(x2, y2));
            let t = tol(m) * 10.0;
            check(linalg::rank(d) == 2, str("rank 2 of ", m, "x", m));
            check(linalg::svd(d, u, s, v), str("rank-deficient converges, n=", m));
            check(orthonormal_cols(u, t), str("rank-deficient u orthonormal, n=", m));
            check(orthonormal_cols(v, t), str("rank-deficient v orthonormal, n=", m));
            check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), d, t, str("rank-deficient u s v^T = a, n=", m));
            let p = linalg::pinv(d);
            check_mat(linalg::matmul(linalg::matmul(d, p), d), d, t * 100.0, str("rank-deficient a p a = a, n=", m));
        }
    }
}

// Sizes past the blocked algorithms' panel width (32): one column over a
// block, exactly two blocks, three blocks plus a remainder, and one outer
// block (256) plus a remainder.
// eigh past EIGH_DC_MIN rows is divide and conquer: spectra that make it
// deflate, rotate close pairs apart, and solve secular equations whose
// roots crowd their poles.
fn test_eigh_dc() {
    group("eigh, divide and conquer");
    fn tridiag(dg: f64[:], off: f64[:]) -> linalg::matrix {
        let n = dg.len;
        var a = linalg::zeros(n, n);
        for i in n {
            linalg::set(a, i, i, dg[i]);
            if i > 0 {
                linalg::set(a, i, i - 1, off[i]);
                linalg::set(a, i - 1, i, off[i]);
            }
        }
        return a;
    }
    fn check_eigh(a, want: f64[:], what: const u8[:]) {
        let n = a.rows;
        var w: f64[>..] = [];
        var v = linalg::zeros(0, 0);
        let ok = linalg::eigh(a, w, v);
        // n eps |a|, with room: the worst measured was under one unit.
        let unit = 10.0 * n * linalg::EPS * max(linalg::max_abs(a), 1.0e-300);
        var sorted = true;
        for i in 1..n { sorted = sorted && w[i - 1] <= w[i]; }
        check(ok && sorted, str(what, ": converges, ascending"));
        check(orthonormal_cols(v, 10.0 * n * linalg::EPS), str(what, ": v^T v = I"));
        check_mat(linalg::matmul(a, v), linalg::matmul(v, linalg::diag(w)), unit, str(what, ": a v = v diag(w)"));
        if want.len == n {
            var ws: f64[>..] = [];
            ws.append(want);
            sort(ws);
            check_vec(w, ws, unit, str(what, ": eigenvalues"));
        }
    }
    let none: f64[>..] = [];
    var g = rng { 29 };
    // Wilkinson's W+ : pairs of eigenvalues agreeing to working precision.
    for m in [50, 100] {
        var dg: f64[>..] = [];
        var off: f64[>..] = [];
        for i in 2 * m + 1 {
            dg.push(abs(m - i) * 1.0);
            off.push(1.0);
        }
        check_eigh(tridiag(dg, off), none, str("W", 2 * m + 1, "+"));
    }
    // The path graph's Laplacian, known exactly.
    {
        let n = 200;
        var dg: f64[>..] = [];
        var off: f64[>..] = [];
        var want: f64[>..] = [];
        for i in n {
            dg.push(2.0);
            off.push(-1.0);
            want.push(2.0 - 2.0 * cos((i + 1) * PI / (n + 1)));
        }
        check_eigh(tridiag(dg, off), want, "Laplacian n=200");
    }
    // Zero off-diagonals split the problem; tiny ones nearly do.
    {
        var dg: f64[>..] = [];
        var off: f64[>..] = [];
        var dg2: f64[>..] = [];
        var off2: f64[>..] = [];
        for i in 150 {
            dg.push((i % 7) * 1.0);
            off.push(if i % 13 == 0 { 0.0 } else { 0.5 });
            dg2.push(i * 1.0);
            off2.push(1.0e-12);
        }
        check_eigh(tridiag(dg, off), none, "zero off-diagonals");
        check_eigh(tridiag(dg2, off2), none, "tiny off-diagonals");
    }
    var ones: f64[>..] = [];
    for i in 100 { ones.push(1.0); }
    check_eigh(linalg::identity(100), ones, "identity n=100");
    // q diag(lam) q^T: clusters, a graded spectrum, rank one.
    for spec in 4 {
        let n = 130;
        var lam: f64[>..] = [];
        for i in n {
            lam.push(if spec == 0 {
                if i < 65 { 1.0 } else { (i - 64) * 0.37 }
            } else if spec == 1 {
                1.0 + i * 1.0e-12
            } else if spec == 2 {
                pow(10.0, -i * 15.0 / n)
            } else {
                if i == 0 { 1.0 } else { 0.0 }
            });
        }
        var q = linalg::zeros(0, 0);
        var r = linalg::zeros(0, 0);
        linalg::qr(linalg::random(n, n, g), q, r);
        let a = linalg::matmul(linalg::matmul(q, linalg::diag(lam)), linalg::transpose(q));
        let names = ["65 equal eigenvalues", "all within 2e-10", "graded 1 to 1e-15", "rank one"];
        check_eigh(linalg::scale(linalg::add(a, linalg::transpose(a)), 0.5), lam, names[spec]);
    }
    // Far from 1 in scale both ways.
    let s = random_symmetric(120, g);
    check_eigh(linalg::scale(s, 1.0e150), none, "scaled by 1e150");
    check_eigh(linalg::scale(s, 1.0e-150), none, "scaled by 1e-150");
    // Either side of the switch from QL.
    for n in [linalg::EIGH_DC_MIN, linalg::EIGH_DC_MIN + 1] {
        check_eigh(random_symmetric(n, g), none, str("random n=", n));
    }
}

// svd past SVD_DC_MIN columns is divide and conquer on the bidiagonal:
// spectra that deflate, crowd, or reach zero, both shapes, and a matrix
// large enough for the blocked bidiagonalization.
fn test_svd_dc() {
    group("svd, divide and conquer");
    fn check_svd(a, want: f64[:], what: const u8[:]) {
        let m = a.rows;
        let n = a.cols;
        let k = min(m, n);
        var u = linalg::zeros(0, 0);
        var s: f64[>..] = [];
        var v = linalg::zeros(0, 0);
        let ok = linalg::svd(a, u, s, v);
        // n eps |a|, with room: the worst measured was under three units.
        let nn = max(m, n);
        let unit = 10.0 * nn * linalg::EPS * max(linalg::max_abs(a), 1.0e-300);
        var ordered = true;
        for i in k { ordered = ordered && s[i] >= 0.0 && (i == 0 || s[i - 1] >= s[i]); }
        check(ok && ordered, str(what, ": converges, descending, non-negative"));
        check(orthonormal_cols(u, 10.0 * nn * linalg::EPS), str(what, ": u^T u = I"));
        check(orthonormal_cols(v, 10.0 * nn * linalg::EPS), str(what, ": v^T v = I"));
        check_mat(linalg::matmul(linalg::matmul(u, linalg::diag(s)), linalg::transpose(v)), a, unit, str(what, ": u s v^T = a"));
        if want.len == k {
            var ws: f64[>..] = [];
            ws.append(want);
            sort(ws) { x, y => x > y };
            check_vec(s, ws, unit, str(what, ": singular values"));
        }
    }
    fn orthogonal(n: i64, g: rng&) -> linalg::matrix {
        var q = linalg::zeros(0, 0);
        var r = linalg::zeros(0, 0);
        linalg::qr(linalg::random(n, n, g), q, r);
        return q;
    }
    let none: f64[>..] = [];
    var g = rng { 31 };
    for spec in 5 {
        let n = 110;
        var sig: f64[>..] = [];
        for i in n {
            sig.push(if spec == 0 {
                if i < 55 { 1.0 } else { 0.0 }
            } else if spec == 1 {
                if i < 55 { 2.0 } else { (i - 54) * 0.01 }
            } else if spec == 2 {
                pow(10.0, -i * 15.0 / n)
            } else if spec == 3 {
                if i == 0 { 1.0 } else { 0.0 }
            } else {
                (i / 10) * 1.0
            });
        }
        let a = linalg::matmul(linalg::matmul(orthogonal(n, g), linalg::diag(sig)), linalg::transpose(orthogonal(n, g)));
        let names = ["rank 55", "55 equal", "graded 1 to 1e-15", "rank one", "11 clusters of 10, one at 0"];
        check_svd(a, sig, names[spec]);
    }
    // Zeros straight on the bidiagonal.
    {
        let n = 120;
        var b = linalg::zeros(n, n);
        for i in n {
            linalg::set(b, i, i, if i % 9 == 0 { 0.0 } else { 1.0 + i * 0.01 });
            if i + 1 < n { linalg::set(b, i, i + 1, if i % 11 == 0 { 0.0 } else { 0.5 }); }
        }
        check_svd(b, none, "bidiagonal with zeros");
    }
    check_svd(linalg::zeros(90, 70), none, "zero matrix");
    check_svd(linalg::random(200, 90, g), none, "tall 200x90");
    check_svd(linalg::random(80, 170, g), none, "wide 80x170");
    let r = linalg::random(100, 90, g);
    check_svd(linalg::scale(r, 1.0e150), none, "scaled by 1e150");
    check_svd(linalg::scale(r, 1.0e-150), none, "scaled by 1e-150");
    // Past BRD_BLOCKED_MIN entries: the blocked bidiagonalization.
    check_svd(linalg::random(420, 390, g), none, "420x390, blocked reduction");
    // Either side of the switch from QR.
    for n in [linalg::SVD_DC_MIN, linalg::SVD_DC_MIN + 1] {
        check_svd(linalg::random(n + 3, n, g), none, str("random ", n + 3, "x", n));
    }
}

fn test_blocked() {
    group("blocked sizes");
    var g = rng { 23 };
    for n in [33, 64, 97, 300] {
        let t = 1.0e-13 * n;
        let a = linalg::random(n, n, g);
        // p a = l u.
        var lu = linalg::clone(a);
        var piv: i64[>..] = [];
        let sign, ok = linalg::lu_factor(lu, piv);
        check(ok, str("lu ok, n=", n));
        var pm = linalg::zeros(n, n);
        var l = linalg::identity(n);
        var u = linalg::zeros(n, n);
        for i in n {
            linalg::set(pm, i, piv[i], 1.0);
            for j in n {
                if j < i { linalg::set(l, i, j, linalg::at(lu, i, j)); } else { linalg::set(u, i, j, linalg::at(lu, i, j)); }
            }
        }
        check_mat(linalg::matmul(pm, a), linalg::matmul(l, u), t, str("p a = l u, n=", n));
        check_close(linalg::det(pm), sign, 0.0, str("sign is det(p), n=", n));
        var bounded = true;
        for i in n { for j in i { bounded = bounded && abs(linalg::at(l, i, j)) <= 1.0; } }
        check(bounded, str("partial pivoting keeps |l| <= 1, n=", n));
        // Inverse and multi-RHS solve.
        let c = linalg::cond(a);
        let inv, iok = linalg::inverse(a);
        check(iok, str("inverse ok, n=", n));
        check_mat(linalg::matmul(a, inv), linalg::identity(n), t * c, str("a inv = I, n=", n));
        check_mat(linalg::matmul(inv, a), linalg::identity(n), t * c, str("inv a = I, n=", n));
        let bm = linalg::random(n, 5, g);
        let xm, mok = linalg::solve_matrix(a, bm);
        check(mok, str("solve_matrix ok, n=", n));
        check_mat(linalg::matmul(a, xm), bm, t * c, str("solve_matrix residual, n=", n));
        let rhs = linalg::col(bm, 0);
        let x, sok = linalg::solve(a, rhs);
        check(sok, str("solve ok, n=", n));
        check_vec(linalg::matvec(a, x), rhs, t * c, str("solve residual, n=", n));
        // A singular matrix whose zero column sits past the first block.
        var sing = linalg::clone(a);
        for i in n { linalg::set(sing, i, n - 1, 0.0); }
        let sx, sing_ok = linalg::solve(sing, rhs);
        check(!sing_ok, str("singular past the first block, n=", n));
        // Cholesky.
        var spd = linalg::matmul(a, linalg::transpose(a));
        linalg::add_scaled(spd, n * 1.0, linalg::identity(n));
        let f, fok = linalg::cholesky(spd);
        check(fok, str("cholesky ok, n=", n));
        check(is_lower(f), str("cholesky factor lower, n=", n));
        check_mat(linalg::matmul(f, linalg::transpose(f)), spd, t * n, str("l l^T = a, n=", n));
        var notpd = linalg::clone(spd);
        linalg::set(notpd, n - 1, n - 1, -1.0);
        let nf, nfok = linalg::cholesky(notpd);
        check(!nfok, str("not positive definite in the last block, n=", n));
        // QR, tall and wide.
        var q = linalg::zeros(0, 0);
        var r = linalg::zeros(0, 0);
        for shape in 2 {
            let m = if shape == 0 { linalg::random(n + 7, n, g) } else { linalg::random(n, n + 7, g) };
            linalg::qr(m, q, r);
            check(orthonormal_cols(q, t), str("q orthogonal, ", m.rows, "x", m.cols));
            check(is_upper(r), str("r upper, ", m.rows, "x", m.cols));
            check_mat(linalg::matmul(q, r), m, t, str("q r = a, ", m.rows, "x", m.cols));
        }
        // Eigen and SVD.
        let sym = linalg::add(a, linalg::transpose(a));
        var w: f64[>..] = [];
        var v = linalg::zeros(0, 0);
        check(linalg::eigh(sym, w, v), str("eigh converges, n=", n));
        check(orthonormal_cols(v, t), str("eigh v orthonormal, n=", n));
        check_mat(linalg::matmul(sym, v), linalg::matmul(v, linalg::diag(w)), t * 10.0, str("eigh a v = v w, n=", n));
        var su = linalg::zeros(0, 0);
        var sv = linalg::zeros(0, 0);
        var ss: f64[>..] = [];
        let tall = linalg::random(n + 5, n, g);
        check(linalg::svd(tall, su, ss, sv), str("svd converges, n=", n));
        check(orthonormal_cols(su, t) && orthonormal_cols(sv, t), str("svd factors orthonormal, n=", n));
        check_mat(linalg::matmul(linalg::matmul(su, linalg::diag(ss)), linalg::transpose(sv)), tall, t * 10.0,
                  str("svd u s v^T = a, n=", n));
    }
}

// Read-only parameters take temporaries, so expressions nest.
fn test_composition() {
    group("composition");
    let a = linalg::from_rows(2, 2, [1, 2, 3, 4]);
    let r = linalg::trace(linalg::matmul(linalg::transpose(a), linalg::add(a, linalg::identity(2))));
    check(r == 35.0, "nested temporaries");
    check_close(linalg::det(linalg::matmul(a, linalg::inverse(a))), 1.0, 1.0e-14, "first result of a two-result call");
    var acc = linalg::identity(2);
    for i in 3 {
        let next = linalg::matmul(acc, a);
        acc = copy(next);
    }
    check_mat(acc, linalg::power(a, 3), 0.0, "accumulating by whole assignment");
}

fn main() {
    let t0 = clock();
    test_construction();
    test_access();
    test_elementwise();
    test_products();
    test_vectors();
    test_scalars();
    test_triangular();
    test_lu();
    test_cholesky();
    test_qr();
    test_eigh();
    test_svd();
    test_eigh_dc();
    test_svd_dc();
    test_blocked();
    test_composition();
    var note: u8[>..] = [];
    format(note, passed + failed, " checks, ", passed, " passed, ", failed, " failed (");
    format_flt(note, clock() - t0, 2);
    format(note, " s)");
    print(note);
    if failed > 0 { exit(1); }
}
