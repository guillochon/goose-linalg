// Tests for linalg.parallel. Threads need a real C compiler:
//
//   goose build test/test_parallel.goose && test/test_parallel
//
// gemm_mt shares a product's rows out in multiples of four, which group
// into gemm_acc's blocks exactly as one serial call does, so every _mt
// result must equal its serial namesake's bit for bit. The serial results
// are checked for correctness by test_linalg.goose; here each _mt one is
// checked for equality with them, across sizes on both sides of the
// threshold, odd shapes, and several thread counts.

import std;
import os;
import linalg from "..";
import linalg.parallel;

var passed = 0;
var failed = 0;

fn check(ok: bool, what: const u8[:]) {
    if ok { passed++; return; }
    failed++;
    print("FAIL ", what);
}

// Nothing left in any queue once an _mt call has returned: a stale message
// would be read by the next call.
fn left_over<T>(n: i64) -> i64 {
    var k = n;
    loop {
        let t, got = qpoll<T>();
        guard got else { break; }
        k++;
    }
    return k;
}

fn queues_empty() -> bool {
    var k = left_over<linalg::mt_job>(0);
    k = left_over<linalg::mt_done>(k);
    k = left_over<linalg::lu_cols>(k);
    k = left_over<linalg::solve_task>(k);
    k = left_over<linalg::solve_done>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s0>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s1>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s2>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s3>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s4>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s5>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s6>>(k);
    k = left_over<linalg::lu_msg<linalg::lu_s7>>(k);
    return k == 0;
}

// Bit for bit: same shape, and no entry differs at all.
fn same(a, b) -> bool { linalg::same_shape(a, b) && a.data == b.data }

fn run(threads: i64) {
    linalg::set_threads(threads);
    check(linalg::threads() == threads, str("threads() reports ", threads));
    var g = rng { (1000 + threads) as u64 };
    for n in [3, 33, 97, 130, 257, 300] {
        let what = str(" n=", n, " threads=", threads);
        let a = linalg::random(n, n, g);
        let b = linalg::random(n, n + 5, g);
        let before = linalg::parallel_jobs();
        check(same(linalg::matmul_mt(a, b), linalg::matmul(a, b)), str("matmul", what));
        // With the cost model bypassed, every product past MT_MIN_FLOPS goes
        // to the pool: one share per worker.
        let used = linalg::parallel_jobs() - before;
        let past = 2.0 * n * n * (n + 5) >= linalg::MT_MIN_FLOPS;
        check(if threads > 1 && past { used == threads - 1 } else { used == 0 }, str("matmul shares", what));
        let tall = linalg::random(n + 9, n, g);
        check(same(linalg::matmul_mt(tall, a), linalg::matmul(tall, a)), str("matmul tall", what));

        var lu1 = linalg::clone(a);
        var lu2 = linalg::clone(a);
        var p1: i64[>..] = [];
        var p2: i64[>..] = [];
        let s1, ok1 = linalg::lu_factor_mt(lu1, p1);

        let s2, ok2 = linalg::lu_factor(lu2, p2);
        check(ok1 == ok2 && s1 == s2 && p1 == p2 && same(lu1, lu2), str("lu_factor", what));
        var lu3 = linalg::clone(a);
        var p3: i64[>..] = [];
        let s3, ok3 = linalg::lu_factor_oc(lu3, p3);
        check(ok3 == ok2 && s3 == s2 && p3 == p2 && same(lu3, lu2), str("lu_factor_oc", what));
        check(linalg::det_mt(a) == linalg::det(a), str("det", what));
        let rhs = linalg::col(b, 0);
        let x1, xo1 = linalg::solve_mt(a, rhs);
        let x2, xo2 = linalg::solve(a, rhs);
        check(xo1 == xo2 && x1 == x2, str("solve", what));
        let m1, mo1 = linalg::solve_matrix_mt(a, b);
        let m2, mo2 = linalg::solve_matrix(a, b);
        check(mo1 == mo2 && same(m1, m2), str("solve_matrix", what));
        let i1, io1 = linalg::inverse_mt(a);
        let i2, io2 = linalg::inverse(a);
        check(io1 == io2 && same(i1, i2), str("inverse", what));

        var spd = linalg::matmul(a, linalg::transpose(a));
        linalg::add_scaled(spd, n * 1.0, linalg::identity(n));
        let c1, co1 = linalg::cholesky_mt(spd);
        let c2, co2 = linalg::cholesky(spd);
        check(co1 == co2 && same(c1, c2), str("cholesky", what));
        check(queues_empty(), str("no messages left over", what));
    }
}

fn main() {
    let t0 = clock();
    // Every product the pool can take goes to it, both splits included:
    // tall times square splits rows, square times wide splits columns.
    linalg::set_parallel_always(true);
    for t in [1, 2, 3, 6] { run(t); }
    // Again with the LU family owner-computes past one outer block, the
    // last past the worker slots it has.
    linalg::set_owner_computes_from(linalg::OB + 1);
    for t in [2, 3, 6, 10] { run(t); }
    linalg::set_owner_computes_from(0);
    linalg::set_parallel_always(false);

    // The cost model on its own: a large product pays for its copies, a
    // modest one does not.
    linalg::set_threads(6);
    var gm = rng { 11 };
    let big_a = linalg::random(700, 700, gm);
    let big_b = linalg::random(700, 700, gm);
    var before = linalg::parallel_jobs();
    check(same(linalg::matmul_mt(big_a, big_b), linalg::matmul(big_a, big_b)), "700 x 700 matmul, cost model");
    check(linalg::parallel_jobs() - before == 5, "a 700 x 700 matmul goes to the pool");
    let mid = linalg::random(150, 150, gm);
    before = linalg::parallel_jobs();
    let mm = linalg::matmul_mt(mid, mid);
    check(linalg::parallel_jobs() == before, "a 150 x 150 matmul stays on the calling thread");
    // Past one outer block (256), the factorizations' outer updates and
    // the solves' outer products are deep enough to share out, and at 1200
    // (four outer blocks and a remainder) they pay for their copies.
    let big = linalg::random(1200, 1200, gm);
    var lu1 = linalg::clone(big);
    var lu2 = linalg::clone(big);
    var p1: i64[>..] = [];
    var p2: i64[>..] = [];
    before = linalg::parallel_jobs();
    let s1, ok1 = linalg::lu_factor_mt(lu1, p1);
    check(linalg::parallel_jobs() > before, "a 1200 x 1200 lu_factor uses the pool");
    let s2, ok2 = linalg::lu_factor(lu2, p2);
    check(ok1 && ok2 && s1 == s2 && p1 == p2 && same(lu1, lu2), "lu_factor 1200, cost model");
    before = linalg::parallel_jobs();
    let i1, io1 = linalg::inverse_mt(big);
    check(linalg::parallel_jobs() > before, "a 1200 x 1200 inverse uses the pool");
    let i2, io2 = linalg::inverse(big);
    check(io1 && io2 && same(i1, i2), "inverse 1200, cost model");
    check(linalg::det_mt(big) == linalg::det(big), "det 1200, cost model");
    let sx1, so1 = linalg::solve_mt(big, linalg::col(big, 0));
    let sx2, so2 = linalg::solve(big, linalg::col(big, 0));
    check(so1 && so2 && sx1 == sx2, "solve 1200, cost model");
    let rhs = linalg::random(1200, 300, gm);
    let x1, xo1 = linalg::solve_matrix_mt(big, rhs);
    let x2, xo2 = linalg::solve_matrix(big, rhs);
    check(xo1 && xo2 && same(x1, x2), "solve_matrix 1200 x 300, cost model");
    // Fewer right-hand sides than tasks for the workers, owner-computes
    // anyway: the case that once left a message behind, a worker told to
    // quit before the factored matrix reached it down the tree.
    linalg::set_owner_computes_from(linalg::OB + 1);
    for m in [1, 40] {
        let narrow = linalg::random(1200, m, gm);
        let y1, yo1 = linalg::solve_matrix_mt(big, narrow);
        let y2, yo2 = linalg::solve_matrix(big, narrow);
        check(yo1 && yo2 && same(y1, y2), str("solve_matrix 1200 x ", m, ", owner-computes"));
        check(queues_empty(), str("no messages left over after solve_matrix 1200 x ", m));
    }
    linalg::set_owner_computes_from(0);
    var spd = linalg::matmul(big, linalg::transpose(big));
    linalg::add_scaled(spd, 1200.0, linalg::identity(1200));
    // cholesky's outer updates are a third of LU's work and sit near the
    // cost model's break-even, so only its result is checked here.
    let c1, co1 = linalg::cholesky_mt(spd);
    let c2, co2 = linalg::cholesky(spd);
    check(co1 && co2 && same(c1, c2), "cholesky 1200, cost model");
    linalg::set_parallel_always(true);
    for t in [2, 4] {
        linalg::set_threads(t);
        let l3, lo3 = linalg::inverse_mt(big);
        check(lo3 && same(l3, i2), str("inverse 1200, every product shared, threads=", t));
        let c3, co3 = linalg::cholesky_mt(spd);
        check(co3 && same(c3, c2), str("cholesky 1200, every product shared, threads=", t));
    }
    linalg::set_parallel_always(false);
    // A product over the threshold whose row count is not a multiple of
    // four, with a pool resized in between.
    linalg::set_parallel_always(true);
    linalg::set_threads(5);
    var g = rng { 7 };
    let a = linalg::random(301, 290, g);
    let b = linalg::random(290, 299, g);
    check(same(linalg::matmul_mt(a, b), linalg::matmul(a, b)), "matmul 301x290 * 290x299, 5 threads");
    linalg::stop_threads();
    check(same(linalg::matmul_mt(a, b), linalg::matmul(a, b)), "matmul after stop_threads restarts the pool");
    var note: u8[>..] = [];
    format(note, passed + failed, " checks, ", passed, " passed, ", failed, " failed (");
    format_flt(note, clock() - t0, 2);
    format(note, " s)");
    print(note);
    if failed > 0 { exit(1); }
}
