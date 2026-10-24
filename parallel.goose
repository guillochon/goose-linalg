// linalg.parallel: multithreaded versions of linalg's heavy operations.
//
//   import linalg from "guillochon/goose-linalg@v0.2.2";
//   import linalg.parallel;
//
//   linalg::set_threads(6);              // optional; hardware_threads() by default
//   let c = linalg::matmul_mt(a, b);
//   let inv, ok = linalg::inverse_mt(a);
//
// Each _mt function is its serial namesake with the matrix-multiply kernel
// swapped for gemm_mt, which splits a product's rows between a pool of
// worker threads and the calling thread. The rest of every algorithm is
// unchanged and runs on the calling thread. From LU_OC_MIN rows up, the LU
// family factors owner-computes instead (lu_factor_oc, below): the trailing
// matrix's columns live on the workers. Either way the results are bit for
// bit the serial ones.
//
// Goose threads share no memory: a worker is a separate program, and
// operands and results are copied through typed queues. Copying is O(n^2)
// against the product's O(n^3), so it pays for large matrices; products
// below MT_MIN_FLOPS stay on the calling thread.
//
// A program that imports this module contains a thread_fn, so it has to be
// built with `goose build`: the in-process `goose run` cannot run threads.
// Programs that import only linalg are unaffected.
//
// Call the _mt functions from main's thread. They share one pool and its
// queues, kept in globals no worker may reach, so a thread_fn that calls
// them is a compile error rather than a race.

import std;

namespace linalg;

// A block of numbers, row-major: a flat value, so it can cross a queue.
struct tile { rows: i64, cols: i64, data: f64[] }

// One worker's share of a product: c += alpha a b, and where in the
// caller's c its rows and columns came from.
struct mt_job { row0: i64, col0: i64, alpha: f64, a: tile, b: tile, c: tile }
struct mt_done { row0: i64, col0: i64, c: tile }

// A product smaller than this, in flops, stays on the calling thread.
let MT_MIN_FLOPS = 2.0e6;

// What a flop costs in gemm_acc, and what a number costs to cross a queue
// to a worker and back into place, in nanoseconds (an i5-12600K with
// AVX2). They decide only whether sharing a product out pays.
let FLOP_NS = 0.025;
let COPY_NS = 2.0;

// The pool: worker ids, the thread count asked for (0: not yet asked),
// and how many shares have gone to workers so far.
var mt_pool: i64[>..];
var mt_threads = 0;
var mt_jobs = 0;
var mt_always = false;

thread_fn mt_worker() {
    loop {
        let job = qget<mt_job>();
        guard job.row0 >= 0 else { break; }
        var c = matrix { rows: job.c.rows, cols: job.c.cols, data: [] };
        c.data.append(job.c.data);
        gemm_acc(c, 0, 0, job.a, 0, 0, false, job.b, 0, 0, c.rows, c.cols, job.a.cols, job.alpha);
        qput(mt_done { row0: job.row0, col0: job.col0, c: tile { rows: c.rows, cols: c.cols, data: c.data[..] } });
    }
}

// The threads the _mt functions use, the calling thread included.
fn threads() -> i64 {
    if mt_threads > 0 { mt_threads } else { max(1, hardware_threads()) }
}

// Use n threads, the calling thread included. On a CPU with performance
// and efficiency cores, the number of performance cores is usually best.
fn set_threads(n: i64) {
    assert(n >= 1);
    stop_threads();
    mt_threads = n;
}

// Ends the pool's workers; the next _mt call starts them again.
fn stop_threads() {
    for i in mt_pool.len { qput(mt_job { row0: -1, col0: 0, alpha: 0.0, a: empty_tile(), b: empty_tile(), c: empty_tile() }); }
    for id in mt_pool { thread_wait(id); }
    mt_pool.clear();
}

// For testing: share out every product past MT_MIN_FLOPS, whether or not
// the copies pay, so small matrices exercise the parallel paths.
fn set_parallel_always(on: bool) { mt_always = on; }

// For testing: from n rows (past one outer block, any size works), the LU
// family factors owner-computes (on LU_OC_THREADS threads or more), and
// solve_matrix_mt and inverse_mt solve owner-computes however few
// right-hand sides there are. n <= 0 restores the defaults.
fn set_owner_computes_from(n: i64) {
    lu_oc_min = if n > 0 { n } else { LU_OC_MIN };
    solve_oc_min = if n > 0 { n } else { SOLVE_OC_MIN };
    solve_oc_any_width = n > 0;
}

// How many shares of work the workers have taken so far, products from
// the pool and panels sent to owner-computes workers: zero while every
// product has stayed on the calling thread.
fn parallel_jobs() -> i64 { mt_jobs }

fn empty_tile() -> tile { tile { rows: 0, cols: 0, data: [] } }

fn mt_ensure() {
    while mt_pool.len < threads() - 1 { mt_pool.push(thread_spawn(mt_worker)); }
}

// Rows r..r + rows, columns cc..cc + cols of m, as a tile: straight from a
// slice when they are whole rows, so the numbers are copied once.
fn rows_tile(m, r: i64, cc: i64, rows: i64, cols: i64) -> tile {
    if cc == 0 && cols == m.cols { return tile { rows: rows, cols: cols, data: m.data[r * cols..(r + rows) * cols] }; }
    return tile { rows: rows, cols: cols, data: sub_rows(m, r, cc, rows, cols) };
}

// Rows r..r + rows, columns cc..cc + cols of m.
fn sub_rows(m, r: i64, cc: i64, rows: i64, cols: i64) -> f64[>..] {
    var out: f64[>..] = [];
    for i in rows { out.append(row(m, r + i)[cc..cc + cols]); }
    return out;
}

// gemm_acc, shared out between the pool and the calling thread when the
// copies pay for themselves. Each worker gets a copy of the operands its
// share reads, so the split duplicates the smaller of a and b: rows of c
// when b is the smaller, columns otherwise. Row shares are a multiple of
// four rows, which group into gemm_acc's four-row blocks exactly as one call
// over all of them does, and a column share changes no sum at all: the
// result is bit for bit gemm_acc's either way.
fn gemm_mt(c: matrix&, ci: i64, cj: i64, a, ai: i64, aj: i64, trans_a: bool,
           b, bi: i64, bj: i64, m: i64, n: i64, k: i64, alpha: f64) {
    let p = threads();
    let flops = 2.0 * m * n * k;
    let by_rows = n <= m;
    // Numbers through the queues: p - 1 copies of the duplicated operand,
    // and for the rows (or columns) the workers take, their share of the
    // other operand and of c, there and back.
    let moved = if by_rows { (p - 1.0) * k * n + (m - m / p) * (k + 2.0 * n) }
                else { (p - 1.0) * m * k + (n - n / p) * (k + 2.0 * m) };
    let serial_ns = flops * FLOP_NS;
    let pays = mt_always || moved * COPY_NS + serial_ns / p <= 0.8 * serial_ns;
    // A transposed a is gemm_acc's alone: no _mt operation passes one.
    if p <= 1 || trans_a || flops < MT_MIN_FLOPS || !pays {
        gemm_acc(c, ci, cj, a, ai, aj, trans_a, b, bi, bj, m, n, k, alpha);
        return;
    }
    mt_ensure();
    var jobs = 0;
    if by_rows {
        let share = ((m + p - 1) / p + 3) / 4 * 4;
        var r0 = share;
        while r0 < m {
            let r1 = min(r0 + share, m);
            qput(mt_job {
                row0: r0,
                col0: 0,
                alpha: alpha,
                a: rows_tile(a, ai + r0, aj, r1 - r0, k),
                b: rows_tile(b, bi, bj, k, n),
                c: rows_tile(c, ci + r0, cj, r1 - r0, n),
            });
            jobs++;
            r0 = r1;
        }
        // The first share, here, while the workers run.
        gemm_acc(c, ci, cj, a, ai, aj, trans_a, b, bi, bj, min(share, m), n, k, alpha);
    } else {
        let share = (n + p - 1) / p;
        var c0 = share;
        while c0 < n {
            let c1 = min(c0 + share, n);
            qput(mt_job {
                row0: 0,
                col0: c0,
                alpha: alpha,
                a: rows_tile(a, ai, aj, m, k),
                b: rows_tile(b, bi, bj + c0, k, c1 - c0),
                c: rows_tile(c, ci, cj + c0, m, c1 - c0),
            });
            jobs++;
            c0 = c1;
        }
        gemm_acc(c, ci, cj, a, ai, aj, trans_a, b, bi, bj, m, min(share, n), k, alpha);
    }
    mt_jobs += jobs;
    for done in jobs {
        let d = qget<mt_done>();
        let w = d.c.cols;
        for i in d.c.rows {
            let dst = row(c, ci + d.row0 + i)[cj + d.col0..cj + d.col0 + w];
            let src = d.c.data[i * w..(i + 1) * w];
            assert(dst.len == w && src.len == w);
            for j in w { dst[j] = src[j]; }
        }
    }
}

// The multithreaded operations: each is its serial namesake with gemm_mt as
// the kernel of its large products. Up to one outer block (OB) a
// factorization has no product deep enough to share, and with one thread
// nothing is shared at all, so those cases run the serial code itself.
//
// qr, lstsq, eigh and svd have no _mt versions: their products are blocks
// of NB reflectors, too narrow ever to pay for their copies (see the
// README).
fn matmul_mt(a, b) -> matrix { return matmul_with(a, b, gemm_mt); }

fn serial_here(n: i64) -> bool { n <= OB || threads() <= 1 }

// From LU_OC_MIN rows, on at least LU_OC_THREADS threads, the owner-computes
// LU beats gemm_mt's: below that its serial panels outweigh the copies it
// saves. On an i5-12600K it overtook gemm_mt from n = 1000 on 3 and 4
// threads and from 1100 on 6; on 2, one worker never hid the panels (up
// to n = 1400 it ran 20% behind).
let LU_OC_MIN = 1100;
let LU_OC_THREADS = 3;
var lu_oc_min = LU_OC_MIN;

fn lu_factor_mt(a: matrix&, piv: i64[>..]&) -> f64, bool {
    if serial_here(a.rows) { return lu_factor(a, piv); }
    if a.rows >= lu_oc_min && threads() >= LU_OC_THREADS { return lu_factor_oc(a, piv); }
    return lu_factor_with(a, piv, gemm_mt);
}

fn det_mt(a) -> f64 {
    if serial_here(a.rows) { return det(a); }
    return det_with(a, lu_factor_mt);
}

fn solve_mt(a, b: f64[:]) -> f64[>..], bool {
    if serial_here(a.rows) { return solve(a, b); }
    return solve_with(a, b, lu_factor_mt);
}

// From SOLVE_OC_MIN rows, solve_matrix_mt and inverse_mt share their
// triangular solves owner-computes, by columns of the right-hand sides
// (lu_solve_oc), when there are at least half as many right-hand sides as
// rows (inverse always has as many). Otherwise they go through gemm_mt. On
// an i5-12600K, owner-computes won from n = 600 on 2, 3, 4 and 6 threads;
// with fewer right-hand sides, sending every worker the factored matrix
// cost more than it saved (at n = 1600 on 6 threads, 39 ms against 4 for
// 16 columns, and even at 512).
let SOLVE_OC_MIN = 600;
var solve_oc_min = SOLVE_OC_MIN;
var solve_oc_any_width = false;

fn solve_oc_here(n: i64, m: i64) -> bool {
    n >= solve_oc_min && (solve_oc_any_width || 2 * m >= n)
}

fn solve_matrix_mt(a, b) -> matrix, bool {
    if serial_here(a.rows) { return solve_matrix(a, b); }
    if !solve_oc_here(a.rows, b.cols) { return solve_matrix_with(a, b, lu_factor_mt, gemm_mt); }
    assert(a.rows == a.cols && a.rows == b.rows);
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor_mt(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    return lu_solve_oc(lu, piv, b, false), true;
}

fn inverse_mt(a) -> matrix, bool {
    if serial_here(a.rows) { return inverse(a); }
    if !solve_oc_here(a.rows, a.rows) { return inverse_with(a, lu_factor_mt, gemm_mt); }
    assert(a.rows == a.cols);
    var lu = clone(a);
    var piv: i64[>..] = [];
    let sign, ok = lu_factor_mt(lu, piv);
    guard ok else { return zeros(0, 0), false; }
    return lu_solve_oc(lu, piv, a, true), true;
}

fn cholesky_mt(a) -> matrix, bool {
    if serial_here(a.rows) { return cholesky(a); }
    return cholesky_with(a, gemm_mt);
}

// ---------------------------------------------------------------------------
// Owner-computes LU
//
// gemm_mt ships the whole trailing matrix out and back for every outer
// block, so its copies grow as n^3 / OB, like the flops. Here the columns
// past the first outer block live on the workers instead, dealt out in
// blocks of LU_DB columns round-robin, and come back only when their turn
// as a panel comes. Each outer step the calling thread factors the panel
// with the serial code and sends it, with its row swaps, down a tree of all
// the workers (lu_broadcast). Each worker that still owns columns applies the
// swaps and runs
// lu_outer_update on its own columns: those of the next panel first, which
// it sends back at once so the calling thread can factor them while the
// rest are updated. Copies are O(p n^2) in all, and the result is
// lu_factor's bit for bit, since lu_outer_update computes every column on
// its own.
//
// Goose has one queue per type, read by every worker, so a message for one
// worker needs a type of its own: lu_msg<S> for that worker's marker S. A
// thread_fn can't be generic, so each slot has a thread_fn of its own around
// the generic body, and LU_SLOTS bounds the workers.

// Columns are dealt out LU_DB at a time: narrow blocks keep the workers'
// shares even as the trailing matrix shrinks. The calling thread sends a
// panel to LU_FAN workers, and each passes it on to LU_FAN more. Both were
// the fastest of 32..256 and 1..5 at n = 1600 and 2400 on 6 threads.
let LU_DB = 32;
let LU_FAN = 2;
let LU_SLOTS = 8;
let LU_INIT = 0;
let LU_PANEL = 1;
let LU_QUIT = 2;
let LU_SOLVE = 3;
let LU_INVERT = 4;

struct lu_s0 {}
struct lu_s1 {}
struct lu_s2 {}
struct lu_s3 {}
struct lu_s4 {}
struct lu_s5 {}
struct lu_s6 {}
struct lu_s7 {}

// To one worker. LU_INIT: its columns as an n-row tile, o0 = n. LU_PANEL:
// the factored outer block o0..o1, rows o0.., and its row swaps, for the
// workers in dests: the panel goes down them as a tree, each worker passing
// it on to LU_FAN more before it starts its own work, so the calling thread
// sends it only LU_FAN times.
//
// The triangular solves (lu_solve_oc): LU_SOLVE, or LU_INVERT for the
// identity's columns, which each worker makes itself, carries the factored
// matrix down the tree as a panel goes. The worker then takes solve_tasks
// until one with c0 < 0.
struct lu_msg<S> { kind: i64, o0: i64, o1: i64, swaps: i64[], dests: i64[], p: tile }
// From a worker: columns col0.. of the next panel, every row.
struct lu_cols { col0: i64, c: tile }
// Columns c0..c1 of the right-hand sides, in pivot order (none for the
// identity's), for whichever thread takes them; and back, solved.
struct solve_task { c0: i64, c1: i64, x: tile }
struct solve_done { c0: i64, c1: i64, x: tile }

// The columns worker `me` of nw owns, as runs [c0, c1, c0, c1, ...].
fn lu_runs(n: i64, me: i64, nw: i64) -> i64[>..] {
    var runs: i64[>..] = [];
    var b = me;
    while OB + b * LU_DB < n {
        let c0 = OB + b * LU_DB;
        runs.push(c0);
        runs.push(min(c0 + LU_DB, n));
        b += nw;
    }
    return runs;
}

// The right-hand sides go out SOLVE_DB columns to a task: narrow enough to
// even out the threads' shares, wide enough for gemm_acc to run well. 128
// beat 32 and 64 by 10-25% at n = 1600 on 3 and 6 threads; 192 and 256
// were no better.
let SOLVE_DB = 128;

// Columns c0..c1 of the right-hand sides, x (with tri, the identity's, made
// here), through both triangular solves with lu: what serial
// solve_matrix_with and inverse_with do to each of them, bit for bit.
fn solve_block(lu, x: matrix&, tri: bool, c0: i64, c1: i64) {
    let n = lu.rows;
    var below: i64[>..] = [];
    if tri {
        x.rows = n;
        x.cols = c1 - c0;
        x.data.clear();
        x.data.resize(n * (c1 - c0), 0.0);
        for j in c1 - c0 { x.data[(c0 + j) * (c1 - c0) + j] = 1.0; }
        for r in n + 1 { below.push(max(0, min(r - c0, c1 - c0))); }
    }
    forward_unit_cols_with(lu, x, tri, below, gemm_acc);
    backward_with(lu, x, gemm_acc);
}

fn lu_worker_body<S>(marker: S, me: i64, nw: i64) {
    var s = matrix { rows: 0, cols: 0, data: [] };
    var gcol: i64[>..] = [];        // the column of a that each of s's is
    var first = 0;                  // s's columns before this have gone back
    loop {
        let m = qget<lu_msg<S>>();
        if m.kind == LU_INIT {
            s.rows = m.o0;
            s.cols = m.p.cols;
            s.data.append(m.p.data);
            let runs = lu_runs(m.o0, me, nw);
            for r in runs.len / 2 { for c in runs[2 * r]..runs[2 * r + 1] { gcol.push(c); } }
            continue;
        }
        var q = 0;
        while m.dests[q] != me { q++; }
        for child in (q + 1) * LU_FAN..(q + 2) * LU_FAN {
            if child < m.dests.len { lu_send(m.dests[child], m.kind, m.o0, m.o1, m.swaps, m.dests, m.p); }
        }
        guard m.kind != LU_QUIT else { break; }
        if m.kind != LU_PANEL {
            loop {
                let t = qget<solve_task>();
                guard t.c0 >= 0 else { break; }
                var x = matrix { rows: t.x.rows, cols: t.x.cols, data: [] };
                x.data.append(t.x.data);
                solve_block(m.p, x, m.kind == LU_INVERT, t.c0, t.c1);
                qput(solve_done { c0: t.c0, c1: t.c1, x: tile { rows: x.rows, cols: x.cols, data: x.data[..] } });
            }
            continue;
        }
        let n = s.rows;
        let w = s.cols;
        let o0 = m.o0;
        let o1 = m.o1;
        // The panel's row swaps, on the columns still here.
        lu_swap_rows(s, first, w, m.swaps, o0, o1);
        // The next panel's columns first, sent back as soon as they are done.
        let o2 = min(o1 + OB, n);
        var mid = first;
        while mid < w && gcol[mid] < o2 { mid++; }
        if mid > first {
            lu_outer_update(s, first, mid - first, m.p, o0, 0, o0, o1, NB, gemm_acc);
            var c = first;
            while c < mid {
                var e = c + 1;
                while e < mid && gcol[e] == gcol[e - 1] + 1 { e++; }
                qput(lu_cols { col0: gcol[c], c: rows_tile(s, 0, c, n, e - c) });
                c = e;
            }
            first = mid;
        }
        if w > first { lu_outer_update(s, first, w - first, m.p, o0, 0, o0, o1, NB, gemm_acc); }
    }
}

thread_fn lu_w0(me: i64, nw: i64) { lu_worker_body(lu_s0 {}, me, nw); }
thread_fn lu_w1(me: i64, nw: i64) { lu_worker_body(lu_s1 {}, me, nw); }
thread_fn lu_w2(me: i64, nw: i64) { lu_worker_body(lu_s2 {}, me, nw); }
thread_fn lu_w3(me: i64, nw: i64) { lu_worker_body(lu_s3 {}, me, nw); }
thread_fn lu_w4(me: i64, nw: i64) { lu_worker_body(lu_s4 {}, me, nw); }
thread_fn lu_w5(me: i64, nw: i64) { lu_worker_body(lu_s5 {}, me, nw); }
thread_fn lu_w6(me: i64, nw: i64) { lu_worker_body(lu_s6 {}, me, nw); }
thread_fn lu_w7(me: i64, nw: i64) { lu_worker_body(lu_s7 {}, me, nw); }

fn lu_spawn(me: i64, nw: i64) -> i64 {
    if me == 0 { return thread_spawn(lu_w0, me, nw); }
    if me == 1 { return thread_spawn(lu_w1, me, nw); }
    if me == 2 { return thread_spawn(lu_w2, me, nw); }
    if me == 3 { return thread_spawn(lu_w3, me, nw); }
    if me == 4 { return thread_spawn(lu_w4, me, nw); }
    if me == 5 { return thread_spawn(lu_w5, me, nw); }
    if me == 6 { return thread_spawn(lu_w6, me, nw); }
    return thread_spawn(lu_w7, me, nw);
}

fn lu_put<S>(marker: S, kind: i64, o0: i64, o1: i64, swaps: i64[:], dests: i64[:], p) {
    qput(lu_msg<S> { kind: kind, o0: o0, o1: o1, swaps: swaps, dests: dests, p: tile { rows: p.rows, cols: p.cols, data: p.data[..] } });
}

fn lu_send(me: i64, kind: i64, o0: i64, o1: i64, swaps: i64[:], dests: i64[:], p) {
    if me == 0 { lu_put(lu_s0 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 1 { lu_put(lu_s1 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 2 { lu_put(lu_s2 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 3 { lu_put(lu_s3 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 4 { lu_put(lu_s4 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 5 { lu_put(lu_s5 {}, kind, o0, o1, swaps, dests, p); }
    else if me == 6 { lu_put(lu_s6 {}, kind, o0, o1, swaps, dests, p); }
    else { lu_put(lu_s7 {}, kind, o0, o1, swaps, dests, p); }
}

// Sends a message to all nw workers down the tree: the calling thread to
// the first LU_FAN, each of those to LU_FAN more, and so on. The tree is the
// same for every message, so each worker hears everything from one sender,
// in the order it was sent: a worker can't get a panel ahead of the one
// before it, or LU_QUIT ahead of the last broadcast.
fn lu_broadcast(nw: i64, kind: i64, o0: i64, o1: i64, swaps: i64[:], p) {
    var dests: i64[>..] = [];
    for w in nw { dests.push(w); }
    for q in min(LU_FAN, nw) { lu_send(q, kind, o0, o1, swaps, dests, p); }
}

// lu_factor, owner-computes: the same factors bit for bit.
fn lu_factor_oc(a: matrix&, piv: i64[>..]&) -> f64, bool {
    assert(a.rows == a.cols);
    let n = a.rows;
    let nw = min(threads() - 1, LU_SLOTS);
    if n <= OB || nw < 1 { return lu_factor(a, piv); }
    var ids: i64[>..] = [];
    var last: i64[>..] = [];        // the last column each worker owns
    for w in nw {
        ids.push(lu_spawn(w, nw));
        let runs = lu_runs(n, w, nw);
        last.push(if runs.len > 0 { runs[runs.len - 1] - 1 } else { -1 });
        var d: f64[>..] = [];
        for i in n {
            let ri = row(a, i);
            for r in runs.len / 2 { d.append(ri[runs[2 * r]..runs[2 * r + 1]]); }
        }
        lu_send(w, LU_INIT, n, 0, [], [], tile { rows: n, cols: d.len / n, data: d[..] });
    }
    piv.clear();
    for i in n { piv.push(i); }
    var sign = 1.0;
    var ok = true;
    var swaps: i64[>..] = [];
    var o0 = 0;
    while o0 < n {
        let o1 = min(o0 + OB, n);
        // Columns from o1 on are stale here: their owners apply the swaps.
        let s, pok = lu_outer_panel(a, piv, o0, o1, NB, swaps);
        sign *= s;
        if !pok { ok = false; }
        if o1 < n {
            let p = rows_tile(a, o0, o0, n - o0, o1 - o0);
            lu_broadcast(nw, LU_PANEL, o0, o1, swaps, p);
            for w in nw { if last[w] >= o1 { mt_jobs++; } }
            // The next panel, whole columns, from their owners.
            let o2 = min(o1 + OB, n);
            var got = 0;
            while got < o2 - o1 {
                let d = qget<lu_cols>();
                let cw = d.c.cols;
                for i in n {
                    let dst = row(a, i)[d.col0..d.col0 + cw];
                    let src = d.c.data[i * cw..(i + 1) * cw];
                    assert(dst.len == cw && src.len == cw);
                    for j in cw { dst[j] = src[j]; }
                }
                got += cw;
            }
        }
        o0 = o1;
    }
    lu_broadcast(nw, LU_QUIT, 0, 0, [], empty_tile());
    for id in ids { thread_wait(id); }
    return sign, ok;
}

// The triangular solves of solve_matrix and inverse, owner-computes: the
// right-hand sides' columns are independent, so each worker and the calling
// thread take a share of them, the workers getting the factored matrix once
// down the tree. With tri, the right-hand sides are the identity, which each
// owner makes for itself, and the result is a's inverse: the solved columns
// go back in piv's order. Otherwise b's rows go out in piv's order and the
// result is x with a x = b. Either is the serial result bit for bit.
// The triangular solves of solve_matrix and inverse, owner-computes: the
// right-hand sides' columns are independent, so they go out as tasks of
// SOLVE_DB columns, which the workers take as soon as they have the
// factored matrix (sent once, down the tree) and the calling thread takes
// meanwhile, so a worker that starts late takes fewer. With tri, the
// right-hand sides are the identity, which each task makes for itself, and
// the result is a's inverse: the solved columns go back in piv's order.
// Otherwise b's rows go out in piv's order and the result is x with
// a x = b. Either is the serial result bit for bit.
fn lu_solve_oc(lu, piv: i64[:], b, tri: bool) -> matrix {
    let n = lu.rows;
    let m = if tri { n } else { b.cols };
    let nw = min(threads() - 1, LU_SLOTS);
    var ids: i64[>..] = [];
    for w in nw { ids.push(lu_spawn(w, nw)); }
    lu_broadcast(nw, if tri { LU_INVERT } else { LU_SOLVE }, 0, 0, [], lu);
    var tasks = 0;
    var c0 = 0;
    while c0 < m {
        let c1 = min(c0 + SOLVE_DB, m);
        if tri {
            qput(solve_task { c0: c0, c1: c1, x: empty_tile() });
        } else {
            var d: f64[>..] = [];
            for i in n { d.append(row(b, piv[i])[c0..c1]); }
            qput(solve_task { c0: c0, c1: c1, x: tile { rows: n, cols: c1 - c0, data: d[..] } });
        }
        tasks++;
        c0 = c1;
    }
    for w in nw { qput(solve_task { c0: -1, c1: -1, x: empty_tile() }); }
    var out = zeros(n, m);
    // The calling thread's tasks, until it meets the first end marker
    // (every task is ahead of them), which it leaves for its worker.
    var mine = 0;
    loop {
        let t, got = qpoll<solve_task>();
        guard got else { break; }
        if t.c0 < 0 {
            qput(t);
            break;
        }
        var x = matrix { rows: t.x.rows, cols: t.x.cols, data: [] };
        x.data.append(t.x.data);
        solve_block(lu, x, tri, t.c0, t.c1);
        solve_place(out, x, t.c0, piv, tri);
        mine++;
    }
    mt_jobs += tasks - mine;
    for k in tasks - mine {
        let d = qget<solve_done>();
        solve_place(out, d.x, d.c0, piv, tri);
    }
    lu_broadcast(nw, LU_QUIT, 0, 0, [], empty_tile());
    for id in ids { thread_wait(id); }
    return out;
}

// Solved columns c0.. of the right-hand sides, x, into their places in out:
// column c is column c of the solution, or with tri column piv[c] of the
// inverse.
fn solve_place(out: matrix&, x, c0: i64, piv: i64[:], tri: bool) {
    let w = x.cols;
    for i in out.rows {
        let xi = x.data[i * w..(i + 1) * w];
        let oi = row(out, i)[c0..c0 + w];
        assert(xi.len == w && oi.len == w);
        if tri {
            let full = row(out, i);
            for j in w { full[piv[c0 + j]] = xi[j]; }
        } else {
            for j in w { oi[j] = xi[j]; }
        }
    }
}
