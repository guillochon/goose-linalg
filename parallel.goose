// linalg.parallel: multithreaded versions of linalg's heavy operations.
//
//   import linalg from "guillochon/goose-linalg@v0.2.0";
//   import linalg.parallel;
//
//   linalg::set_threads(6);              // optional; hardware_threads() by default
//   let c = linalg::matmul_mt(a, b);
//   let inv, ok = linalg::inverse_mt(a);
//
// Each _mt function is its serial namesake with the matrix-multiply kernel
// swapped for gemm_mt, which splits a product's rows between a pool of
// worker threads and the calling thread. The rest of every algorithm is
// unchanged and runs on the calling thread, and the results are bit for
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

// How many shares of products the pool has computed so far: zero while
// every product has stayed on the calling thread.
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

fn lu_factor_mt(a: matrix&, piv: i64[>..]&) -> f64, bool {
    if serial_here(a.rows) { return lu_factor(a, piv); }
    return lu_factor_with(a, piv, gemm_mt);
}

fn det_mt(a) -> f64 {
    if serial_here(a.rows) { return det(a); }
    return det_with(a, gemm_mt);
}

fn solve_mt(a, b: f64[:]) -> f64[>..], bool {
    if serial_here(a.rows) { return solve(a, b); }
    return solve_with(a, b, gemm_mt);
}

fn solve_matrix_mt(a, b) -> matrix, bool {
    if serial_here(a.rows) { return solve_matrix(a, b); }
    return solve_matrix_with(a, b, gemm_mt);
}

fn inverse_mt(a) -> matrix, bool {
    if serial_here(a.rows) { return inverse(a); }
    return inverse_with(a, gemm_mt);
}

fn cholesky_mt(a) -> matrix, bool {
    if serial_here(a.rows) { return cholesky(a); }
    return cholesky_with(a, gemm_mt);
}
