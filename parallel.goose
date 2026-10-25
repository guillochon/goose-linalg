// linalg.parallel: multithreaded versions of linalg's heavy operations.
//
//   import linalg from "guillochon/goose-linalg@v0.3.3";
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
// operands and results are copied through typed queues, in parts small
// enough to stay cheap (MT_CHUNK). Copying is O(n^2) against the product's
// O(n^3), so it pays for large matrices; products below MT_MIN_FLOPS stay on
// the calling thread.
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

// One worker's share of a product, c += alpha a b, goes out as a header on
// the worker's own queue, then the operands in parts of at most MT_CHUNK
// numbers, row-major. Every worker needs the whole of one operand (b when
// the shares are rows of c, a when they are columns): it goes down a tree
// of the workers on a queue of each worker's own (mt_dup), each worker
// passing every part on to MT_FAN more as it arrives, so the calling thread
// sends it to MT_FAN workers rather than to all. The worker's own shares of
// the other operand and of c come straight from the calling thread on
// another queue of its own (mt_part). Each queue has one sender, so a worker
// reads its parts in the order they were sent. The share's rows of c come
// back in parts that say where they go, on one queue all the workers share.
struct mt_hdr<S> { quit: bool, by_rows: bool, me: i64, nw: i64, row0: i64, col0: i64, m: i64, n: i64, k: i64, alpha: f64 }
struct mt_part<S> { data: f64[] }
struct mt_dup<S> { data: f64[] }
struct mt_done { row0: i64, col0: i64, rows: i64, cols: i64, data: f64[] }

// A message much past half a megabyte is a fresh allocation from the
// operating system each time, whose pages fault in as it is filled: on the
// test machine a number cost 2.4-3 ns to cross a queue in messages of 512K
// numbers or more, against 0.2-0.4 ns in messages of 32K or fewer. Of 2K to
// 128K numbers a part, 8K was as fast as any at n = 1600 on 6 threads.
let MT_CHUNK = 8192;

// The rows of the duplicated operand a worker multiplies at a time as they
// arrive: a multiple of four. 32 and 128 ran alike.
let MT_STEP = 128;

// The tree's fan-out: 1 makes it a chain, each worker passing the operand
// on to the next. A chain ran up to 10% faster than 2 at n = 1600 on 6
// threads.
let MT_FAN = 1;

// A product smaller than this, in flops, stays on the calling thread.
let MT_MIN_FLOPS = 2.0e6;

// What a flop costs in gemm_acc, and what a number costs to cross a queue
// to a worker and back into place, in nanoseconds (an i5-12600K with
// AVX2). They decide only whether sharing a product out pays: with these,
// matmul_mt shares from about n = 190 on 6 threads (it measured faster than
// matmul from n = 150), and the LU's 256-deep updates at n = 400, which
// gained nothing, stay on the calling thread. At 0.5 they were shared, and
// solve_mt there ran 2-5% behind solve.
let FLOP_NS = 0.025;
let COPY_NS = 0.8;

// What a number costs the calling thread to send its parts out, in
// nanoseconds, for sizing its own share: half what the sends measured (1.2),
// since the workers spend about as long again receiving theirs.
let SEND_NS = 0.6;

// The pool: worker ids, slot w's at w, the thread count asked for (0: not
// yet asked), and how many shares have gone to workers so far.
var mt_pool: i64[>..];
var mt_threads = 0;
var mt_jobs = 0;
var mt_always = false;

// A worker keeps its operands from one share to the next, so receiving
// them reuses their storage. Its own parts come first; it then multiplies
// as the duplicated operand arrives, MT_STEP rows of it at a time: steps of
// the inner dimension with rows of b, rows of c with rows of a. Steps are a
// multiple of four from the share's start, so the result is one gemm_acc's
// bit for bit.
fn mt_worker_body<S>(marker: S) {
    var a = matrix { rows: 0, cols: 0, data: [] };
    var b = matrix { rows: 0, cols: 0, data: [] };
    var c = matrix { rows: 0, cols: 0, data: [] };
    loop {
        let h = qget<mt_hdr<S>>();
        guard !h.quit else { break; }
        if h.by_rows {
            mt_receive(marker, a, h.m, h.k);
            mt_receive(marker, c, h.m, h.n);
            mt_start_dup(b, h.k, h.n);
            var t0 = 0;
            while t0 < h.k {
                let t1 = min(t0 + MT_STEP, h.k);
                mt_receive_dup(marker, b, t1 * h.n, h.me, h.nw);
                gemm_acc(c, 0, 0, a, 0, t0, false, b, t0, 0, h.m, h.n, t1 - t0, h.alpha);
                t0 = t1;
            }
        } else {
            mt_receive(marker, b, h.k, h.n);
            mt_receive(marker, c, h.m, h.n);
            mt_start_dup(a, h.m, h.k);
            var r0 = 0;
            while r0 < h.m {
                let r1 = min(r0 + MT_STEP, h.m);
                mt_receive_dup(marker, a, r1 * h.k, h.me, h.nw);
                gemm_acc(c, r0, 0, a, r0, 0, false, b, 0, 0, r1 - r0, h.n, h.k, h.alpha);
                r0 = r1;
            }
        }
        let per = max(1, MT_CHUNK / max(h.n, 1));
        var r = 0;
        while r < h.m {
            let r1 = min(r + per, h.m);
            qput(mt_done { row0: h.row0 + r, col0: h.col0, rows: r1 - r, cols: h.n, data: c.data[r * h.n..r1 * h.n] });
            r = r1;
        }
    }
}

fn mt_receive<S>(marker: S, m: matrix&, rows: i64, cols: i64) {
    m.rows = rows;
    m.cols = cols;
    m.data.clear();
    while m.data.len < rows * cols {
        let p = qget<mt_part<S>>();
        m.data.append(p.data);
    }
}

fn mt_start_dup(m: matrix&, rows: i64, cols: i64) {
    m.rows = rows;
    m.cols = cols;
    m.data.clear();
}

// The operand every worker needs, from the tree, until m holds at least
// `upto` numbers of it, each part passed on to worker me's children as it
// arrives. Then every part already here as well, without waiting: a worker
// passes parts on only between its steps, so parts left waiting through a
// step would hold up every worker below it.
fn mt_receive_dup<S>(marker: S, m: matrix&, upto: i64, me: i64, nw: i64) {
    while m.data.len < upto {
        let p = qget<mt_dup<S>>();
        m.data.append(p.data);
        mt_dup_on(me, nw, p.data);
    }
    loop {
        let p, got = qpoll<mt_dup<S>>();
        guard got else { break; }
        m.data.append(p.data);
        mt_dup_on(me, nw, p.data);
    }
}

// Worker me's children in the tree: (me + 1) MT_FAN + i for i < MT_FAN;
// the calling thread is me = -1, whose children are the first MT_FAN.
fn mt_dup_on(me: i64, nw: i64, data: f64[:]) {
    for i in MT_FAN {
        let kid = (me + 1) * MT_FAN + i;
        if kid < nw { mt_dup_send(kid, data); }
    }
}

thread_fn mt_w0() { mt_worker_body(lu_s0 {}); }
thread_fn mt_w1() { mt_worker_body(lu_s1 {}); }
thread_fn mt_w2() { mt_worker_body(lu_s2 {}); }
thread_fn mt_w3() { mt_worker_body(lu_s3 {}); }
thread_fn mt_w4() { mt_worker_body(lu_s4 {}); }
thread_fn mt_w5() { mt_worker_body(lu_s5 {}); }
thread_fn mt_w6() { mt_worker_body(lu_s6 {}); }
thread_fn mt_w7() { mt_worker_body(lu_s7 {}); }
thread_fn mt_w8() { mt_worker_body(lu_s8 {}); }
thread_fn mt_w9() { mt_worker_body(lu_s9 {}); }
thread_fn mt_w10() { mt_worker_body(lu_s10 {}); }
thread_fn mt_w11() { mt_worker_body(lu_s11 {}); }
thread_fn mt_w12() { mt_worker_body(lu_s12 {}); }
thread_fn mt_w13() { mt_worker_body(lu_s13 {}); }
thread_fn mt_w14() { mt_worker_body(lu_s14 {}); }
thread_fn mt_w15() { mt_worker_body(lu_s15 {}); }

fn mt_part_put<S>(marker: S, data: f64[:]) { qput(mt_part<S> { data: data }); }
fn mt_dup_put<S>(marker: S, data: f64[:]) { qput(mt_dup<S> { data: data }); }
fn mt_hdr_put<S>(marker: S, quit: bool, by_rows: bool, me: i64, nw: i64, row0: i64, col0: i64,
                 m: i64, n: i64, k: i64, alpha: f64) {
    qput(mt_hdr<S> { quit: quit, by_rows: by_rows, me: me, nw: nw, row0: row0, col0: col0, m: m, n: n, k: k, alpha: alpha });
}

fn mt_part_send(w: i64, data: f64[:]) {
    if w == 0 { mt_part_put(lu_s0 {}, data); }
    else if w == 1 { mt_part_put(lu_s1 {}, data); }
    else if w == 2 { mt_part_put(lu_s2 {}, data); }
    else if w == 3 { mt_part_put(lu_s3 {}, data); }
    else if w == 4 { mt_part_put(lu_s4 {}, data); }
    else if w == 5 { mt_part_put(lu_s5 {}, data); }
    else if w == 6 { mt_part_put(lu_s6 {}, data); }
    else if w == 7 { mt_part_put(lu_s7 {}, data); }
    else if w == 8 { mt_part_put(lu_s8 {}, data); }
    else if w == 9 { mt_part_put(lu_s9 {}, data); }
    else if w == 10 { mt_part_put(lu_s10 {}, data); }
    else if w == 11 { mt_part_put(lu_s11 {}, data); }
    else if w == 12 { mt_part_put(lu_s12 {}, data); }
    else if w == 13 { mt_part_put(lu_s13 {}, data); }
    else if w == 14 { mt_part_put(lu_s14 {}, data); }
    else { mt_part_put(lu_s15 {}, data); }
}

fn mt_dup_send(w: i64, data: f64[:]) {
    if w == 0 { mt_dup_put(lu_s0 {}, data); }
    else if w == 1 { mt_dup_put(lu_s1 {}, data); }
    else if w == 2 { mt_dup_put(lu_s2 {}, data); }
    else if w == 3 { mt_dup_put(lu_s3 {}, data); }
    else if w == 4 { mt_dup_put(lu_s4 {}, data); }
    else if w == 5 { mt_dup_put(lu_s5 {}, data); }
    else if w == 6 { mt_dup_put(lu_s6 {}, data); }
    else if w == 7 { mt_dup_put(lu_s7 {}, data); }
    else if w == 8 { mt_dup_put(lu_s8 {}, data); }
    else if w == 9 { mt_dup_put(lu_s9 {}, data); }
    else if w == 10 { mt_dup_put(lu_s10 {}, data); }
    else if w == 11 { mt_dup_put(lu_s11 {}, data); }
    else if w == 12 { mt_dup_put(lu_s12 {}, data); }
    else if w == 13 { mt_dup_put(lu_s13 {}, data); }
    else if w == 14 { mt_dup_put(lu_s14 {}, data); }
    else { mt_dup_put(lu_s15 {}, data); }
}

fn mt_hdr_send(w: i64, quit: bool, by_rows: bool, nw: i64, row0: i64, col0: i64, m: i64, n: i64, k: i64, alpha: f64) {
    if w == 0 { mt_hdr_put(lu_s0 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 1 { mt_hdr_put(lu_s1 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 2 { mt_hdr_put(lu_s2 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 3 { mt_hdr_put(lu_s3 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 4 { mt_hdr_put(lu_s4 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 5 { mt_hdr_put(lu_s5 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 6 { mt_hdr_put(lu_s6 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 7 { mt_hdr_put(lu_s7 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 8 { mt_hdr_put(lu_s8 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 9 { mt_hdr_put(lu_s9 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 10 { mt_hdr_put(lu_s10 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 11 { mt_hdr_put(lu_s11 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 12 { mt_hdr_put(lu_s12 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 13 { mt_hdr_put(lu_s13 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else if w == 14 { mt_hdr_put(lu_s14 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
    else { mt_hdr_put(lu_s15 {}, quit, by_rows, w, nw, row0, col0, m, n, k, alpha); }
}

// Rows r..r + rows, columns cc..cc + cols of m, in parts: to worker w's
// own queue, or with dup down the tree to all nw workers. Whole rows of m
// go straight from its storage, others through buf.
fn mt_send_rows(w: i64, dup: bool, nw: i64, m, r: i64, cc: i64, rows: i64, cols: i64, buf: f64[>..]&) {
    let per = max(1, MT_CHUNK / max(cols, 1));
    var i = 0;
    while i < rows {
        let i1 = min(i + per, rows);
        if cc == 0 && cols == m.cols {
            let d = m.data[(r + i) * cols..(r + i1) * cols];
            if dup { mt_dup_on(-1, nw, d); } else { mt_part_send(w, d); }
        } else {
            buf.clear();
            for t in i..i1 { buf.append(row(m, r + t)[cc..cc + cols]); }
            if dup { mt_dup_on(-1, nw, buf[..]); } else { mt_part_send(w, buf[..]); }
        }
        i = i1;
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
    for w in mt_pool.len { mt_hdr_send(w, true, false, 0, 0, 0, 0, 0, 0, 0.0); }
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

// The pool's workers: threads() - 1 of them, at most MT_SLOTS.
fn mt_workers() -> i64 { min(threads() - 1, MT_SLOTS) }

// The pool's slots: each worker needs queue types of its own (see the
// owner-computes LU below), so there is a thread_fn and a marker per slot.
let MT_SLOTS = 16;

fn mt_ensure() {
    while mt_pool.len < mt_workers() {
        let w = mt_pool.len;
        if w == 0 { mt_pool.push(thread_spawn(mt_w0)); }
        else if w == 1 { mt_pool.push(thread_spawn(mt_w1)); }
        else if w == 2 { mt_pool.push(thread_spawn(mt_w2)); }
        else if w == 3 { mt_pool.push(thread_spawn(mt_w3)); }
        else if w == 4 { mt_pool.push(thread_spawn(mt_w4)); }
        else if w == 5 { mt_pool.push(thread_spawn(mt_w5)); }
        else if w == 6 { mt_pool.push(thread_spawn(mt_w6)); }
        else if w == 7 { mt_pool.push(thread_spawn(mt_w7)); }
        else if w == 8 { mt_pool.push(thread_spawn(mt_w8)); }
        else if w == 9 { mt_pool.push(thread_spawn(mt_w9)); }
        else if w == 10 { mt_pool.push(thread_spawn(mt_w10)); }
        else if w == 11 { mt_pool.push(thread_spawn(mt_w11)); }
        else if w == 12 { mt_pool.push(thread_spawn(mt_w12)); }
        else if w == 13 { mt_pool.push(thread_spawn(mt_w13)); }
        else if w == 14 { mt_pool.push(thread_spawn(mt_w14)); }
        else { mt_pool.push(thread_spawn(mt_w15)); }
    }
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
    let p = mt_workers() + 1;
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
    var buf: f64[>..] = [];
    // The calling thread keeps the first `own` rows (or columns) and the
    // workers share the rest, worker j taking rows from own + j share. The
    // calling thread sends before it multiplies, so it takes fewer: as many
    // as leave it finishing with the workers, by the send's estimated cost.
    // By rows, every share starts a multiple of four rows in.
    let total = if by_rows { m } else { n };
    let other = if by_rows { k + n } else { k + m };    // numbers per row (column) of a share
    let dup = if by_rows { k * n } else { m * k };
    let line_ns = 2.0 * (if by_rows { n } else { m }) * k * FLOP_NS;
    let send_ns = SEND_NS * (MT_FAN * dup + (total - total / p) * other);
    var own = max(0.0, (total - (p - 1.0) * send_ns / line_ns) / p) as i64;
    if by_rows { own = own / 4 * 4; }
    var share = (total - own + p - 2) / (p - 1);
    if by_rows { share = (share + 3) / 4 * 4; }
    let jobs = (total - own + share - 1) / share;
    var owed = 0;               // numbers of c still to come back
    // Each worker gets its header and its own parts, then the duplicated
    // operand goes down the tree.
    for j in jobs {
        let s0 = own + j * share;
        let s1 = min(s0 + share, total);
        if by_rows {
            mt_hdr_send(j, false, true, jobs, s0, 0, s1 - s0, n, k, alpha);
            mt_send_rows(j, false, jobs, a, ai + s0, aj, s1 - s0, k, buf);
            mt_send_rows(j, false, jobs, c, ci + s0, cj, s1 - s0, n, buf);
            owed += (s1 - s0) * n;
        } else {
            mt_hdr_send(j, false, false, jobs, 0, s0, m, s1 - s0, k, alpha);
            mt_send_rows(j, false, jobs, b, bi, bj + s0, k, s1 - s0, buf);
            mt_send_rows(j, false, jobs, c, ci, cj + s0, m, s1 - s0, buf);
            owed += m * (s1 - s0);
        }
    }
    if by_rows { mt_send_rows(0, true, jobs, b, bi, bj, k, n, buf); }
    else { mt_send_rows(0, true, jobs, a, ai, aj, m, k, buf); }
    // The calling thread's own share, while the workers run.
    if own > 0 {
        if by_rows { gemm_acc(c, ci, cj, a, ai, aj, trans_a, b, bi, bj, own, n, k, alpha); }
        else { gemm_acc(c, ci, cj, a, ai, aj, trans_a, b, bi, bj, m, own, k, alpha); }
    }
    mt_jobs += jobs;
    while owed > 0 {
        let d = qget<mt_done>();
        let w = d.cols;
        for i in d.rows {
            let dst = row(c, ci + d.row0 + i)[cj + d.col0..cj + d.col0 + w];
            let src = d.data[i * w..(i + 1) * w];
            assert(dst.len == w && src.len == w);
            for j in w { dst[j] = src[j]; }
        }
        owed -= d.rows * w;
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
// saves. On an i5-12600K it beat gemm_mt at n = 800 on 3, 4 and 6 threads
// and lost at 700 on 6; on 2, one worker never hid the panels (at n = 800
// to 1000 it ran 35% behind the serial code).
let LU_OC_MIN = 800;
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
    // Strips tall enough for gemm_mt to share.
    return cholesky_with(a, OB, gemm_mt);
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
struct lu_s8 {}
struct lu_s9 {}
struct lu_s10 {}
struct lu_s11 {}
struct lu_s12 {}
struct lu_s13 {}
struct lu_s14 {}
struct lu_s15 {}

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
//
// The matrix (rows x cols, row-major) follows its message in parts of at
// most MT_CHUNK numbers, on lu_part<S>, as gemm_mt's operands do and for the
// same reason; a worker passes each on down the tree as it arrives.
struct lu_msg<S> { kind: i64, o0: i64, o1: i64, swaps: i64[], dests: i64[], rows: i64, cols: i64 }
struct lu_part<S> { data: f64[] }
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
    var pm = matrix { rows: 0, cols: 0, data: [] };     // the message's matrix
    var kids: i64[>..] = [];
    loop {
        let m = qget<lu_msg<S>>();
        if m.kind == LU_INIT {
            s.rows = m.o0;
            s.cols = m.cols;
            s.data.clear();
            kids.clear();
            lu_receive(marker, s, m.rows * m.cols, kids);
            let runs = lu_runs(m.o0, me, nw);
            for r in runs.len / 2 { for c in runs[2 * r]..runs[2 * r + 1] { gcol.push(c); } }
            continue;
        }
        var q = 0;
        while m.dests[q] != me { q++; }
        kids.clear();
        for child in (q + 1) * LU_FAN..(q + 2) * LU_FAN {
            if child < m.dests.len { kids.push(m.dests[child]); }
        }
        for kid in kids { lu_send_hdr(kid, m.kind, m.o0, m.o1, m.swaps, m.dests, m.rows, m.cols); }
        pm.rows = m.rows;
        pm.cols = m.cols;
        pm.data.clear();
        lu_receive(marker, pm, m.rows * m.cols, kids);
        guard m.kind != LU_QUIT else { break; }
        if m.kind != LU_PANEL {
            loop {
                let t = qget<solve_task>();
                guard t.c0 >= 0 else { break; }
                var x = matrix { rows: t.x.rows, cols: t.x.cols, data: [] };
                x.data.append(t.x.data);
                solve_block(pm, x, m.kind == LU_INVERT, t.c0, t.c1);
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
            lu_outer_update(s, first, mid - first, pm, o0, 0, o0, o1, NB, gemm_acc);
            var c = first;
            while c < mid {
                var e = c + 1;
                while e < mid && gcol[e] == gcol[e - 1] + 1 { e++; }
                qput(lu_cols { col0: gcol[c], c: rows_tile(s, 0, c, n, e - c) });
                c = e;
            }
            first = mid;
        }
        if w > first { lu_outer_update(s, first, w - first, pm, o0, 0, o0, o1, NB, gemm_acc); }
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

// A message's matrix, `count` numbers of it, into m, each part passed on to
// the workers in kids as it arrives.
fn lu_receive<S>(marker: S, m: matrix&, count: i64, kids: i64[:]) {
    while m.data.len < count {
        let p = qget<lu_part<S>>();
        m.data.append(p.data);
        for kid in kids { lu_send_part(kid, p.data); }
    }
}

fn lu_hdr_put<S>(marker: S, kind: i64, o0: i64, o1: i64, swaps: i64[:], dests: i64[:], rows: i64, cols: i64) {
    qput(lu_msg<S> { kind: kind, o0: o0, o1: o1, swaps: swaps, dests: dests, rows: rows, cols: cols });
}

fn lu_part_put<S>(marker: S, data: f64[:]) { qput(lu_part<S> { data: data }); }

fn lu_send_hdr(w: i64, kind: i64, o0: i64, o1: i64, swaps: i64[:], dests: i64[:], rows: i64, cols: i64) {
    if w == 0 { lu_hdr_put(lu_s0 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 1 { lu_hdr_put(lu_s1 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 2 { lu_hdr_put(lu_s2 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 3 { lu_hdr_put(lu_s3 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 4 { lu_hdr_put(lu_s4 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 5 { lu_hdr_put(lu_s5 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else if w == 6 { lu_hdr_put(lu_s6 {}, kind, o0, o1, swaps, dests, rows, cols); }
    else { lu_hdr_put(lu_s7 {}, kind, o0, o1, swaps, dests, rows, cols); }
}

fn lu_send_part(w: i64, data: f64[:]) {
    if w == 0 { lu_part_put(lu_s0 {}, data); }
    else if w == 1 { lu_part_put(lu_s1 {}, data); }
    else if w == 2 { lu_part_put(lu_s2 {}, data); }
    else if w == 3 { lu_part_put(lu_s3 {}, data); }
    else if w == 4 { lu_part_put(lu_s4 {}, data); }
    else if w == 5 { lu_part_put(lu_s5 {}, data); }
    else if w == 6 { lu_part_put(lu_s6 {}, data); }
    else { lu_part_put(lu_s7 {}, data); }
}

// A message to worker w, then its matrix p in parts.
fn lu_send(w: i64, kind: i64, o0: i64, o1: i64, swaps: i64[:], dests: i64[:], p) {
    lu_send_hdr(w, kind, o0, o1, swaps, dests, p.rows, p.cols);
    var i = 0;
    while i < p.data.len {
        let i1 = min(i + MT_CHUNK, p.data.len);
        lu_send_part(w, p.data[i..i1]);
        i = i1;
    }
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

// lu_broadcast of rows r.., columns cc..cc + cols of m, sent part by part
// from m's rows rather than copied out whole first.
fn lu_broadcast_rows(nw: i64, kind: i64, o0: i64, o1: i64, swaps: i64[:], m, r: i64, cc: i64, rows: i64, cols: i64) {
    var dests: i64[>..] = [];
    for w in nw { dests.push(w); }
    let fan = min(LU_FAN, nw);
    for q in fan { lu_send_hdr(q, kind, o0, o1, swaps, dests, rows, cols); }
    let per = max(1, MT_CHUNK / max(cols, 1));
    var buf: f64[>..] = [];
    var i = 0;
    while i < rows {
        let i1 = min(i + per, rows);
        buf.clear();
        for t in i..i1 { buf.append(row(m, r + t)[cc..cc + cols]); }
        for q in fan { lu_send_part(q, buf[..]); }
        i = i1;
    }
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
            lu_broadcast_rows(nw, LU_PANEL, o0, o1, swaps, a, o0, o0, n - o0, o1 - o0);
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
