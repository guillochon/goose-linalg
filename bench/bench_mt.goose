// Times linalg's operations single-threaded and through linalg.parallel at
// large sizes, for comparison with bench_lapack.py run at the same sizes on
// one and on several OpenBLAS threads. Each timing is the best of several
// runs; the results go to stdout and to the CSV named on the command line
// (bench_goose_mt.csv by default), as op,n,threads,seconds.
//
//   goose build bench/bench_mt.goose && bench/bench_mt [out.csv] [threads]
//
// run_pinned.ps1 builds it with production flags and pins it to one
// logical CPU of each performance core.

import std;
import os;
import linalg from "..";
import linalg.parallel;

let SIZES = [400, 800, 1600];

var csv: u8[>..] = [];
var sink = 0.0;

fn time_it<F>(op: const u8[:], n: i64, threads: i64) {
    var best = 1.0e30;
    var total = 0.0;
    var reps = 0;
    while (reps < 3 && total < 5.0) || total < 0.3 {
        let t0 = clock();
        F();
        let dt = clock() - t0;
        best = min(best, dt);
        total += dt;
        reps++;
    }
    format(csv, op, ",", n, ",", threads, ",", best, "\n");
    var line: u8[>..] = [];
    format(line, op, " n=", n, " threads=", threads, ": ");
    format_flt(line, best * 1000.0, 3);
    format(line, " ms (", reps, " runs)");
    print(line);
}

fn main() {
    let argv = args();
    let path = if argv.len > 1 { argv[1] } else { "bench_goose_mt.csv" };
    var p = 6;
    if argv.len > 2 {
        let v, ok = parse_int(argv[2]);
        if ok { p = v; }
    }
    linalg::set_threads(p);
    csv.append("op,n,threads,seconds\n");
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let b = linalg::random(n, n, r);
        let x = linalg::col(b, 0);
        var spd = linalg::matmul(a, linalg::transpose(a));
        linalg::add_scaled(spd, n * 1.0, linalg::identity(n));

        time_it("matmul", n, 1) { let c = linalg::matmul(a, b); sink += c.data[0]; };
        time_it("matmul", n, p) { let c = linalg::matmul_mt(a, b); sink += c.data[0]; };
        time_it("det", n, 1) { sink += linalg::det(a); };
        time_it("det", n, p) { sink += linalg::det_mt(a); };
        time_it("solve", n, 1) { let y, ok = linalg::solve(a, x); sink += y[0]; };
        time_it("solve", n, p) { let y, ok = linalg::solve_mt(a, x); sink += y[0]; };
        time_it("solve_matrix", n, 1) { let y, ok = linalg::solve_matrix(a, b); sink += y.data[0]; };
        time_it("solve_matrix", n, p) { let y, ok = linalg::solve_matrix_mt(a, b); sink += y.data[0]; };
        time_it("inverse", n, 1) { let m, ok = linalg::inverse(a); sink += m.data[0]; };
        time_it("inverse", n, p) { let m, ok = linalg::inverse_mt(a); sink += m.data[0]; };
        time_it("cholesky", n, 1) { let l, ok = linalg::cholesky(spd); sink += l.data[0]; };
        time_it("cholesky", n, p) { let l, ok = linalg::cholesky_mt(spd); sink += l.data[0]; };
    }
    if !write_file(path, csv) { abort(str("cannot write ", path)); }
    print("checksum ", sink, ", shares to the pool ", linalg::parallel_jobs());
}
