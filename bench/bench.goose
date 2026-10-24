// Times linalg's operations at several sizes, for comparison with
// bench_lapack.py (NumPy on OpenBLAS). Each timing is the best of several
// runs; the results go to stdout and to the CSV named on the command line
// (bench_goose.csv by default).
//
//   goose build bench/bench.goose && bench/bench [out.csv]
//
// For production flags with MSVC, set CL first (run_pinned.ps1 does):
//   CL="/arch:AVX2 /fp:contract" goose build bench/bench.goose

import std;
import os;
import linalg from "..";

let SIZES = [50, 100, 200, 400];

var sink = 0.0;       // results feed this, so no computation is dead
var csv: u8[>..] = [];

// The best wall time of F over at least 3 runs and 0.3 s, or a single run
// past 5 s.
fn time_it<F>(op: const u8[:], n: i64) {
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
    format(csv, op, ",", n, ",", best, "\n");
    var line: u8[>..] = [];
    format(line, op, " n=", n, ": ");
    format_flt(line, best * 1000.0, 3);
    format(line, " ms (", reps, " runs)");
    print(line);
}

fn main() {
    csv.append("op,n,seconds\n");
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let b = linalg::random(n, n, r);
        let x = linalg::col(linalg::random(n, 1, r), 0);
        var spd = linalg::matmul(a, linalg::transpose(a));
        linalg::add_scaled(spd, n * 1.0, linalg::identity(n));
        let sym = linalg::add(a, linalg::transpose(a));

        time_it("matmul", n) { let c = linalg::matmul(a, b); sink += c.data[0]; };
        time_it("matvec", n) { let y = linalg::matvec(a, x); sink += y[0]; };
        time_it("solve", n) { let y, ok = linalg::solve(a, x); sink += y[0]; };
        time_it("det", n) { sink += linalg::det(a); };
        time_it("inverse", n) { let m, ok = linalg::inverse(a); sink += m.data[0]; };
        time_it("cholesky", n) { let l, ok = linalg::cholesky(spd); sink += l.data[0]; };
        time_it("qr", n) {
            var q = linalg::zeros(0, 0);
            var rr = linalg::zeros(0, 0);
            linalg::qr(a, q, rr);
            sink += q.data[0] + rr.data[0];
        };
        time_it("eigh", n) {
            var w: f64[>..] = [];
            var v = linalg::zeros(0, 0);
            linalg::eigh(sym, w, v);
            sink += w[0] + v.data[0];
        };
        time_it("svd", n) {
            var u = linalg::zeros(0, 0);
            var v = linalg::zeros(0, 0);
            var s: f64[>..] = [];
            linalg::svd(a, u, s, v);
            sink += s[0] + u.data[0];
        };
    }
    let argv = args();
    let path = if argv.len > 1 { argv[1] } else { "bench_goose.csv" };
    if !write_file(path, csv) { abort(str("cannot write ", path)); }
    print("checksum ", sink);
}
