// Times det and solve alone, single-threaded, at several sizes, and prints
// GFLOP/s (2 n^3 / 3 flops for the factorization) and a hash of each result,
// so two builds can be checked for bit-identical results.
//
//   goose build bench/bench_lu.goose && bench/bench_lu

import std;
import os;
import linalg from "..";

let SIZES = [50, 100, 200, 400, 800, 1600];

fn report(op: const u8[:], n: i64, best: f64, h: u64) {
    var line: u8[>..] = [];
    format(line, op, " n=", n, ": ");
    format_flt(line, best * 1000.0, 4);
    format(line, " ms  ");
    format_flt(line, 2.0 * n * n * n / 3.0 / best * 1.0e-9, 1);
    format(line, " GFLOP/s  hash ", h);
    print(line);
}

fn hash(h: u64&, x: f64) { h = (h ^ ((x * 1.0e15) as i64 as u64)) * 1099511628211; }

fn main() {
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let b = linalg::col(linalg::random(n, 1, r), 0);
        var best = 1.0e30;
        var total = 0.0;
        var reps = 0;
        var h: u64 = 1469598103934665603;
        while reps < 3 || total < 0.5 {
            let t0 = clock();
            let d = linalg::det(a);
            let dt = clock() - t0;
            if reps == 0 { hash(h, d / abs(d)); hash(h, log(abs(d))); }
            best = min(best, dt);
            total += dt;
            reps++;
        }
        report("det", n, best, h);
        best = 1.0e30;
        total = 0.0;
        reps = 0;
        h = 1469598103934665603;
        while reps < 3 || total < 0.5 {
            let t0 = clock();
            let x, ok = linalg::solve(a, b);
            let dt = clock() - t0;
            if reps == 0 { for v in x { hash(h, v); } }
            best = min(best, dt);
            total += dt;
            reps++;
        }
        report("solve", n, best, h);
    }
}
