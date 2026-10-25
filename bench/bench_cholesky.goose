// Times cholesky alone, single-threaded, at several sizes, and prints
// GFLOP/s (n^3 / 3 flops) and a hash of each factor, so two builds can be
// checked for bit-identical results.
//
//   goose build bench/bench_cholesky.goose && bench/bench_cholesky

import std;
import os;
import linalg from "..";

let SIZES = [50, 100, 200, 400, 800, 1600];

fn main() {
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        var spd = linalg::matmul(a, linalg::transpose(a));
        linalg::add_scaled(spd, n * 1.0, linalg::identity(n));
        var best = 1.0e30;
        var total = 0.0;
        var reps = 0;
        var h: u64 = 1469598103934665603;
        while reps < 3 || total < 0.5 {
            let t0 = clock();
            let l, ok = linalg::cholesky(spd);
            let dt = clock() - t0;
            if reps == 0 {
                for x in l.data { h = (h ^ ((x * 1.0e15) as i64 as u64)) * 1099511628211; }
            }
            best = min(best, dt);
            total += dt;
            reps++;
        }
        var line: u8[>..] = [];
        format(line, "cholesky n=", n, ": ");
        format_flt(line, best * 1000.0, 4);
        format(line, " ms  ");
        format_flt(line, 1.0 * n * n * n / 3.0 / best * 1.0e-9, 1);
        format(line, " GFLOP/s  hash ", h);
        print(line);
    }
}
