// Times matmul alone, single-threaded, at several sizes, and prints GFLOP/s.
//
//   goose build bench/bench_matmul.goose && bench/bench_matmul

import std;
import os;
import linalg from "..";

let SIZES = [50, 100, 200, 400, 800, 1600];

var sink = 0.0;

fn main() {
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let b = linalg::random(n, n, r);
        var best = 1.0e30;
        var total = 0.0;
        var reps = 0;
        while reps < 3 || total < 0.5 {
            let t0 = clock();
            let c = linalg::matmul(a, b);
            let dt = clock() - t0;
            sink += c.data[0];
            best = min(best, dt);
            total += dt;
            reps++;
        }
        var line: u8[>..] = [];
        format(line, "matmul n=", n, ": ");
        format_flt(line, best * 1000.0, 3);
        format(line, " ms  ");
        format_flt(line, 2.0 * n * n * n / best * 1.0e-9, 1);
        format(line, " GFLOP/s");
        print(line);
    }
    print("checksum ", sink);
}
