// Times eigh and svd alone, single-threaded, at several sizes, and prints a
// hash of each result, so two builds can be checked for bit-identical
// results.
//
//   goose build bench/bench_eig.goose && bench/bench_eig

import std;
import os;
import linalg from "..";

let SIZES = [100, 200, 400, 800];

fn hash(h: u64&, x: f64) { h = (h ^ ((x * 1.0e15) as i64 as u64)) * 1099511628211; }

fn report(op: const u8[:], n: i64, best: f64, h: u64) {
    var line: u8[>..] = [];
    format(line, op, " n=", n, ": ");
    format_flt(line, best * 1000.0, 3);
    format(line, " ms  hash ", h);
    print(line);
}

fn main() {
    var r = rng { 42 };
    for n in SIZES {
        let a = linalg::random(n, n, r);
        let sym = linalg::add(a, linalg::transpose(a));
        var best = 1.0e30;
        var total = 0.0;
        var reps = 0;
        var h: u64 = 1469598103934665603;
        while reps < 3 || total < 0.5 {
            var w: f64[>..] = [];
            var v = linalg::zeros(0, 0);
            let t0 = clock();
            let ok = linalg::eigh(sym, w, v);
            let dt = clock() - t0;
            if reps == 0 { for x in w { hash(h, x); } for x in v.data { hash(h, x); } }
            best = min(best, dt);
            total += dt;
            reps++;
        }
        report("eigh", n, best, h);
        best = 1.0e30;
        total = 0.0;
        reps = 0;
        h = 1469598103934665603;
        while reps < 3 || total < 0.5 {
            var u = linalg::zeros(0, 0);
            var v = linalg::zeros(0, 0);
            var s: f64[>..] = [];
            let t0 = clock();
            let ok = linalg::svd(a, u, s, v);
            let dt = clock() - t0;
            if reps == 0 { for x in s { hash(h, x); } for x in u.data { hash(h, x); } for x in v.data { hash(h, x); } }
            best = min(best, dt);
            total += dt;
            reps++;
        }
        report("svd", n, best, h);
    }
}
