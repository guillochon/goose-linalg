// Two everyday uses of linalg: a least-squares polynomial fit, and the
// principal axes of a 2-D point cloud from its covariance matrix.
//
//   goose examples/fit.goose

import std;
import math;
import linalg from "..";

fn main() {
    // y = 0.5 x^2 - 2 x + 1, sampled with a little deterministic wobble.
    var design = linalg::zeros(20, 3);
    var ys: f64[>..] = [];
    for i in 20 {
        let x = i * 0.25;
        linalg::set(design, i, 0, x * x);
        linalg::set(design, i, 1, x);
        linalg::set(design, i, 2, 1.0);
        ys.push(0.5 * x * x - 2.0 * x + 1.0 + 0.01 * sin(7.0 * i));
    }
    let coef, ok = linalg::lstsq(design, ys);
    print("quadratic fit ", ok, ": ", coef, "  (true: [0.5, -2.0, 1.0])");
    print("condition number of the design matrix: ", linalg::cond(design));

    // An elongated cloud along the direction (2, 1).
    var r = rng { 1 };
    var pts = linalg::zeros(500, 2);
    for i in 500 {
        let along = 3.0 * (r.rand_flt() - 0.5);
        let across = 0.4 * (r.rand_flt() - 0.5);
        linalg::set(pts, i, 0, 2.0 * along / sqrt(5.0) - across / sqrt(5.0));
        linalg::set(pts, i, 1, along / sqrt(5.0) + 2.0 * across / sqrt(5.0));
    }
    let cov = linalg::scale(linalg::matmul(linalg::transpose(pts), pts), 1.0 / 500);
    var w: f64[>..] = [];
    var v = linalg::zeros(0, 0);
    linalg::eigh(cov, w, v);
    let major = linalg::col(v, 1);
    print("covariance ", cov);
    print("variances along the principal axes: ", w);
    print("major axis: ", major, "  (expected +-[0.894, 0.447])");
}
