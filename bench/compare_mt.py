"""The multithreaded comparison: a table and compare_mt.png.

    python compare_mt.py      (in this directory, after run_pinned.ps1)

Reads bench_goose_mt.csv (Goose on 1 thread and through linalg.parallel)
and bench_lapack_mt1.csv / bench_lapack_mt.csv (OpenBLAS on 1 thread and
on as many threads as Goose used), all as op,n,threads,seconds.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, threads=None):
    # threads: for a CSV without a threads column, the count it was run with.
    with open(path) as f:
        return {(r["op"], int(r["n"]), int(r.get("threads") or threads)): float(r["seconds"])
                for r in csv.DictReader(f)}


goose = load("bench_goose_mt.csv")
p = max(t for _, _, t in goose)
blas1 = load("bench_lapack_mt1.csv", 1)
blasp = load("bench_lapack_mt.csv", p)
ops = ["matmul", "det", "solve", "solve_matrix", "inverse", "cholesky"]
sizes = sorted({n for _, n, _ in goose})
pb = max(t for _, _, t in blasp)

print(f"{'op':13}{'n':>5} {'goose 1T':>10} {'goose ' + str(p) + 'T':>10} {'speedup':>8} "
      f"{'blas 1T':>10} {'blas ' + str(pb) + 'T':>10} {'speedup':>8} {'goose/blas ' + str(p) + 'T':>15}")
for op in ops:
    for n in sizes:
        g1, gp = goose[(op, n, 1)], goose[(op, n, p)]
        b1, bp = blas1[(op, n, 1)], blasp[(op, n, pb)]
        print(f"{op:13}{n:5} {g1 * 1e3:10.2f} {gp * 1e3:10.2f} {g1 / gp:7.2f}x "
              f"{b1 * 1e3:10.2f} {bp * 1e3:10.2f} {b1 / bp:7.2f}x {gp / bp:14.2f}x")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
colors = plt.cm.tab10.colors
for i, op in enumerate(ops):
    ax[0].plot(sizes, [goose[(op, n, 1)] / goose[(op, n, p)] for n in sizes], "o-", color=colors[i], label=op)
    ax[0].plot(sizes, [blas1[(op, n, 1)] / blasp[(op, n, pb)] for n in sizes], ":", color=colors[i], alpha=0.6)
ax[0].axhline(1, color="k", lw=0.8)
ax[0].set(xscale="log", xlabel="n (n x n matrices)", ylabel="speedup over 1 thread",
          title=f"Speedup on {p} threads (solid: Goose, dotted: OpenBLAS)")
ax[0].set_xticks(sizes, [str(n) for n in sizes])
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(ncol=2, fontsize=8)

n = sizes[-1]
series = [("Goose, 1 thread", lambda op: goose[(op, n, 1)]), (f"Goose, {p} threads", lambda op: goose[(op, n, p)]),
          ("OpenBLAS, 1 thread", lambda op: blas1[(op, n, 1)]), (f"OpenBLAS, {pb} threads", lambda op: blasp[(op, n, pb)])]
w = 0.8 / len(series)
for s, (name, get) in enumerate(series):
    ax[1].bar([i + (s - (len(series) - 1) / 2) * w for i in range(len(ops))], [get(op) * 1e3 for op in ops], w, label=name)
ax[1].set(yscale="log", ylabel="ms (log)", title=f"Absolute time at n = {n}")
ax[1].set_xticks(range(len(ops)), ops, rotation=30)
ax[1].legend(fontsize=8)
fig.suptitle(f"linalg.parallel vs OpenBLAS 0.3.29, {p} threads pinned to one logical CPU per P-core, i5-12600K, "
             "MSVC /O2 /arch:AVX2 /fp:contract", fontsize=10)
fig.tight_layout()
fig.savefig("compare_mt.png", dpi=120)
print("wrote compare_mt.png")
