"""Compare the Goose runs against OpenBLAS: a table and compare.png.

    python bench/compare.py      (in the directory holding the CSVs)

Reads bench_goose.csv (current code), bench_goose_v0.1.csv (the first
version, optional), bench_lapack_1t.csv and bench_lapack_mt.csv. For stable
numbers on a hybrid CPU, run every benchmark pinned to the same performance
core.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return {(r["op"], int(r["n"])): float(r["seconds"]) for r in csv.DictReader(f)}


goose = load("bench_goose.csv")
old = load("bench_goose_v0.1.csv") if os.path.exists("bench_goose_v0.1.csv") else None
one = load("bench_lapack_1t.csv")
multi = load("bench_lapack_mt.csv")
ops = ["matmul", "matvec", "solve", "det", "inverse", "cholesky", "qr", "eigh", "svd"]
sizes = sorted({n for _, n in goose})

# Leading flop counts, for GFLOP/s.
flops = {"matmul": lambda n: 2 * n**3, "matvec": lambda n: 2 * n**2, "solve": lambda n: 2 / 3 * n**3,
         "det": lambda n: 2 / 3 * n**3, "inverse": lambda n: 2 * n**3, "cholesky": lambda n: n**3 / 3,
         "qr": lambda n: 8 / 3 * n**3}

head = f"{'op':9} {'n':>4} "
if old:
    head += f"{'v0.1 ms':>10} "
head += f"{'goose ms':>10} "
if old:
    head += f"{'speedup':>8} "
head += f"{'blas-1t ms':>11} {'goose/1t':>9} {'goose GF/s':>10} {'1t GF/s':>8}"
print(head)
for op in ops:
    for n in sizes:
        g, o = goose[(op, n)], one[(op, n)]
        line = f"{op:9} {n:4} "
        if old:
            line += f"{old[(op, n)] * 1e3:10.3f} "
        line += f"{g * 1e3:10.3f} "
        if old:
            line += f"{old[(op, n)] / g:7.2f}x "
        line += f"{o * 1e3:11.3f} {g / o:8.1f}x "
        line += f"{flops[op](n) / g / 1e9:10.1f} {flops[op](n) / o / 1e9:8.1f}" if op in flops else ""
        print(line)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
colors = plt.cm.tab10.colors
for i, op in enumerate(ops):
    ax[0].plot(sizes, [goose[(op, n)] / one[(op, n)] for n in sizes], "o-", color=colors[i], label=op)
    if old:
        ax[0].plot(sizes, [old[(op, n)] / one[(op, n)] for n in sizes], ":", color=colors[i], alpha=0.6)
ax[0].axhline(1, color="k", lw=0.8)
ax[0].set(xscale="log", yscale="log", xlabel="n (n x n matrices)", ylabel="Goose time / OpenBLAS time",
          title="Slowdown vs 1-thread OpenBLAS (solid: now, dotted: v0.1)")
ax[0].set_xticks(sizes, [str(n) for n in sizes])
ax[0].legend(ncol=3, fontsize=8)

n = sizes[-1]
series = ([("Goose v0.1", old)] if old else []) + [("Goose now", goose), ("OpenBLAS, 1 thread", one),
                                                    ("OpenBLAS, 16 threads", multi)]
w = 0.8 / len(series)
for s, (name, data) in enumerate(series):
    ax[1].bar([i + (s - (len(series) - 1) / 2) * w for i in range(len(ops))],
              [data[(op, n)] * 1e3 for op in ops], w, label=name)
ax[1].set(yscale="log", ylabel="ms (log)", title=f"Absolute time at n = {n}")
ax[1].set_xticks(range(len(ops)), ops, rotation=30)
ax[1].legend(fontsize=8)
fig.suptitle("Goose linalg (MSVC /O2, 1 thread) vs NumPy + OpenBLAS 0.3.29, i5-12600K, pinned to one P-core")
fig.tight_layout()
fig.savefig("compare.png", dpi=120)
print("wrote compare.png")
