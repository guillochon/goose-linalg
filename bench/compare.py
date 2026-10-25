"""Compare the Goose runs against OpenBLAS: a table and compare.png.

    python compare.py [--with-prev]      (in this directory, after run_pinned.ps1)

The headline comparison is production against production: Goose built with
/arch:AVX2 (bench_goose_avx2.csv) against single-threaded
OpenBLAS (bench_lapack_1t.csv), both pinned to one P-core. Other runs are
shown when their CSVs exist: bench_goose_prev_avx2.csv (an earlier version,
same flags; in the table always, in the chart only with --with-prev),
bench_goose_base.csv (plain /O2) and bench_goose_fast.csv (/fp:fast,
reference only). The multithreaded comparison is compare_mt.py's.
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {(r["op"], int(r["n"])): float(r["seconds"]) for r in csv.DictReader(f)}


now = load("bench_goose_avx2.csv")
prev = load("bench_goose_prev_avx2.csv")
chart_prev = prev if "--with-prev" in sys.argv else None
base = load("bench_goose_base.csv")
fast = load("bench_goose_fast.csv")
one = load("bench_lapack_1t.csv")
ops = ["matmul", "matvec", "solve", "det", "inverse", "cholesky", "qr", "eigh", "svd"]
sizes = sorted({n for _, n in now})

flops = {"matmul": lambda n: 2 * n**3, "matvec": lambda n: 2 * n**2, "solve": lambda n: 2 / 3 * n**3,
         "det": lambda n: 2 / 3 * n**3, "inverse": lambda n: 2 * n**3, "cholesky": lambda n: n**3 / 3,
         "qr": lambda n: 8 / 3 * n**3}

cols = [("prev", prev), ("now", now), ("base", base), ("fast", fast), ("blas1t", one)]
cols = [(name, data) for name, data in cols if data]
print(f"{'op':9}{'n':>4} " + " ".join(f"{name + ' ms':>10}" for name, _ in cols)
      + f" {'now/blas':>9} {'now GF/s':>9} {'blas GF/s':>9}")
for op in ops:
    for n in sizes:
        k = (op, n)
        line = f"{op:9}{n:4} " + " ".join(f"{data[k] * 1e3:10.3f}" for _, data in cols)
        line += f" {now[k] / one[k]:8.2f}x"
        if op in flops:
            line += f" {flops[op](n) / now[k] / 1e9:9.1f} {flops[op](n) / one[k] / 1e9:9.1f}"
        print(line)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
colors = plt.cm.tab10.colors
for i, op in enumerate(ops):
    ax[0].plot(sizes, [now[(op, n)] / one[(op, n)] for n in sizes], "o-", color=colors[i], label=op)
    if chart_prev:
        ax[0].plot(sizes, [chart_prev[(op, n)] / one[(op, n)] for n in sizes], ":", color=colors[i], alpha=0.6)
ax[0].axhline(1, color="k", lw=0.8)
ax[0].set(xscale="log", yscale="log", xlabel="n (n x n matrices)", ylabel="Goose time / OpenBLAS time",
          title="vs 1-thread OpenBLAS, both AVX2" + (" (dotted: earlier version)" if chart_prev else ""))
ax[0].set_xticks(sizes, [str(n) for n in sizes])
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
ax[0].yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x" if v in (0.5, 2, 3, 5) else ""))
ax[0].legend(ncol=3, fontsize=8)

n = sizes[-1]
series = [(name, data) for name, data in [("Goose, earlier version", chart_prev), ("Goose linalg", now),
                                          ("OpenBLAS, 1 thread", one)] if data]
w = 0.8 / len(series)
for s, (name, data) in enumerate(series):
    ax[1].bar([i + (s - (len(series) - 1) / 2) * w for i in range(len(ops))],
              [data[(op, n)] * 1e3 for op in ops], w, label=name)
ax[1].set(yscale="log", ylabel="ms (log)", title=f"Absolute time at n = {n}")
ax[1].set_xticks(range(len(ops)), ops, rotation=30)
ax[1].legend(fontsize=8)
fig.suptitle("Goose linalg (clang-cl 22 /O2 /arch:AVX2, 1 thread) vs NumPy + OpenBLAS 0.3.29, "
             "i5-12600K, pinned to one P-core", fontsize=10)
fig.tight_layout()
fig.savefig("compare.png", dpi=120)
print("wrote compare.png")
