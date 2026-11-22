#!/usr/bin/env python3
"""
Figures and tables for the Track 2 report.

Step 1:  python3 aod_beam.py --exp all --n 100 --seed 0 | tee results.txt
Step 2:  python3 make_figures.py

Needs matplotlib (pip3 install matplotlib); aod_beam.py must be in the same folder.
Outputs in ./figures:
  fig1_toy.png          greedy vs beam on the worked example
  fig2_e1_optimal.png   % of runs reaching the exact optimum (E1)
  fig3_e4_noise.png     mean gap vs noise level (E4)
  fig4_e5_tradeoff.png  quality vs runtime for beam widths (E5)
  tables.md             all tables in Markdown (copy into Word / Google Docs)
  tables.tex            the same tables in LaTeX (needs \\usepackage{booktabs})
"""
import csv, os, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aod_beam import Instance

OUT = "figures"
ORDER = ["greedy", "greedy+clean", "greedy+clean+swap", "beam4", "beam4+clean",
         "beam16", "beam16+clean", "beam16+clean+swap"]
PLOT = ["greedy", "greedy+clean", "beam4", "beam16", "beam16+clean"]
E5_ORDER = ["k=1", "k=2", "k=4", "k=8", "k=16", "k=32", "k=64",
            "k=1+clean", "k=4+clean", "k=16+clean"]
plt.rcParams.update({"font.size": 8})


# ---------------- loading and statistics
def load(path):
    if not os.path.exists(path):
        print(f"  {path} not found -> skipped")
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["size"], r["ref"], r["time"] = int(r["size"]), int(r["ref"]), float(r["time"])
    return rows


def group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def stats(rs):
    if not rs:
        return None
    gaps = [r["size"] / r["ref"] for r in rs if r["ref"] > 0] or [1.0]
    return dict(runs=len(rs),
                pct=100 * sum(r["size"] <= r["ref"] for r in rs) / len(rs),
                gap=statistics.mean(gaps),
                worst=max(gaps),
                excess=sum(r["size"] - r["ref"] for r in rs),
                ms=1000 * statistics.mean(r["time"] for r in rs))


def methods_in(rows, order):
    present = {r["method"] for r in rows}
    return [m for m in order if m in present]


# ---------------- tables
def table(rows, ctx_keys, order, base="greedy"):
    header = list(ctx_keys) + ["method", "runs", "% at ref", "mean gap", "worst gap",
                               "excess cut vs greedy", "ms/run", "x greedy time"]
    body = []
    for ctx, rs in group(rows, ctx_keys).items():
        per = {m: stats(x) for (m,), x in group(rs, ["method"]).items()}
        b = per.get(base)
        for m in methods_in(rs, order):
            s = per[m]
            cut = "n/a" if not b or b["excess"] == 0 else f"{100 * (1 - s['excess'] / b['excess']):.0f}%"
            slow = "n/a" if not b or b["ms"] == 0 else f"{s['ms'] / b['ms']:.1f}"
            body.append(list(ctx) + [m, s["runs"], f"{s['pct']:.1f}", f"{s['gap']:.3f}",
                                     f"{s['worst']:.2f}", cut, f"{s['ms']:.2f}", slow])
    return header, body


def md(title, header, body):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n\n"


def tex(title, header, body):
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_")
    lines = [r"\begin{table}[t]\centering\small", rf"\caption{{{esc(title)}}}",
             r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}\end{table}"]
    return "\n".join(lines) + "\n\n"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}")


# ---------------- figures
def fig_toy():
    rows = [(0, 20)] * 2 + [(0, 4)] * 2 + [(1, 0), (1, 9), (1, 9)]
    inst = Instance(rows, 2, "avg")
    idx = {t: i for i, t in enumerate(inst.types)}

    def path(dels):
        c = list(inst.init)
        s = [inst.score(c)]
        for t in dels:
            c[idx[t]] -= 1
            s.append(inst.score(c))
        return s

    g = path([(1, 0), (0, 20), (0, 20)])   # greedy: delete BSc 0, then both HS 20s
    b = path([(0, 20), (0, 20)])           # beam k=2: delete both HS 20s
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(range(len(g)), g, "o-", label="Greedy (HeurRepair): 3 deletions")
    ax.plot(range(len(b)), b, "s--", label="Beam k=2: 2 deletions (optimal)")
    for x, y in enumerate(g):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for x, y in enumerate(b):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xlabel("Number of deleted tuples")
    ax.set_ylabel("Violation score $S_{MVI}$")
    ax.set_title("Toy example: education \u2197 avg(salary)")
    ax.legend(fontsize=7)
    save(fig, "fig1_toy.png")


def fig_e1(rows):
    aggs = sorted({r["agg"] for r in rows})
    ms = methods_in(rows, PLOT)
    w = 0.8 / len(ms)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for j, m in enumerate(ms):
        vals = [stats([r for r in rows if r["agg"] == a and r["method"] == m])["pct"] for a in aggs]
        xs = [i + (j - (len(ms) - 1) / 2) * w for i in range(len(aggs))]
        bars = ax.bar(xs, vals, w, label=m)
        ax.bar_label(bars, fmt="%.0f", fontsize=6)
    ax.set_xticks(range(len(aggs)))
    ax.set_xticklabels(aggs)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% runs reaching exact optimum")
    ax.legend(ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "fig2_e1_optimal.png")


def fig_e4(rows):
    aggs = sorted({r["agg"] for r in rows})
    ms = methods_in(rows, PLOT)
    fig, axes = plt.subplots(1, len(aggs), figsize=(6, 2.6), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], aggs):
        noises = sorted({float(r["noise"]) for r in rows if r["agg"] == a})
        for m in ms:
            xs, ys = [], []
            for n in noises:
                s = stats([r for r in rows if r["agg"] == a and r["method"] == m
                           and float(r["noise"]) == n])
                if s:
                    xs.append(100 * n); ys.append(s["gap"])
            ax.plot(xs, ys, marker="o", ms=3, label=m)
        ax.axhline(1, ls=":", c="gray")
        ax.set_title(f"aggregate = {a}")
        ax.set_xlabel("injected noise (%)")
    axes[0][0].set_ylabel("mean gap (method / best known)")
    axes[0][-1].legend(fontsize=6)
    save(fig, "fig3_e4_noise.png")


def fig_e5(rows):
    ms = methods_in(rows, E5_ORDER)
    pts = {m: stats([r for r in rows if r["method"] == m]) for m in ms}
    fig, ax = plt.subplots(figsize=(4.5, 3))
    plain = [m for m in ms if "+clean" not in m]
    clean = [m for m in ms if "+clean" in m]
    ax.plot([pts[m]["ms"] for m in plain], [pts[m]["gap"] for m in plain], "o-", label="beam (no cleanup)")
    ax.plot([pts[m]["ms"] for m in clean], [pts[m]["gap"] for m in clean], "s--", label="beam + cleanup")
    for m in ms:
        ax.annotate(m, (pts[m]["ms"], pts[m]["gap"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.axhline(1, ls=":", c="gray")
    ax.set_xscale("log")
    ax.set_xlabel("runtime per instance (ms, log scale)")
    ax.set_ylabel("mean gap (method / best known)")
    ax.legend(fontsize=7)
    save(fig, "fig4_e5_tradeoff.png")


# ---------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    fig_toy()
    e1, e4, e5 = load("e1_results.csv"), load("e4_results.csv"), load("e5_results.csv")
    md_out = tex_out = ""
    specs = []
    if e1:
        specs.append(("E1 pooled: tiny tables, exact optimum as reference", e1, [], ORDER, "greedy"))
        specs.append(("E1 by aggregate", e1, ["agg"], ORDER, "greedy"))
        fig_e1(e1)
    if e4:
        specs.append(("E4 noise sweep, best-known reference", e4, ["agg", "noise"],
                      ["greedy", "greedy+clean", "beam16", "beam16+clean"], "greedy"))
        fig_e4(e4)
    if e5:
        specs.append(("E5 beam-width trade-off, avg, 10% noise", e5, [], E5_ORDER, "k=1"))
        fig_e5(e5)
    for title, rows, keys, order, base in specs:
        h, b = table(rows, keys, order, base)
        md_out += md(title, h, b)
        tex_out += tex(title, h, b)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(md_out)
    with open(os.path.join(OUT, "tables.tex"), "w") as f:
        f.write(tex_out)
    print(md_out)
    print(f"  saved {OUT}/tables.md and {OUT}/tables.tex")


if __name__ == "__main__":
    main()