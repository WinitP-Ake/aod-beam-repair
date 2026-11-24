#!/usr/bin/env python3
"""
Section 4.5: choosing the beam width k.
Re-creates the SAME 300 E1 tables (same generator, seed 0).
Consistency check: raw beam at k=1 must print 83.0% and at k=16 96.7% (Table 4).

  python3 choose_k.py | tee choose_k.txt
Outputs: choose_k_results.csv, figures/fig4_choose_k.png
"""
import csv, os, random, statistics
from aod_beam import Instance, beam, cleanup, random_tiny

KS = [1, 2, 4, 8, 16, 32, 64, 128]
AGGS = ("avg", "sum", "median")
EPS = 1e-9


def e1_instances(n=100, seed=0):
    """Identical instance stream to aod_beam.e1 (rng is used only by random_tiny)."""
    rng, out = random.Random(seed), []
    for a in AGGS:
        made = 0
        while made < n:
            rows, G = random_tiny(rng)
            inst = Instance(rows, G, a)
            if inst.holds(inst.init):
                continue
            made += 1
            out.append(inst)
    return out


def beam_lb(inst, k):
    """BeamRepair returning (state, #score evaluations, lower bound on the optimum).
    Until the beam discards a state, the candidates at depth d are ALL states with
    d deletions; none satisfied the AOD, so optimum > d. If the first discard happens
    after depth p, then optimum >= p + 1."""
    st0 = inst.init
    if inst.holds(st0):
        return st0, 0, 0
    frontier, depth, lb, evals = [st0], 0, None, 0
    while True:
        depth += 1
        cands = {}
        for st in frontier:
            for i in inst.order:
                if st[i] == 0:
                    continue
                new = st[:i] + (st[i] - 1,) + st[i + 1:]
                if new not in cands:
                    cands[new] = round(inst.score(new), 9)
                    evals += 1
        done = [s for s, v in cands.items() if v <= EPS]
        if done:
            return done[0], evals, (depth if lb is None else lb)
        ranked = sorted(cands, key=cands.get)
        if lb is None and len(ranked) > k:
            lb = depth + 1
        frontier = ranked[:k]


def doubling(rows, m, use_lb):
    """Iterative doubling over KS; rows = this instance's records sorted by k."""
    best, LB, stall, work = None, 0, 0, 0
    for r in rows:
        work += r["evals"]
        LB = max(LB, r["lb"])
        if best is None or r["beam_clean"] < best:
            best, stall = r["beam_clean"], 0
        else:
            stall += 1
        if use_lb and best == LB:
            return best, work, r["k"], True
        if stall >= m:
            break
    return best, work, r["k"], best == LB


def pct(xs):
    return 100 * statistics.mean(xs)


def main():
    insts = e1_instances()
    recs = []
    for j, inst in enumerate(insts):
        opt = inst.deleted(beam(inst, None))
        for k in KS:
            st, ev, lb = beam_lb(inst, k)
            assert lb <= opt, "lower bound exceeded the optimum - bug"
            recs.append(dict(inst=j, agg=inst.agg, k=k, opt=opt, beam=inst.deleted(st),
                             beam_clean=inst.deleted(cleanup(inst, st)), lb=lb, evals=ev))
    with open("choose_k_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)

    per_inst = {}
    for r in recs:
        per_inst.setdefault(r["inst"], []).append(r)
    for v in per_inst.values():
        v.sort(key=lambda r: r["k"])

    # (a) + (b): fixed k
    print(f"\n(a,b) Fixed k on {len(insts)} E1 tables")
    print(f"{'k':>4} {'raw%opt':>8} " + " ".join(f"{a + '%opt':>10}" for a in AGGS)
          + f" {'all%opt':>8} {'%proved':>8} {'evals':>8}")
    fixed = {}
    for k in KS:
        rs = [r for r in recs if r["k"] == k]
        raw = pct([r["beam"] == r["opt"] for r in rs])
        by_agg = [pct([r["beam_clean"] == r["opt"] for r in rs if r["agg"] == a]) for a in AGGS]
        allopt = pct([r["beam_clean"] == r["opt"] for r in rs])
        proved = pct([r["beam_clean"] == r["lb"] for r in rs])
        work = statistics.mean(r["evals"] for r in rs)
        fixed[k] = (allopt, proved, work)
        print(f"{k:>4} {raw:>8.1f} " + " ".join(f"{x:>10.1f}" for x in by_agg)
              + f" {allopt:>8.1f} {proved:>8.1f} {work:>8.0f}")
    print("  (raw%opt = beam without cleanup; must be 83.0 at k=1 and 96.7 at k=16)")

    # (c): iterative doubling
    base = fixed[16][2]
    print("\n(c) Iterative doubling (beam + cleanup), work relative to fixed k=16")
    print(f"{'strategy':<22} {'%opt':>6} {'%proved':>8} {'work':>8} {'xk16':>6} {'mean final k':>13}")
    strat = {}
    for use_lb in (True, False):
        for m in (1, 2, 3, 4):
            res = [doubling(v, m, use_lb) for v in per_inst.values()]
            opts = [per_inst[j][0]["opt"] for j in per_inst]
            o = pct([b == op for (b, _, _, _), op in zip(res, opts)])
            p = pct([pr for *_, pr in res])
            wk = statistics.mean(w for _, w, _, _ in res)
            fk = statistics.mean(k for _, _, k, _ in res)
            name = f"double m={m}" + (" +LB" if use_lb else "")
            strat[name] = (o, wk)
            print(f"{name:<22} {o:>6.1f} {p:>8.1f} {wk:>8.0f} {wk / base:>6.2f} {fk:>13.1f}")

    # k*: smallest k reaching the optimum
    print("\nSmallest k reaching the exact optimum (beam + cleanup), k* per table")
    print(f"{'agg':<7} {'median k*':>9} {'p90 k*':>7} {'max k*':>7} {'%k*<=16':>8} {'%non-monotone':>14}")
    fmt = lambda x: ">128" if x == float("inf") else str(x)
    for a in AGGS + ("all",):
        ks, nonmono = [], 0
        for v in per_inst.values():
            if a != "all" and v[0]["agg"] != a:
                continue
            hits = [r["k"] == r["k"] and r["beam_clean"] == r["opt"] for r in v]
            first = next((r["k"] for r, h in zip(v, hits) if h), float("inf"))
            ks.append(first)
            if first != float("inf") and not all(h for r, h in zip(v, hits) if r["k"] >= first):
                nonmono += 1
        ks.sort()
        print(f"{a:<7} {fmt(ks[len(ks) // 2]):>9} {fmt(ks[int(0.9 * (len(ks) - 1))]):>7} "
              f"{fmt(ks[-1]):>7} {pct([x <= 16 for x in ks]):>8.1f} {100 * nonmono / len(ks):>14.1f}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
        for a in AGGS:
            ax[0].plot(KS, [pct([r["beam_clean"] == r["opt"] for r in recs if r["k"] == k and r["agg"] == a])
                            for k in KS], "o-", ms=3, label=a)
            ax[1].plot(KS, [pct([r["beam_clean"] == r["lb"] for r in recs if r["k"] == k and r["agg"] == a])
                            for k in KS], "o-", ms=3, label=a)
        for i, t in enumerate(["(a) % reaching exact optimum", "(b) % proved optimal by lower bound"]):
            ax[i].set_xscale("log", base=2); ax[i].set_xlabel("beam width k"); ax[i].set_title(t, fontsize=8)
            ax[i].set_ylim(0, 105); ax[i].legend(fontsize=7)
        ax[2].plot([fixed[k][2] for k in KS], [fixed[k][0] for k in KS], "o-", label="fixed k")
        for k in KS:
            ax[2].annotate(f"k={k}", (fixed[k][2], fixed[k][0]), fontsize=6, xytext=(2, -8), textcoords="offset points")
        for tag, mk in ((" +LB", "s--"), ("", "^:")):
            pts = [strat[f"double m={m}{tag}"] for m in (1, 2, 3, 4)]
            ax[2].plot([w for _, w in pts], [o for o, _ in pts], mk, label="doubling" + (" + bound" if tag else ""))
        ax[2].set_xscale("log"); ax[2].set_xlabel("score evaluations per table (log)")
        ax[2].set_ylabel("% optimal"); ax[2].set_title("(c) quality vs work", fontsize=8); ax[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig("figures/fig4_choose_k.png", dpi=300)
        print("\n  saved figures/fig4_choose_k.png")
    except ImportError:
        print("  (matplotlib not installed, figure skipped)")


if __name__ == "__main__":
    main()