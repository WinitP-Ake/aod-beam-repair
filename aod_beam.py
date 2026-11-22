#!/usr/bin/env python3
"""
Beam-search + local-search cleanup for AOD C-repair (violation score = S_MVI).
No dependencies (matplotlib optional, for plots).

  python3 aod_beam.py --toy                       # reproduces the worked example
  python3 aod_beam.py --exp e1 --n 300            # greedy failure study vs exact
  python3 aod_beam.py --exp e4 --n 100            # noise sweep
  python3 aod_beam.py --exp e5 --n 100            # beam-width trade-off
  python3 aod_beam.py --exp all
  python3 aod_beam.py --csv data.csv --group-col education --value-col salary \
        --agg avg --sample 2000 --round 0 [--group-order "HS,BSc,MSc,PhD"]
"""
import argparse, csv, random, statistics, time
from collections import Counter, defaultdict

EPS = 1e-9


# ---------- aggregates (input: list of (value, count), sorted by value, count>0)
def _kth(vc, k):
    for v, c in vc:
        if k < c:
            return v
        k -= c


def agg(name, vc):
    n = sum(c for _, c in vc)
    if name == "avg":
        return sum(v * c for v, c in vc) / n
    if name == "sum":
        return sum(v * c for v, c in vc)
    if name == "max":
        return vc[-1][0]
    if name == "min":
        return vc[0][0]
    if name == "count":
        return n
    if name == "median":
        if n % 2:
            return _kth(vc, n // 2)
        return (_kth(vc, n // 2 - 1) + _kth(vc, n // 2)) / 2
    raise ValueError(name)


# ---------- instance: identical (group, value) tuples are merged into "types"
class Instance:
    def __init__(self, rows, n_groups, agg_name="avg"):
        cnt = Counter(rows)
        self.types = sorted(cnt)                      # (group, value)
        self.init = tuple(cnt[t] for t in self.types)
        self.G, self.agg, self.n = n_groups, agg_name, sum(self.init)
        self.by_group = defaultdict(list)
        for i, (g, _) in enumerate(self.types):
            self.by_group[g].append(i)                # ascending value
        # tie-break like the paper: lowest group first, then highest value
        self.order = sorted(range(len(self.types)),
                            key=lambda i: (self.types[i][0], -self.types[i][1]))

    def score(self, counts):
        prev, total = None, 0.0
        for g in range(self.G):
            vc = [(self.types[i][1], counts[i]) for i in self.by_group[g] if counts[i] > 0]
            if not vc:
                continue                              # empty group is skipped
            a = agg(self.agg, vc)
            if prev is not None and prev > a:
                total += prev - a
            prev = a
        return total

    def holds(self, counts):
        return self.score(counts) <= EPS

    def deleted(self, counts):
        return self.n - sum(counts)


# ---------- beam search. k=1 == greedy HeurRepair, k=None == exact BFS
def beam(inst, k):
    start = inst.init
    if inst.holds(start):
        return start
    frontier = [start]
    while True:
        cands = {}
        for st in frontier:
            for i in inst.order:
                if st[i] == 0:
                    continue
                new = st[:i] + (st[i] - 1,) + st[i + 1:]
                if new not in cands:
                    cands[new] = round(inst.score(new), 9)
        done = [s for s, v in cands.items() if v <= EPS]
        if done:
            return done[0]          # all candidates have equal depth -> minimal
        ranked = sorted(cands, key=cands.get)         # stable sort keeps tie-break
        frontier = ranked if k is None else ranked[:k]


# ---------- local-search cleanup
def _swap_pass(inst, counts):
    """put back two deleted tuples, delete one other tuple (net -1)."""
    T, init = len(counts), inst.init
    dele = [i for i in range(T) if counts[i] < init[i]]
    for a in range(len(dele)):
        for b in range(a, len(dele)):
            i1, i2 = dele[a], dele[b]
            for j in range(T):
                if j in (i1, i2) or counts[j] == 0:
                    continue
                new = list(counts)
                new[i1] += 1; new[i2] += 1; new[j] -= 1
                if new[i1] > init[i1] or new[i2] > init[i2]:
                    continue
                if inst.holds(new):
                    return tuple(new)
    return None


def cleanup(inst, state, swap=False):
    counts = list(state)
    while True:
        for i in inst.order:                          # add-back pass
            while counts[i] < inst.init[i]:
                counts[i] += 1
                if not inst.holds(counts):
                    counts[i] -= 1
                    break
        if not swap:
            break
        s = _swap_pass(inst, counts)
        if s is None:
            break
        counts = list(s)
    return tuple(counts)


# ---------- methods: name -> (beam width, 0=none | 1=add-back | 2=add-back+swap)
METHODS = {
    "greedy": (1, 0), "greedy+clean": (1, 1), "greedy+clean+swap": (1, 2),
    "beam4": (4, 0), "beam4+clean": (4, 1),
    "beam16": (16, 0), "beam16+clean": (16, 1), "beam16+clean+swap": (16, 2),
}


def run_method(inst, k, mode):
    t = time.perf_counter()
    st = beam(inst, k)
    if mode:
        st = cleanup(inst, st, swap=(mode == 2))
    dt = time.perf_counter() - t
    assert inst.holds(st), "invalid repair"
    return inst.deleted(st), dt


def best_known(inst):
    return inst.deleted(cleanup(inst, beam(inst, 64), swap=True))


def evaluate(inst, ref, extra, recs, methods=METHODS):
    for name, (k, mode) in methods.items():
        size, dt = run_method(inst, k, mode)
        recs.append(dict(extra, method=name, size=size, ref=ref, time=dt))


# ---------- instance generators
def random_tiny(rng):
    G, n = rng.choice([3, 4]), rng.randint(8, 13)
    return [(rng.randrange(G), rng.randint(0, 20)) for _ in range(n)], G


def planted(n, G, noise, rng, agg_name="avg", step=3, sd=6, hi=40):
    """1) draw a clean table whose trend HOLDS (resample until it does);
       2) give a `noise` fraction of tuples values from the REVERSED trend."""
    def draw(mean):
        return min(hi, max(0, round(rng.gauss(mean, sd))))
    for _ in range(500):
        groups = [rng.randrange(G) for _ in range(n)]
        rows = [(g, draw(10 + step * g)) for g in groups]
        inst = Instance(rows, G, agg_name)
        if inst.holds(inst.init):
            break
    for idx in rng.sample(range(n), max(1, round(noise * n))):
        g = rows[idx][0]
        rows[idx] = (g, draw(10 + step * (G - 1 - g)))
    return rows, G


def violating_planted(n, G, noise, rng, agg_name, max_tries=200):
    """keep generating until the noisy table violates the AOD."""
    for _ in range(max_tries):
        rows, G = planted(n, G, noise, rng, agg_name)
        inst = Instance(rows, G, agg_name)
        if not inst.holds(inst.init):
            return inst
    return None


# ---------- reporting
def summarize(recs, by=("method",)):
    if not recs:
        print("  (no violating instances generated - nothing to summarize)")
        return []
    groups = defaultdict(list)
    for r in recs:
        groups[tuple(r[b] for b in by)].append(r)
    print("  ".join(f"{b:<12}" for b in by),
          f"{'runs':>5} {'%at_ref':>8} {'mean_gap':>9} {'ms/run':>9}")
    out = []
    for key, rs in groups.items():
        gaps = [r["size"] / r["ref"] for r in rs if r["ref"] > 0]
        pct = 100 * sum(r["size"] <= r["ref"] for r in rs) / len(rs)
        gap = statistics.mean(gaps) if gaps else 1.0
        ms = 1000 * statistics.mean(r["time"] for r in rs)
        print("  ".join(f"{str(x):<12}" for x in key), f"{len(rs):>5} {pct:>8.1f} {gap:>9.3f} {ms:>9.2f}")
        out.append((key, pct, gap, ms))
    return out


def save_csv(recs, path):
    if not recs:
        print(f"  no results, {path} not written")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)
    print(f"  saved {path}")


# ---------- experiments
def e1(n_inst, seed):
    print("\n=== E1: tiny random tables, ground truth = exact BFS ===")
    rng, recs, example = random.Random(seed), [], None
    for agg_name in ("avg", "sum", "median"):
        made = 0
        while made < n_inst:
            rows, G = random_tiny(rng)
            inst = Instance(rows, G, agg_name)
            if inst.holds(inst.init):
                continue
            made += 1
            ref = inst.deleted(beam(inst, None))
            evaluate(inst, ref, {"agg": agg_name}, recs)
            g, _ = run_method(inst, 1, 0)
            if g > ref and (example is None or inst.n < example[0].n):
                example = (inst, g, ref)
    summarize(recs, ("agg", "method"))
    save_csv(recs, "e1_results.csv")
    if example:
        inst, g, ref = example
        print(f"\nSmallest greedy failure ({inst.agg}, {inst.n} tuples): greedy={g}, optimal={ref}")
        print("rows (group, value):", sorted(Counter(r for r, c in zip(inst.types, inst.init)
                                                        for _ in range(c)).elements()))


def e4(n_inst, seed):
    print("\n=== E4: noise sweep (40 tuples, 4 groups; reference = best known) ===")
    rng, recs = random.Random(seed), []
    for agg_name in ("avg", "median"):
        for noise in (0.05, 0.10, 0.15, 0.20, 0.30):
            made = 0
            for _ in range(n_inst):
                inst = violating_planted(40, 4, noise, rng, agg_name)
                if inst is None:
                    continue
                made += 1
                evaluate(inst, best_known(inst), {"agg": agg_name, "noise": noise}, recs)
            print(f"  {agg_name:<7} noise={noise:<5} violating instances: {made}/{n_inst}")
    res = summarize(recs, ("agg", "noise", "method"))
    save_csv(recs, "e4_results.csv")
    return res


def e5(n_inst, seed):
    print("\n=== E5: beam-width trade-off (50 tuples, avg, noise 10%; reference = best known) ===")
    rng, recs = random.Random(seed), []
    ks = {f"k={k}": (k, 0) for k in (1, 2, 4, 8, 16, 32, 64)}
    ks.update({f"k={k}+clean": (k, 1) for k in (1, 4, 16)})
    made = 0
    for _ in range(n_inst):
        inst = violating_planted(50, 4, 0.10, rng, "avg")
        if inst is None:
            continue
        made += 1
        evaluate(inst, best_known(inst), {}, recs, ks)
    print(f"  violating instances: {made}/{n_inst}")
    res = summarize(recs)
    save_csv(recs, "e5_results.csv")
    if not res:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.4))
        for (m,), pct, gap, ms in res:
            plt.scatter(ms, gap)
            plt.annotate(m, (ms, gap), fontsize=7)
        plt.xscale("log"); plt.xlabel("ms per run"); plt.ylabel("mean gap vs best known")
        plt.tight_layout(); plt.savefig("e5_tradeoff.png", dpi=200)
        print("  saved e5_tradeoff.png")
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")


def toy():
    rows = [(0, 20)] * 2 + [(0, 4)] * 2 + [(1, 0), (1, 9), (1, 9)]
    inst = Instance(rows, 2, "avg")
    print("start score:", inst.score(inst.init))
    for name, (k, mode) in {"greedy": (1, 0), "greedy+clean": (1, 1),
                            "beam k=2": (2, 0), "exact": (None, 0)}.items():
        st = beam(inst, k)
        if mode:
            st = cleanup(inst, st)
        removed = [(t, inst.init[i] - st[i]) for i, t in enumerate(inst.types) if st[i] < inst.init[i]]
        print(f"{name:<13} deletions={inst.deleted(st)}  removed (group,value)x count: {removed}")


def real(args):
    rows_raw = list(csv.DictReader(open(args.csv)))
    rows_raw = [r for r in rows_raw if r[args.group_col] != "" and r[args.value_col] != ""]
    rng = random.Random(args.seed)
    if args.sample and len(rows_raw) > args.sample:
        rows_raw = rng.sample(rows_raw, args.sample)
    names = args.group_order.split(",") if args.group_order else None
    if names is None:
        raw = {r[args.group_col] for r in rows_raw}
        try:
            names = sorted(raw, key=float)
        except ValueError:
            names = sorted(raw)
    idx = {nme: i for i, nme in enumerate(names)}
    rows = [(idx[r[args.group_col]], round(float(r[args.value_col]), args.round))
            for r in rows_raw if r[args.group_col] in idx]
    inst = Instance(rows, len(names), args.agg)
    print(f"{len(rows)} tuples, {len(inst.types)} distinct (group,value) types, groups={names}")
    print("start score:", round(inst.score(inst.init), 4))
    ms = {k: METHODS[k] for k in ("greedy", "greedy+clean", "beam4", "beam16+clean", "beam16+clean+swap")}
    sizes = {}
    for name, (k, mode) in ms.items():
        sizes[name] = run_method(inst, k, mode)
    ref = min(s for s, _ in sizes.values())
    for name, (size, dt) in sizes.items():
        print(f"{name:<20} deletions={size:<6} gap_vs_best={size / ref if ref else 1:.3f} time={dt:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--exp", choices=["e1", "e4", "e5", "all"])
    ap.add_argument("--n", type=int, default=100, help="instances per setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv"); ap.add_argument("--group-col"); ap.add_argument("--value-col")
    ap.add_argument("--agg", default="avg"); ap.add_argument("--group-order")
    ap.add_argument("--sample", type=int); ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()
    if a.toy: toy()
    if a.exp in ("e1", "all"): e1(a.n, a.seed)
    if a.exp in ("e4", "all"): e4(max(5, a.n // 4), a.seed)
    if a.exp in ("e5", "all"): e5(max(5, a.n // 4), a.seed)
    if a.csv: real(a)