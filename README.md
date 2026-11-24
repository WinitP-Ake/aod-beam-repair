# Beam Search and Local-Search Cleanup for AOD Repair

Individual Short Project, Track #2 (Improvement)

Improves the greedy heuristic HeurRepair from:
Agmon et al., "Analyzing Deviations from Monotonic Trends through Database Repair",
SIGMOD/PACMMOD 2026 (original code: https://github.com/shunita/aod).

## Methods
- **BeamRepair**: keeps the k best partial repairs per step (k = 1 equals HeurRepair).
- **Cleanup**: re-inserts deleted tuples that are no longer needed (never makes a repair worse).

## Requirements
Python 3.9+ and matplotlib:
```bash
pip3 install matplotlib
```

## How to reproduce
```bash
python3 aod_beam.py --toy                                         # worked example: greedy 3, others 2
python3 aod_beam.py --exp all --n 100 --seed 0 | tee results.txt  # E1, E2, E3 (Tables 4-6, A1)
python3 make_figures.py                                           # Figures 1-4 and tables.md / tables.tex
python3 choose_k.py | tee choose_k.txt                            # Section 4.5 (Tables 7-8, Figure 5)
```
The full run takes a few minutes. Seeds are fixed, so the results match the report.

## Files
| File | Description |
| --- | --- |
| `aod_beam.py` | Greedy (k = 1), BeamRepair, cleanup, exact BFS, and experiments E1–E3 |
| `make_figures.py` | Summary figures (Figures 1–4) and Markdown/LaTeX tables |
| `choose_k.py` | Beam-width study, lower-bound certification, and iterative doubling (Section 4.5, Figure 5) |
| `e1_results.csv`, `e4_results.csv`, `e5_results.csv` | Per-instance raw results for E1, E2, and E3 |
| `choose_k_results.csv` | Per-table results for k = 1 to 128 |
| `results.txt`, `choose_k.txt` | Console logs of the full experimental runs |
| `figures/` | Generated PNG figures and summary tables (`tables.md`, `tables.tex`) |
| `README.md` | Reproduction guide |
