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
python3 aod_beam.py --toy                                   # worked example: greedy 3, others 2
python3 aod_beam.py --exp all --n 100 --seed 0 | tee results.txt
python3 make_figures.py                                     # creates figures/ and tables
```
The full run takes a few minutes. Seeds are fixed, so the results match the report.

## Files
| File | Description |
|---|---|
| `aod_beam.py` | Greedy, BeamRepair, cleanup, exact search, experiments E1/E4/E5 |
| `make_figures.py` | Figures and summary tables for the report |
| `results.txt` | Console output of the full run |
| `e1_results.csv`, `e4_results.csv`, `e5_results.csv` | Raw per-instance results |
| `figures/` | Figures (PNG) and tables (`tables.md`, `tables.tex`) |
| `AOD_BeamRepair_Track2_Report.docx` | The report |