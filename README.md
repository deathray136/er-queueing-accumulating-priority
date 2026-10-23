# ER queueing simulation: accumulating priority vs static priority

Code and result tables for the paper "Accumulating Priority versus Static Priority Queueing for Emergency Department Triage under Time-Varying Demand: A Validated Simulation Study".

## Run

Requires Python 3 with `numpy` and `matplotlib` (`pip install numpy matplotlib`).

```
python validate.py
python er_experiments.py
```

- `validate.py` checks the simulator against exact M/M/1 priority formulas.
- `er_experiments.py` runs the experiments and writes the tables (`table_*.csv`) and figures (`fig_*.png`). It can take several minutes.

## Files

- `er_engine.py`: the simulator
- `validate.py`: validation against theory
- `er_experiments.py`: experiments A, B, C
- `table_*.csv`, `fig_*.png`: results from the paper
