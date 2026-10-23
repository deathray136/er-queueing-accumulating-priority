"""Validate the engine against M/M/1 non-preemptive priority theory.

The engine steps in whole minutes. To remove the time-step effect we run it with a 100x finer
tick (arrival prob 0.0015 per tick, mean service 500 ticks), where Bernoulli arrivals are
indistinguishable from Poisson. Waits are then converted back to the original minute scale.
"""
import numpy as np
from er_engine import generate_patients, simulate
lam, mean_s, K = 0.15, 5.0, 100          # lam, mean service in minutes; K ticks per minute
rho = lam*mean_s; p_hi = 0.3; sig1 = lam*p_hi*mean_s
W0 = lam*(2*mean_s**2)/2
theory = {"fifo": (W0/(1-rho),)*2, "strict": (W0/(1-sig1), W0/((1-sig1)*(1-rho)))}
res = {k: [[], []] for k in theory}
N, T, warm = 12, 1_000_000, 100_000
for seed in range(N):
    pts = generate_patients(seed, minutes=T, profile="const", scale=lam/K, dist="exponential", mean=mean_s*K)
    for pol in theory:
        for a, w, low in simulate(pts, pol):
            if a >= warm:
                res[pol][int(low)].append(w/K)
print(f"{'policy':<8}{'class':<6}{'theory (min)':>14}{'simulated':>12}{'diff %':>9}")
for pol in theory:
    for c, name in ((0, "high"), (1, "low")):
        s = np.mean(res[pol][c]); t = theory[pol][c]
        print(f"{pol:<8}{name:<6}{t:>14.2f}{s:>12.2f}{100*(s-t)/t:>9.1f}")
