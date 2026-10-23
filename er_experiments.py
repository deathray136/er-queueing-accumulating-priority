"""Experiments for the paper: aging-rate sweep, load sweep, service-time robustness."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from er_engine import generate_patients, simulate

N_A, N_BC = 300, 150
BASE = 1000

def run(configs, n, **gen):
    """configs: dict name -> (policy, alpha). Returns dict name -> stats over replications."""
    store = {k: {"hi": [], "lo": [], "hi_rep": [], "lo_rep": []} for k in configs}
    for rep in range(n):
        pts = generate_patients(BASE + rep, **gen)
        for name, (pol, alpha) in configs.items():
            out = simulate(pts, pol, alpha=alpha)
            hi = [w for a, w, l in out if not l]; lo = [w for a, w, l in out if l]
            store[name]["hi"] += hi; store[name]["lo"] += lo
            store[name]["hi_rep"].append(np.mean(hi)); store[name]["lo_rep"].append(np.mean(lo))
    res = {}
    for k, s in store.items():
        ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
        res[k] = dict(hi=np.mean(s["hi"]), lo=np.mean(s["lo"]), hi_ci=ci(s["hi_rep"]), lo_ci=ci(s["lo_rep"]),
                      hi_p95=np.percentile(s["hi"], 95), lo_p95=np.percentile(s["lo"], 95), lo_max=np.max(s["lo"]))
    return res

def fmt(r): return f"{r['hi']:>7.1f} {r['hi_p95']:>7.1f} {r['lo']:>7.1f} {r['lo_p95']:>7.1f} {r['lo_max']:>7.0f}"
HDR = f"{'':<22}{'hi mean':>8}{'hi p95':>8}{'lo mean':>8}{'lo p95':>8}{'lo max':>8}"

# ---------------- A: aging-rate sweep
alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
cfg = {"fifo": ("fifo", 0)}
cfg.update({f"apq a={a}": ("apq", a) for a in alphas})
A = run(cfg, N_A, dist="exponential")
print("\nA. Aging-rate sweep (exponential service, mean 5 min, daily arrival profile)\n" + HDR)
for k, r in A.items(): print(f"{k:<22}{fmt(r)}")
with open("table_A_aging_sweep.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config", "high_mean", "high_p95", "low_mean", "low_p95", "low_max", "high_ci", "low_ci"])
    for k, r in A.items(): w.writerow([k] + [round(float(r[x]), 2) for x in ("hi", "hi_p95", "lo", "lo_p95", "lo_max", "hi_ci", "lo_ci")])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
xs = [A[f"apq a={a}"]["lo"] for a in alphas]; ys = [A[f"apq a={a}"]["hi"] for a in alphas]
ax[0].errorbar(xs, ys, xerr=[A[f"apq a={a}"]["lo_ci"] for a in alphas], yerr=[A[f"apq a={a}"]["hi_ci"] for a in alphas],
               fmt="-o", color="#0275d8", capsize=3, label="APQ (aging rate $\\alpha$)")
for a, x, y in zip(alphas, xs, ys): ax[0].annotate(f"$\\alpha$={a}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[0].errorbar(A["fifo"]["lo"], A["fifo"]["hi"], xerr=A["fifo"]["lo_ci"], yerr=A["fifo"]["hi_ci"], fmt="s", color="#d9534f", capsize=3, label="FIFO")
ax[0].set_xlabel("Mean wait, low-acuity patients (min)"); ax[0].set_ylabel("Mean wait, high-acuity patients (min)")
ax[0].set_title("Trade-off between the two acuity classes"); ax[0].grid(True, linestyle=":", alpha=.6); ax[0].legend()
xa = np.arange(len(alphas))
ax[1].plot(xa, [A[f"apq a={a}"]["lo_p95"] for a in alphas], "-^", color="#f0ad4e", label="Low-acuity, 95th percentile")
ax[1].plot(xa, [A[f"apq a={a}"]["lo"] for a in alphas], "-o", color="#0275d8", label="Low-acuity, mean")
ax[1].plot(xa, [A[f"apq a={a}"]["hi"] for a in alphas], "-s", color="#5cb85c", label="High-acuity, mean")
ax[1].set_xticks(xa); ax[1].set_xticklabels([str(a) for a in alphas])
ax[1].set_xlabel("Aging rate $\\alpha$ (priority points per minute; 0 = strict priority)"); ax[1].set_ylabel("Waiting time (min)")
ax[1].set_title("Effect of the aging rate"); ax[1].grid(True, linestyle=":", alpha=.6); ax[1].legend()
fig.tight_layout(); fig.savefig("fig_aging_tradeoff.png", dpi=300); plt.close(fig)

# ---------------- B: load sweep
pol_cfg = {"FIFO": ("fifo", 0), "Strict priority": ("strict", 0), "APQ a=0.02": ("apq", 0.02),
           "APQ a=0.05": ("apq", 0.05), "Fast-track (equal cap.)": ("fast_track", 0)}
scales = [0.5, 0.7, 0.85, 1.0, 1.15]
B = {s: run(pol_cfg, N_BC, scale=s, dist="exponential") for s in scales}
print("\nB. Load sweep (arrival scale; scale 1.0 = original profile, mean utilisation 0.875)")
rows = []
for s in scales:
    print(f"\nscale {s}  (mean utilisation {0.875*s:.2f}, peak {1.5*s:.2f})\n" + HDR)
    for k, r in B[s].items():
        print(f"{k:<22}{fmt(r)}"); rows.append([s, k] + [round(float(r[x]), 2) for x in ("hi", "hi_p95", "lo", "lo_p95", "lo_max")])
with open("table_B_load_sweep.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scale", "policy", "high_mean", "high_p95", "low_mean", "low_p95", "low_max"]); w.writerows(rows)
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
styles = {"FIFO": ("#d9534f", "o", "--"), "Strict priority": ("#f0ad4e", "D", "-."), "APQ a=0.02": ("#0275d8", "^", "-"),
          "APQ a=0.05": ("#6f42c1", "v", "-"), "Fast-track (equal cap.)": ("#5cb85c", "s", ":")}
util = [0.875 * s for s in scales]
for k, (c, m, ls) in styles.items():
    ax[0].plot(util, [B[s][k]["hi"] for s in scales], color=c, marker=m, linestyle=ls, label=k)
    ax[1].plot(util, [B[s][k]["lo"] for s in scales], color=c, marker=m, linestyle=ls, label=k)
for a, t in zip(ax, ("High-acuity patients", "Low-acuity patients")):
    a.set_title(t); a.set_xlabel("Mean utilisation over the day"); a.grid(True, linestyle=":", alpha=.6)
ax[0].set_ylabel("Mean waiting time (min)"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig_load_sweep.png", dpi=300); plt.close(fig)

# ---------------- C: service-time distribution
C = {d: run(pol_cfg, N_BC, dist=d) for d in ("deterministic", "exponential", "lognormal")}
print("\nC. Service-time distribution (same mean 5 min)")
rows = []
for d in C:
    print(f"\n{d}\n" + HDR)
    for k, r in C[d].items():
        print(f"{k:<22}{fmt(r)}"); rows.append([d, k] + [round(float(r[x]), 2) for x in ("hi", "hi_p95", "lo", "lo_p95", "lo_max")])
with open("table_C_service_dist.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["service_dist", "policy", "high_mean", "high_p95", "low_mean", "low_p95", "low_max"]); w.writerows(rows)
