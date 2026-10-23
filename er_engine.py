"""
Generic single/multi-queue ED simulation engine used by the paper experiments.

Design
  - Time advances in 1-minute steps. At most one arrival per minute (Bernoulli), as in the
    original model; the arrival probability can follow a daily profile or be constant.
  - Each patient gets an acuity class and a service time at arrival time, drawn from a
    per-replication random stream. Every policy is replayed on the SAME patients
    (common random numbers), so differences between policies are not sampling noise.
  - A server starts a patient as soon as it is free and someone is waiting. Waiting time is the
    time from arrival to start of service.
  - After the last arrival the system keeps running until every patient is served (no censoring).
"""
import random
import numpy as np

P_LOW = 0.70


def arrival_prob(minute, profile, scale):
    if profile == "daily":
        hour = (minute // 60) % 24
        return scale * (0.05 + 0.25 * np.sin(np.pi * hour / 24) ** 2)
    return scale  # constant probability per minute


def draw_service(rng, dist, mean):
    if dist == "deterministic":
        return mean
    if dist == "exponential":
        return rng.expovariate(1.0 / mean)
    if dist == "lognormal":  # coefficient of variation 1.5-ish, same mean
        cv2 = 2.25
        sigma2 = np.log(1 + cv2)
        mu = np.log(mean) - sigma2 / 2
        return rng.lognormvariate(mu, np.sqrt(sigma2))
    raise ValueError(dist)


def generate_patients(seed, minutes=1440, profile="daily", scale=1.0, dist="exponential", mean=5.0):
    rng = random.Random(seed)
    pts = []
    for m in range(minutes):
        if rng.random() < arrival_prob(m, profile, scale):
            low = rng.random() < P_LOW
            pts.append({"arrival": m, "low": low, "prio": 2 if low else 1,
                        "service": draw_service(rng, dist, mean)})
    return pts


def _pick(queue, policy, now, alpha):
    if policy == "apq":
        best = min(queue, key=lambda p: (p["prio"] - alpha * (now - p["arrival"]), p["arrival"]))
    elif policy == "strict":
        best = min(queue, key=lambda p: (p["prio"], p["arrival"]))
    else:  # fifo
        best = queue[0]
    queue.remove(best)
    return best


def simulate(patients, policy, alpha=0.0, split=None):
    """Return list of (arrival_minute, wait, is_low).

    policy: 'fifo' | 'strict' | 'apq' (needs alpha) | 'fast_track'
    For 'fast_track' the total capacity of the single server is split into two servers:
    the service time of a patient is scaled by 1/share, share = 0.3 for the main (high-acuity)
    server and 0.7 for the fast (low-acuity) server, so total capacity is unchanged.
    """
    split = split or {"main": 1 - P_LOW, "fast": P_LOW}
    queues = {"main": [], "fast": []}
    free_at = {"main": 0.0, "fast": 0.0}
    out = []
    i, minute, n = 0, 0, len(patients)
    ft = policy == "fast_track"
    inner = "fifo" if ft else policy
    names = ("main", "fast") if ft else ("main",)
    while i < n or queues["main"] or queues["fast"]:
        while i < n and patients[i]["arrival"] == minute:
            p = patients[i]
            queues["fast" if (ft and p["low"]) else "main"].append(p)
            i += 1
        for name in names:
            if queues[name] and free_at[name] <= minute:
                p = _pick(queues[name], inner, minute, alpha)
                svc = p["service"] / split[name] if ft else p["service"]
                free_at[name] = minute + svc
                out.append((p["arrival"], minute - p["arrival"], p["low"]))
        minute += 1
    return out
