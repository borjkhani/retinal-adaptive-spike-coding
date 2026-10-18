"""Collect every number quoted in the manuscript into results/numbers.json."""
import os, json
import numpy as np
import figures as F
R = F.R
out = {}
ref = json.load(open(os.path.join(R, "ref_params.json"))); out["ref"] = ref
d = F.load("e1.npz"); out["e1_snr"] = json.loads(str(d["snrs"]))
out["e1_rate"] = {k: float(np.count_nonzero(d[k]) / 9) for k in ("spkF", "spkA")}
out["e1_theta"] = dict(low=float(d["thA"][10000:30000].mean()), high=float(d["thA"][40000:60000].mean()))
d = F.load("e2.npz")
out["e2_rates"] = {k: float((d[k] != 0).sum(1).mean() / 1.2) for k in d.files if k != "xz"}
# E3
arr = F.load("e3.npy")
sig = arr[:, 0].astype(float); enc = arr[:, 2]; p1 = arr[:, 3].astype(float); p2 = arr[:, 4].astype(float)
snr, rate, bits, lag = (arr[:, k].astype(float) for k in (5, 6, 7, 8))
e3 = {}
for s in (0.03, 0.1, 0.3):
    row = {}
    for key in ("F", "AS", "AM", "AI_gain", "AI_fixed", "U_matched", "U_fixed"):
        m = (sig == s) & (enc == key)
        params = np.unique(np.stack([p1[m], p2[m]], 1), axis=0)
        xs, xb, ys = [], [], []
        for pr in params:
            mm = m & (p1 == pr[0]) & (p2 == pr[1])
            xs.append(rate[mm].mean()); xb.append(bits[mm].mean()); ys.append(snr[mm].mean())
        xs, xb, ys = map(np.array, (xs, xb, ys))
        ok = xs > 2
        if not key.startswith("U"):
            px, py = F.pareto(xs[ok], ys[ok])
            row[key + "_at100Hz"] = float(np.interp(2, np.log10(px), py))
            row[key + "_bps"] = float(np.median(xb[ok] / xs[ok]))
        qx, qy = F.pareto(xb[ok], ys[ok])
        row[key + "_at500b"] = float(np.interp(np.log10(500), np.log10(qx), qy))
        row[key + "_max"] = float(ys.max())
    e3[str(s)] = row
out["e3"] = e3
dq = F.load("e3_dq.npy")
out["dq"] = {f"{k}_{v}": [float(dq[(dq[:, 1] == k) & (dq[:, 0].astype(float) == v), c].astype(float).mean()) for c in (3, 4, 5)]
             for k in ("F", "AS") for v in sorted(set(dq[:, 0].astype(float)))}
m = (sig == 0.1) & np.isin(enc, ["F", "AS"]) & (rate > 5)
out["lag_min_ms"] = float(lag[m].min() * 1e3)
# E4
d = F.load("e4.npz"); M, s, keys = d["M"], d["sigmas"], list(d["keys"])
out["e4_snr"] = {k: M[keys.index(k), :, :, 0].mean(1).round(2).tolist() for k in keys}
out["e4_rate"] = {k: M[keys.index(k), :, :, 1].mean(1).round(1).tolist() for k in keys}
out["sigmas"] = s.tolist()
dr = F.fig4()
out["dr"] = {k: dict(lo=v["lo"], hi=v["hi"], dec=float(np.log10(v["hi"] / v["lo"])) if np.isfinite(v["lo"]) else None,
                     open_hi=v["open_hi"], seeds_mean=float(np.mean(v["seeds"])), seeds_sd=float(np.std(v["seeds"], ddof=1)))
           for k, v in dr.items()}
for a, b in (("AS", "F"), ("AM", "F"), ("AS_split", "F_split")):
    dd = np.array(dr[a]["seeds"]) - np.array(dr[b]["seeds"])
    out[f"dr_diff_{a}_{b}"] = [float(dd.mean()), float(dd.mean() - 2.262 * dd.std(ddof=1) / np.sqrt(dd.size)),
                               float(dd.mean() + 2.262 * dd.std(ddof=1) / np.sqrt(dd.size))]
for k in ("AS", "AM"):
    dif = M[keys.index(k), :, :, 0] - M[keys.index("F"), :, :, 0]
    out[f"pairdiff_{k}"] = dif.mean(1).round(2).tolist()
iA = keys.index("AS"); iF = keys.index("F"); iM = keys.index("AM")
out["slope_full"] = {k: float(np.polyfit(np.log10(s), np.log10(M[keys.index(k), :, :, 1].mean(1)), 1)[0]) for k in ("AS", "F", "AM")}
out["slope"] = {k: float(np.polyfit(np.log10(s[4:12]), np.log10(M[keys.index(k), 4:12, :, 1].mean(1)), 1)[0]) for k in ("AS", "F", "AM")}
# E5
d = F.load("e5.npz")
out["e5"] = {k: d[k].round(3).tolist() for k in d.files if k.startswith(("rate_", "dis_"))}
out["e5_amps"] = d["amps"].round(3).tolist()
# E6
out["switch"] = F.switching_stats()
# E7
d = F.load("e7.npz")
out["e7"] = {k: np.asarray(d[k]).round(4).tolist() for k in d.files if "ras" not in k}
# E8
dm, st = F.map_stats()
out["map"] = {nm: dict(max=float(v["mu"].max()), min=float(v["mu"].min()),
                      argmax=[float(dm["ratios"][np.unravel_index(v["mu"].argmax(), v["mu"].shape)[0]]),
                              float(dm["periods"][np.unravel_index(v["mu"].argmax(), v["mu"].shape)[1]])],
                      argmin=[float(dm["ratios"][np.unravel_index(v["mu"].argmin(), v["mu"].shape)[0]]),
                              float(dm["periods"][np.unravel_index(v["mu"].argmin(), v["mu"].shape)[1]])],
                      n_sig_pos=int((v["lo"] > 0).sum()), n_sig_neg=int((v["hi"] < 0).sum()))
              for nm, v in st.items()}
out["map_epoch"] = st["epoch"]["mu"].round(2).tolist()
out["map_overall"] = st["overall"]["mu"].round(2).tolist()
Rr = dm["R"]
out["map_eg"] = dict(F_lo=Rr[-1, -1, :, 0].mean(), AS_lo=Rr[-1, -1, :, 4].mean(), F_hi=Rr[-1, -1, :, 1].mean(),
                     AS_hi=Rr[-1, -1, :, 5].mean())
# E9
d = F.load("e9.npz")
out["e9"] = {k: np.asarray(d[k]).mean(-1).round(2).tolist() if d[k].ndim > 1 else np.asarray(d[k]).round(3).tolist()
             for k in d.files}
json.dump(out, open(os.path.join(R, "numbers.json"), "w"), indent=1, default=float)
