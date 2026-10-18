"""Simulation experiments (revision 2). Run: python experiments.py [name ...] -> results/*.npz

Conventions
- Every parameter set is scored on test realizations that are independent of the
  realization used to choose its alignment lag (calibration) and, for learned decoders,
  of the training realizations.
- The first 2 s of every stationary run are discarded (burn-in > 6 tau_a).
- Spike times are transmitted and decoded at 1 ms resolution unless stated otherwise.
- Comparisons between encoders use the same stimulus and noise realizations (paired).
"""
import sys, os, time, json
import numpy as np
from multiprocessing import Pool
from core import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

P = dict(tau_c=0.03, noise=0.01, sig_ref=0.1, rate_ref=100.0, r_half=20.0, r_cap=150.0,
         T=20.0, burn=2.0)
P["m"] = 1.0 / (P["r_cap"] * 0.3)
SEEDS = list(range(101, 111))      # 10 test seeds
CAL = 7                            # calibration seed offset
NPROC = 2
SIGMAS = np.geomspace(0.01, 0.8, 16)
ENCS = ["F", "AS", "AM", "AI"]


def inp(sigma, seed, T=None, noise=None, tau_c=None):
    rng = np.random.default_rng(seed)
    return make_input(sigma, T or P["T"], rng, tau_c=tau_c or P["tau_c"],
                      noise=P["noise"] if noise is None else noise)


def x_std(sigma):
    x, _, _ = inp(sigma, 999, T=40, noise=0)
    return x.std()


def tune_rate(fn, target, lo, hi, iters=14):
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if fn(mid)["rate"] > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def enc_kw(enc):
    return dict(m=P["m"]) if enc == "AM" else {}


def reference_params():
    """Tune each encoder to 100 spikes/s at contrast 0.1 (tuning realization), then fix
    each decoder's lag on an independent calibration realization."""
    f = os.path.join(OUT, "ref_params.json")
    if os.path.exists(f):
        return json.load(open(f))
    s0 = x_std(P["sig_ref"])
    x, xi, _ = inp(P["sig_ref"], 1)
    ref = dict(sig0=s0, R=4 * s0, theta={}, lag={})
    for enc in ENCS + ["F_split", "AS_split"]:
        fn = lambda t, e=enc: run_event(x, xi, e, t, sig0=s0, decoder="gain", **enc_kw(e))
        ref["theta"][enc] = tune_rate(fn, P["rate_ref"], 1e-5, 0.3)
    xc, xic, _ = inp(P["sig_ref"], 2)
    for enc in ENCS + ["F_split", "AS_split"]:
        decs = ["gain", "fixed"] if enc == "AI" else ["spikes"]
        for dec in decs:
            r = run_event(xc, xic, enc, ref["theta"][enc], sig0=s0, decoder=dec, **enc_kw(enc))
            ref["lag"][enc + ("_" + dec if enc == "AI" else "")] = r["lag"]
    ref["lag"]["U"] = run_adc(xc, xic, 2e-3, 5, ref["R"])["lag"]
    json.dump(ref, open(f, "w"), indent=1)
    return ref


def run_ref(x, xi, key, ref, **kw):
    """Run a named encoder/decoder at its reference operating point and fixed lag."""
    if key == "U":
        return run_adc(x, xi, 2e-3, 5, ref["R"], lag=ref["lag"]["U"], **kw)
    if key.startswith("AI_"):
        enc, dec = "AI", key[3:]
        lag = ref["lag"]["AI_" + dec]
    else:
        enc, dec = key, "spikes"
        lag = ref["lag"][key]
    kk = {**enc_kw(enc), **kw}
    return run_event(x, xi, enc, ref["theta"][enc], decoder=dec, lag=lag, sig0=ref["sig0"], **kk)


# ============================================================================ E1 example
def e1_example():
    ref = reference_params()
    rng = np.random.default_rng(5)
    T = 9.0
    n = int(T / DT)
    sig = np.full(n, 0.04)
    sig[int(3 / DT):int(6 / DT)] = 0.3
    c = contrast_signal(n, 1.0, P["tau_c"], rng) * sig
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    x, xi = front_end(c), front_end(c + eta)
    out = {}
    for key in ("F", "AS", "U"):
        r = run_ref(x, xi, key, ref, burn=0.0)
        L = int(round(r["lag"] / DT))
        seg = []
        for a, b in ((0.5, 3), (3.5, 6), (6.5, 9)):
            m = np.zeros(n, bool); m[int(a / DT):int(b / DT)] = True
            seg.append(snr_at(x, r["est"], r["lag"], burn=0.0, mask=m))
        out[key] = (r, seg)
    np.savez_compressed(os.path.join(OUT, "e1.npz"), x=x, sig=sig,
                        spkF=out["F"][0]["spk"], spkA=out["AS"][0]["spk"],
                        thA=out["AS"][0]["th"], thF=out["F"][0]["th"],
                        estF=out["F"][0]["est"], estA=out["AS"][0]["est"], estU=out["U"][0]["est"],
                        recF=out["F"][0]["rec"], recA=out["AS"][0]["rec"],
                        snrs=json.dumps({k: v[1] for k, v in out.items()}))
    print("E1", {k: np.round(v[1], 1) for k, v in out.items()})


# ============================================================================ E2 frozen-stimulus rasters across contrasts
def e2_rasters():
    ref = reference_params()
    T, ntr = 1.2, 25
    n = int(T / DT)
    z = lowpass_noise(n, P["tau_c"], np.random.default_rng(77))
    res = {}
    for s in (0.02, 0.06, 0.18, 0.5):
        for key in ("F", "AS", "AI_fixed"):
            trains = []
            for tr in range(ntr):
                rng = np.random.default_rng(1000 + tr)
                # 2 s pre-adaptation with independent stimulus, then the frozen segment
                pre = lowpass_noise(int(2 / DT), P["tau_c"], rng)
                c = s * np.r_[pre, z]
                eta = P["noise"] * lowpass_noise(c.size, 0.002, rng, order=1)
                x, xi = front_end(c), front_end(c + eta)
                r = run_ref(x, xi, key, ref, burn=0.0)
                trains.append(r["spk"][-n:])
            res[f"{key}_{s}"] = np.array(trains)
    xz = front_end(np.r_[np.zeros(int(2 / DT)), z])[-n:]
    np.savez_compressed(os.path.join(OUT, "e2.npz"), xz=xz, **res)
    print("E2 done")


# ============================================================================ E3 rate-distortion (stationary)
def _e3_job(args):
    sigma, rep = args
    ref = reference_params()
    seed = SEEDS[rep]
    x, xi, _ = inp(sigma, seed)
    xc, xic, _ = inp(sigma, seed + 5000)       # calibration realization
    s = x.std()
    rows = []

    def ev(enc, th, dec="spikes", tag=0.0, **kw):
        lag = run_event(xc, xic, enc, th, decoder=dec, sig0=ref["sig0"], **kw)["lag"]
        r = run_event(x, xi, enc, th, decoder=dec, lag=lag, sig0=ref["sig0"], **kw)
        name = enc if enc != "AI" else "AI_" + dec
        rows.append((name, th, tag, r["snr"], r["rate"], r["bits"], r["lag"]))

    for th in np.geomspace(5e-4, 0.15, 20):
        ev("F", th)
    for rh in (5, 10, 20, 40, 80):
        for th in np.geomspace(1e-4, 0.05, 14):
            ev("AS", th, tag=rh, r_half=rh)
    for rc in (100, 150, 250, 400):
        for th in np.geomspace(5e-4, 0.08, 14):
            ev("AM", th, tag=rc, m=1 / (rc * 0.3))
    for th in np.geomspace(5e-4, 0.15, 20):
        ev("AI", th, "gain")
        ev("AI", th, "fixed")
    for B in range(1, 9):
        for D in (0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3, 16e-3):
            for nm, R in (("U_matched", 4 * s), ("U_fixed", ref["R"])):
                lag = run_adc(xc, xic, D, B, R)["lag"]
                r = run_adc(x, xi, D, B, R, lag=lag)
                rows.append((nm, B, D, r["snr"], r["rate"], r["bits"], r["lag"]))
    return sigma, seed, rows


def _e3_dq_job(dq):
    ref = reference_params()
    out = []
    for rep in range(3):
        x, xi, _ = inp(0.1, SEEDS[rep])
        xc, xic, _ = inp(0.1, SEEDS[rep] + 5000)
        for enc in ("F", "AS"):
            kw = dict(dq=dq)
            lag = run_event(xc, xic, enc, ref["theta"][enc], **kw)["lag"]
            r = run_event(x, xi, enc, ref["theta"][enc], lag=lag, **kw)
            out.append((dq, enc, rep, r["snr"], r["bits"] / max(r["rate"], 1e-9), r["rate"]))
    return out


def e3_rate_distortion():
    reference_params()
    jobs = [(s, rep) for s in (0.03, 0.1, 0.3) for rep in range(3)]
    with Pool(NPROC) as pool:
        res = pool.map(_e3_job, jobs)
        dqres = pool.map(_e3_dq_job, [1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3])
    out = [(sg, sd) + r for sg, sd, rows in res for r in rows]
    np.save(os.path.join(OUT, "e3.npy"), np.array(out, dtype=object), allow_pickle=True)
    np.save(os.path.join(OUT, "e3_dq.npy"), np.array([r for rr in dqres for r in rr], dtype=object),
            allow_pickle=True)
    print("E3 rows", len(out))


# ============================================================================ learned decoders
def train_learned():
    """Fit history-based gain decoders for AI (on its fixed-gain staircase) and AS (on its
    exact staircase), on training realizations independent of all test seeds."""
    f = os.path.join(OUT, "learned.npz")
    if os.path.exists(f):
        d = np.load(f)
        return {k: d[k] for k in d.files}
    ref = reference_params()
    pairs = {"AI_fixed": [], "AS": []}
    rich = {"AI_fixed": [], "AS": []}
    for s in np.geomspace(0.008, 1.0, 18):
        for k in range(2):
            x, xi, _ = inp(s, 20000 + int(s * 1e4) + 97 * k, T=12)
            for key in pairs:
                r = run_ref(x, xi, key, ref)
                L = int(round(r["lag"] / DT))
                b = int(2 / DT)
                pairs[key].append((r["rx"][b:x.size - L], r["est"][b:x.size - L], x[b + L:]))
                # rich decoder: features from the whole train, target aligned at the same lag
                rich[key].append((r["rx"], x[b:x.size - L], b + L))
    w = {}
    for key, pr in pairs.items():
        w[key] = GainDecoder().fit(pr).w
        th0 = ref["theta"]["AI" if key.startswith("AI") else key]
        w[key + "_rich"] = RichDecoder(theta0=th0).fit(rich[key]).w
    np.savez(f, **w)
    return w


def learned_predict(key, w, r, rich=False):
    """Returns an estimate aligned like r['est'] (scored at the same lag r['lag'])."""
    if rich:
        ref = reference_params()
        g = RichDecoder(theta0=ref["theta"]["AI" if key.startswith("AI") else key])
        g.w = w[key + "_rich"]
        est = g.predict(r["rx"])
        # the rich decoder was fitted to x aligned at lag L, i.e. est[t + L] ~ x[t]: same convention
        return est
    g = GainDecoder(); g.w = w[key]
    return g.predict(r["rx"], r["est"])


# ============================================================================ E4 contrast sweep
KEYS4 = ["U", "F", "AS", "AM", "AI_gain", "AI_fixed", "AI_learned", "AS_learned",
         "F_split", "AS_split", "AI_rich", "AS_rich"]


def _e4_job(args):
    sigma, rep = args
    ref = reference_params()
    w = train_learned()
    x, xi, _ = inp(sigma, SEEDS[rep])
    out = {}
    for key in ("U", "F", "AS", "AM", "AI_gain", "AI_fixed", "F_split", "AS_split"):
        r = run_ref(x, xi, key, ref)
        out[key] = (r["snr"], r["rate"])
        if key in ("AS", "AI_fixed"):
            est = learned_predict(key, w, r)
            nm = "AS_learned" if key == "AS" else "AI_learned"
            out[nm] = (snr_at(x, est, r["lag"]), r["rate"])
            est = learned_predict(key, w, r, rich=True)
            out[nm.replace("learned", "rich")] = (snr_at(x, est, r["lag"]), r["rate"])
    ceil = 10 * np.log10(x.var() / np.var(xi - x))
    return sigma, rep, out, ceil


def e4_contrast():
    reference_params(); train_learned()
    jobs = [(s, rep) for s in SIGMAS for rep in range(len(SEEDS))]
    with Pool(NPROC) as pool:
        res = pool.map(_e4_job, jobs)
    M = np.zeros((len(KEYS4), SIGMAS.size, len(SEEDS), 2))
    C = np.zeros((SIGMAS.size, len(SEEDS)))
    for sigma, rep, d, ceil in res:
        i = int(np.argmin(abs(SIGMAS - sigma)))
        C[i, rep] = ceil
        for k, key in enumerate(KEYS4):
            M[k, i, rep] = d[key]
    np.savez(os.path.join(OUT, "e4.npz"), M=M, C=C, sigmas=SIGMAS, keys=KEYS4)
    print("E4 done")


# ============================================================================ E5 identifiability (scale invariance)
def e5_scale():
    ref = reference_params()
    T = 6.0
    n = int(T / DT)
    rng = np.random.default_rng(55)
    z = lowpass_noise(n, P["tau_c"], rng)
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    base = 0.1
    amps = np.geomspace(0.1, 10, 21)
    s0 = ref["sig0"]
    xb = front_end(base * z)
    s_ss = xb[int(1 / DT):].std()
    cases = {}

    def trains(kind):
        out = []
        for a in amps:
            c = a * base * z
            if kind == "AI_noiseless_scaled_init":
                xi = front_end(c); r = encode(xi, "AI", ref["theta"]["AI"], sig0=s0, s_init=a * s_ss)
            elif kind == "AI_noiseless_fixed_init":
                xi = front_end(c); r = encode(xi, "AI", ref["theta"]["AI"], sig0=s0, s_init=s0)
            elif kind == "AI_noisy":
                xi = front_end(c + eta); r = encode(xi, "AI", ref["theta"]["AI"], sig0=s0)
            else:
                xi = front_end(c + eta); r = encode(xi, kind, ref["theta"][kind], **enc_kw(kind))
            out.append(r[0])
        return np.array(out)

    def dissim(tr, i0):
        """Fraction of spikes (both trains) without a same-polarity partner within 1 ms."""
        b = int(1 / DT)
        ref_idx = np.flatnonzero(tr[i0][b:])
        res = []
        for t in tr:
            idx = np.flatnonzero(t[b:])
            if idx.size + ref_idx.size == 0:
                res.append(0.0); continue
            def unmatched(A, Bv, pa, pb):
                if Bv.size == 0:
                    return A.size
                j = np.clip(np.searchsorted(Bv, A), 1, Bv.size - 1)
                near = np.where(abs(Bv[j] - A) < abs(Bv[j - 1] - A), j, j - 1)
                ok = (abs(Bv[near] - A) <= 10) & (pb[near] == pa)
                return (~ok).sum()
            pa, pb = t[b:][idx], tr[i0][b:][ref_idx]
            u = unmatched(idx, ref_idx, pa, pb) + unmatched(ref_idx, idx, pb, pa)
            res.append(u / (idx.size + ref_idx.size))
        return np.array(res)

    i0 = int(np.argmin(abs(amps - 1)))
    d = {"amps": amps}
    for kind in ("AI_noiseless_scaled_init", "AI_noiseless_fixed_init", "AI_noisy", "AS", "F", "AM"):
        tr = trains(kind)
        d["dis_" + kind] = dissim(tr, i0)
        d["rate_" + kind] = (tr[:, int(1 / DT):] != 0).sum(1) / (T - 1)
        sel = [int(np.argmin(abs(amps - a))) for a in (0.25, 0.5, 1, 2, 4)]
        d["ras_" + kind] = tr[sel][:, int(2 / DT):int(2.6 / DT)]
    d["ras_amps"] = np.array([0.25, 0.5, 1, 2, 4])
    np.savez_compressed(os.path.join(OUT, "e5.npz"), **d)
    print("E5 done", {k: np.round(v[[0, 5, 10, 15, 20]], 3) for k, v in d.items() if k.startswith("dis_")})


# ============================================================================ E6 contrast switching
def _e6_job(tau_a):
    ref = reference_params()
    rng = np.random.default_rng(int(tau_a * 1000) + 3)
    half, ncyc = 4.0, 40
    n = int(2 * half * ncyc / DT)
    t = np.arange(n) * DT
    sig = np.where((t % (2 * half)) < half, 0.04, 0.3)
    c = contrast_signal(n, 1.0, P["tau_c"], rng) * sig
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    x, xi = front_end(c), front_end(c + eta)
    res = {}
    keys = ("AS", "F") if tau_a == 0.3 else ("AS",)
    for key in keys:
        r = run_ref(x, xi, key, ref, tau_a=tau_a, burn=0.0)
        cyc = int(2 * half / DT)
        spk = r["spk"][: ncyc * cyc].reshape(ncyc, cyc)[1:]
        th = r["th"][: ncyc * cyc].reshape(ncyc, cyc)[1:]
        b = int(0.05 / DT)
        psth_cyc = (spk != 0).reshape(ncyc - 1, -1, b).sum(-1) / 0.05
        thm = th.reshape(ncyc - 1, -1, b).mean(-1).mean(0)
        w = int(0.2 / DT)
        L = int(round(r["lag"] / DT))
        est = np.r_[r["est"][L:], np.zeros(L)]
        err = ((x - est) ** 2)[: ncyc * cyc].reshape(ncyc, cyc)[1:]
        pw = (x ** 2)[: ncyc * cyc].reshape(ncyc, cyc)[1:]
        wsnr = 10 * np.log10(pw.reshape(ncyc - 1, -1, w).mean(-1).mean(0) /
                             err.reshape(ncyc - 1, -1, w).mean(-1).mean(0))
        res[key] = dict(psth=psth_cyc.mean(0), psth_cyc=psth_cyc, th=thm, wsnr=wsnr,
                        raster=spk[:30])         # full-resolution raster, 30 cycles
    return tau_a, res


def e6_switching():
    reference_params()
    taus = [0.1, 0.3, 1.0]
    with Pool(NPROC) as pool:
        res = pool.map(_e6_job, taus)
    d = {"taus": taus}
    for tau_a, r in res:
        for key, v in r.items():
            for nm, arr in v.items():
                d[f"{key}_{tau_a}_{nm}"] = arr
    np.savez_compressed(os.path.join(OUT, "e6.npz"), **d)
    print("E6 done")


# ============================================================================ E7 flash detection (paired)
def auc(f, b):
    return np.mean(f[:, None] > b[None, :]) + 0.5 * np.mean(f[:, None] == b[None, :])


def _e7_job(bg):
    ref = reference_params()
    rng = np.random.default_rng(int(bg * 1000) + 11)
    amps = np.geomspace(0.02, 1.0, 12)
    nfl, gap = 120, 0.6
    N = nfl * amps.size
    n = int(N * gap / DT) + int(1 / DT)
    c = contrast_signal(n, bg, P["tau_c"], rng)
    onsets = 1.0 + np.arange(N) * gap + 0.1 * rng.random(N)
    A = np.tile(amps, nfl); rng.shuffle(A)
    dur = int(0.03 / DT)
    for o, a in zip(onsets, A):
        c[int(o / DT):int(o / DT) + dur] += a
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    x, xi = front_end(c), front_end(c + eta)
    W = int(0.06 / DT)
    out = {"amps": amps, "A": A}
    for key in ("F", "AS", "AI_fixed"):
        r = run_ref(x, xi, key, ref, burn=0.0)
        on = (r["spk"] == 1)
        out[key + "_fl"] = np.array([on[int(o / DT):int(o / DT) + W].sum() for o in onsets])
        out[key + "_bl"] = np.array([on[int((o - 0.2) / DT):int((o - 0.2) / DT) + W].sum() for o in onsets])
        lat = []
        for o in onsets:
            k = np.flatnonzero(on[int(o / DT):int(o / DT) + W])
            lat.append(k[0] * DT if k.size else np.nan)
        out[key + "_lat"] = np.array(lat)
        # aligned rasters for two amplitudes, -100..+200 ms, 1 ms bins
        for a_sel in (0.05, 0.3):
            ai = amps[int(np.argmin(abs(amps - a_sel)))]
            idx = np.flatnonzero(A == ai)[:40]
            ras = np.array([r["spk"][int((onsets[i] - 0.1) / DT):int((onsets[i] - 0.1) / DT) + int(0.3 / DT)]
                            for i in idx])
            out[f"{key}_ras_{a_sel}"] = ras
    return bg, out


def _e7b_job(chunk):
    """Post-switch detection, all encoders on identical stimuli."""
    ref = reference_params()
    rng = np.random.default_rng(500 + chunk)
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.8])
    amp, hi_d, lo_d = 0.05, 2.0, 3.0
    reps = 6
    ncyc = 2 * delays.size * reps
    cyc = hi_d + lo_d
    n = int(ncyc * cyc / DT)
    t = np.arange(n) * DT
    sig = np.where((t % cyc) < hi_d, 0.3, 0.04)
    c = contrast_signal(n, 1.0, P["tau_c"], rng) * sig
    dl = np.tile(delays, 2 * reps); rng.shuffle(dl)
    has = np.tile(np.r_[np.ones(delays.size), np.zeros(delays.size)], reps).astype(bool)
    rng.shuffle(has)
    onsets = np.arange(ncyc) * cyc + hi_d + dl
    dur = int(0.03 / DT)
    for o, h in zip(onsets, has):
        if h:
            c[int(o / DT):int(o / DT) + dur] += amp
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    x, xi = front_end(c), front_end(c + eta)
    W = int(0.06 / DT)
    cnt = {}
    for key, kw in (("F", {}), ("AS_0.1", dict(tau_a=0.1)), ("AS_0.3", {}), ("AS_1.0", dict(tau_a=1.0))):
        r = run_ref(x, xi, key.split("_")[0], ref, burn=0.0, **kw)
        on = r["spk"] == 1
        cnt[key] = np.array([on[int(o / DT):int(o / DT) + W].sum() for o in onsets])
    return dl, has, cnt


def e7_flash():
    reference_params()
    with Pool(NPROC) as pool:
        res = pool.map(_e7_job, [0.04, 0.3])
        resb = pool.map(_e7b_job, range(14))
    d = {}
    rng = np.random.default_rng(0)
    for bg, out in res:
        amps, A = out["amps"], out["A"]
        d["amps"] = amps
        for key in ("F", "AS", "AI_fixed"):
            d[f"{key}_{bg}_auc"] = np.array([auc(out[key + "_fl"][A == a], out[key + "_bl"]) for a in amps])
            d[f"{key}_{bg}_lat"] = np.array([np.nanmedian(out[key + "_lat"][A == a]) for a in amps])
            for a_sel in (0.05, 0.3):
                d[f"{key}_{bg}_ras_{a_sel}"] = out[f"{key}_ras_{a_sel}"]
        # paired bootstrap of AUC differences (AS - F, AI_fixed - F), resampling flashes
        for other in ("AS", "AI_fixed"):
            diffs = np.zeros((300, amps.size))
            for b in range(300):
                ib = rng.integers(0, A.size, A.size)
                for j, a in enumerate(amps):
                    m = ib[A[ib] == a]
                    diffs[b, j] = (auc(out[other + "_fl"][m], out[other + "_bl"][ib]) -
                                   auc(out["F_fl"][m], out["F_bl"][ib]))
            d[f"diff_{other}_{bg}_lo"] = np.percentile(diffs, 2.5, 0)
            d[f"diff_{other}_{bg}_hi"] = np.percentile(diffs, 97.5, 0)
            d[f"diff_{other}_{bg}"] = d[f"{other}_{bg}_auc"] - d[f"F_{bg}_auc"]
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.8])
    dl = np.concatenate([r[0] for r in resb]); has = np.concatenate([r[1] for r in resb])
    cnt = {k: np.concatenate([r[2][k] for r in resb]) for k in resb[0][2]}
    d["delays"] = delays
    for k in cnt:
        d[f"ps_{k}_auc"] = np.array([auc(cnt[k][(dl == de) & has], cnt[k][(dl == de) & ~has]) for de in delays])
    # paired bootstrap of post-switch differences vs F (resample cycles within delay)
    for k in ("AS_0.1", "AS_0.3", "AS_1.0"):
        bs = np.zeros((500, delays.size))
        for j, de in enumerate(delays):
            fi = np.flatnonzero((dl == de) & has); bi = np.flatnonzero((dl == de) & ~has)
            for b in range(500):
                f = rng.choice(fi, fi.size); bb = rng.choice(bi, bi.size)
                bs[b, j] = auc(cnt[k][f], cnt[k][bb]) - auc(cnt["F"][f], cnt["F"][bb])
        d[f"ps_diff_{k}"] = d[f"ps_{k}_auc"] - d["ps_F_auc"]
        d[f"ps_diff_{k}_lo"] = np.percentile(bs, 2.5, 0)
        d[f"ps_diff_{k}_hi"] = np.percentile(bs, 97.5, 0)
    d["ps_n"] = np.array([((dl == de) & has).sum() for de in delays])
    np.savez_compressed(os.path.join(OUT, "e7.npz"), **d)
    print("E7 done")


# ============================================================================ E8 helps/hurts map
RATIOS = np.geomspace(1, 32, 9)
PERIODS = np.geomspace(0.1, 20, 9)
MAP_SEEDS = 5


def _switch_input(i, j, seed):
    ratio, per = RATIOS[i], PERIODS[j]
    gm = 0.08
    lo, hi = gm / np.sqrt(ratio), gm * np.sqrt(ratio)
    rng = np.random.default_rng(seed)
    T = max(30.0, 6 * per)
    n = int(T / DT)
    t = np.arange(n) * DT
    sig = np.where((t % per) < per / 2, lo, hi)
    c = contrast_signal(n, 1.0, P["tau_c"], rng) * sig
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    return front_end(c), front_end(c + eta), sig, lo, hi


def _e8_job(args):
    i, j = args
    target = 80.0
    out = []
    xt, xit, _, _, _ = _switch_input(i, j, 3000 + 10 * i + j)          # tuning + lag calibration
    th = {}
    lag = {}
    for enc in ("F", "AS"):
        th[enc] = tune_rate(lambda t, e=enc: run_event(xt, xit, e, t), target, 1e-5, 0.3, 12)
        lag[enc] = run_event(xt, xit, enc, th[enc])["lag"]
    for sd in range(MAP_SEEDS):
        x, xi, sig, lo, hi = _switch_input(i, j, 1000 + 10 * i + j + 7919 * sd)
        b = int(2 / DT)
        m_lo = (sig == lo); m_hi = (sig == hi)
        m_lo[:b] = m_hi[:b] = False
        row = []
        for enc in ("F", "AS"):
            r = run_event(x, xi, enc, th[enc], lag=lag[enc])
            row += [snr_at(x, r["est"], r["lag"], burn=0, mask=m_lo),
                    snr_at(x, r["est"], r["lag"], burn=0, mask=m_hi),
                    r["snr"], r["rate"]]
        out.append(row)
    return i, j, np.array(out)


def e8_map():
    reference_params()
    jobs = [(i, j) for i in range(RATIOS.size) for j in range(PERIODS.size)]
    with Pool(NPROC) as pool:
        res = pool.map(_e8_job, jobs)
    R = np.zeros((RATIOS.size, PERIODS.size, MAP_SEEDS, 8))
    for i, j, o in res:
        R[i, j] = o
    np.savez(os.path.join(OUT, "e8.npz"), R=R, ratios=RATIOS, periods=PERIODS,
             cols=np.array(["F_lo", "F_hi", "F_all", "F_rate", "AS_lo", "AS_hi", "AS_all", "AS_rate"]))
    print("E8 done")


# ============================================================================ E9 robustness
def e9_robustness():
    ref = reference_params()
    pf = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3])
    d = {"pf": pf}
    for leak in (0.2, 0.0):
        for key in ("F", "AS"):
            vals = np.zeros((pf.size, 5))
            for k, p in enumerate(pf):
                for sd in range(5):
                    x, xi, _ = inp(0.1, SEEDS[sd])
                    vals[k, sd] = run_ref(x, xi, key, ref, p_fail=p, tau_leak=leak,
                                          rng=np.random.default_rng(sd))["snr"]
            d[f"rob_{key}_leak{leak}"] = vals
    nz = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08])
    d["nz"] = nz
    for key in ("U", "F", "AS"):
        d["noise_" + key] = np.array([[run_ref(*inp(0.1, SEEDS[sd], noise=nl)[:2], key, ref)["snr"]
                                       for sd in range(5)] for nl in nz])
    d["noise_ceil"] = np.array([10 * np.log10(np.var(inp(0.1, SEEDS[0], noise=nl)[0]) /
                                              max(np.var(np.subtract(*inp(0.1, SEEDS[0], noise=nl)[1::-1])), 1e-30))
                                if nl > 0 else np.nan for nl in nz])
    np.savez(os.path.join(OUT, "e9.npz"), **d)
    print("E9 done")


# ============================================================================ E10 controls (revision 3)
def _tuned(enc, tau_c=None, **kw):
    """Threshold giving 100 spikes/s at contrast 0.1 for this encoder/bandwidth/parameters,
    and the lag fixed on an independent calibration realization."""
    s0 = inp(P["sig_ref"], 999, T=40, noise=0, tau_c=tau_c)[0].std()
    x, xi, _ = inp(P["sig_ref"], 1, tau_c=tau_c)
    kk = {**enc_kw(enc), **kw}
    dec = "gain" if enc == "AI" else "spikes"
    th = tune_rate(lambda t: run_event(x, xi, enc, t, sig0=s0, decoder=dec, **kk), P["rate_ref"], 1e-5, 0.3)
    xc, xic, _ = inp(P["sig_ref"], 2, tau_c=tau_c)
    lag = run_event(xc, xic, enc, th, sig0=s0, decoder=dec, **kk)["lag"]
    return th, lag, s0


def _sweep(enc, th, lag, s0, tau_c=None, seeds=3, **kw):
    kk = {**enc_kw(enc), **kw}
    dec = "gain" if enc == "AI" else "spikes"
    out = np.zeros((SIGMAS.size, seeds))
    for i, s in enumerate(SIGMAS):
        for sd in range(seeds):
            x, xi, _ = inp(s, SEEDS[sd], tau_c=tau_c)
            out[i, sd] = run_event(x, xi, enc, th, lag=lag, sig0=s0, decoder=dec, **kk)["snr"]
    return out


def _e10_sync(_):
    """Encoder/decoder desynchronization caused by spike-time quantization alone."""
    ref = reference_params()
    res = []
    x, xi, _ = inp(0.1, SEEDS[0])
    for enc in ("F", "AS", "AM"):
        spk, th, p = encode(xi, enc, ref["theta"][enc], **enc_kw(enc))
        exact = decode(spk, enc, ref["theta"][enc], p, th=th)
        for dq in (1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3):
            q = quantize_times(spk, dq)
            rec = decode(q, enc, ref["theta"][enc], p, th=th)
            # decoder reference (quantized times) vs encoder reference, both after the 3 ms output
            # smoothing, at the best alignment in [0, dq]
            se, sq = causal_smooth(exact, DT, p["tau_s"]), causal_smooth(rec, DT, p["tau_s"])
            b = int(2 / DT)
            best = np.inf
            for L in range(0, int(round(dq / DT)) + 1):
                e_ = se[b: se.size - L] - sq[b + L:]
                best = min(best, np.sqrt(np.mean(e_ ** 2)))
            rel = best / x[b:].std()
            # decoder threshold at received spikes vs encoder threshold at original spikes
            if enc != "F":
                kap = ref["theta"][enc] / p["r_half"] if enc == "AS" else 0.0
                m = p["m"] if enc == "AM" else 0.0
                thd = _decoder_theta(q, ref["theta"][enc], kap, m, p["tau_a"], 0 if enc == "AS" else 2)
                ie, iq = np.flatnonzero(spk), np.flatnonzero(q)
                n = min(ie.size, iq.size)
                th_err = np.median(np.abs(thd[iq[:n]] / th[ie[:n]] - 1))
            else:
                th_err = 0.0
            res.append((enc, dq, rel, th_err))
    return res


def _decoder_theta(rx, theta0, kappa, m, tau_a, mode):
    n = rx.size
    th = np.empty(n); r = 0.0; a = 0.0
    da = np.exp(-DT / tau_a)
    idx = set(np.flatnonzero(rx).tolist())
    for i in range(n):
        r *= da; a *= da
        t = theta0 + kappa * r if mode == 0 else theta0 + a
        th[i] = t
        if i in idx:
            r += 1.0 / tau_a; a += m * t
    return th


def _e10_steps(job):
    """Switching with different step sizes; returns cycle-averaged rates (50 ms bins)."""
    kind, lo, hi = job
    ref = reference_params()
    rng = np.random.default_rng(int(lo * 1e4) + int(hi * 1e3))
    half, ncyc = 4.0, 30
    n = int(2 * half * ncyc / DT)
    t = np.arange(n) * DT
    sig = np.where((t % (2 * half)) < half, lo, hi)
    c = contrast_signal(n, 1.0, P["tau_c"], rng) * sig
    eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
    x, xi = front_end(c), front_end(c + eta)
    out = {}
    for key in (("AS", "AI_fixed") if kind == "ai" else ("AS",)):
        r = run_ref(x, xi, key, ref, burn=0.0)
        cyc = int(2 * half / DT)
        spk = r["spk"][: ncyc * cyc].reshape(ncyc, cyc)[1:]
        b = int(0.05 / DT)
        out[key] = (spk != 0).reshape(ncyc - 1, -1, b).sum(-1).mean(0) / 0.05
    return job, out


def _e10_leak(job):
    arch, enc, leak = job
    e = enc if arch == "shared" else enc + "_split"
    kw = dict(tau_leak=leak) if arch == "shared" else dict(tau_leak_split=leak)
    th, lag, s0 = _tuned(e, **kw)
    return job, _sweep(e, th, lag, s0, **kw)


def _e10_band(job):
    tau_c, enc = job
    th, lag, s0 = _tuned(enc, tau_c=tau_c)
    return job, _sweep(enc, th, lag, s0, tau_c=tau_c)


def _e10_bandmap(job):
    tau_c, j, sd = job
    ratio, per = 13.45, PERIODS[j]
    gm = 0.08
    lo, hi = gm / np.sqrt(ratio), gm * np.sqrt(ratio)

    def mk(seed):
        rng = np.random.default_rng(seed)
        T = max(30.0, 6 * per)
        n = int(T / DT)
        t = np.arange(n) * DT
        sig = np.where((t % per) < per / 2, lo, hi)
        c = contrast_signal(n, 1.0, tau_c, rng) * sig
        eta = P["noise"] * lowpass_noise(n, 0.002, rng, order=1)
        return front_end(c), front_end(c + eta), sig
    xt, xit, _ = mk(4000 + j + int(tau_c * 1e4))
    x, xi, sig = mk(5000 + j + 31 * sd + int(tau_c * 1e4))
    b = int(2 / DT)
    m_lo = (sig == lo); m_hi = (sig == hi); m_lo[:b] = m_hi[:b] = False
    row = []
    for enc in ("F", "AS"):
        th = tune_rate(lambda t_: run_event(xt, xit, enc, t_), 80.0, 1e-5, 0.3, 12)
        lag = run_event(xt, xit, enc, th)["lag"]
        r = run_event(x, xi, enc, th, lag=lag)
        row += [snr_at(x, r["est"], lag, burn=0, mask=m_lo), snr_at(x, r["est"], lag, burn=0, mask=m_hi), r["snr"]]
    return job, row


def e10_controls():
    reference_params()
    d = {}
    with Pool(NPROC) as pool:
        sync = pool.map(_e10_sync, [0])[0]
        d["sync"] = np.array([(["F", "AS", "AM"].index(e), dq, rel, te) for e, dq, rel, te in sync])
        step_jobs = [("as", 0.04, h) for h in (0.08, 0.16, 0.32, 0.64)] + \
                    [("as", l, 0.32) for l in (0.01, 0.02, 0.08)] + [("ai", 0.04, 0.3)]
        for job, out in pool.map(_e10_steps, step_jobs):
            for k, v in out.items():
                d[f"step_{job[0]}_{job[1]}_{job[2]}_{k}"] = v
        leaks = (0.015, 0.03, 0.06, 0.2)
        for job, v in pool.map(_e10_leak, [(a, e, l) for a in ("shared", "split") for e in ("F", "AS") for l in leaks]):
            d[f"leak_{job[0]}_{job[1]}_{job[2]}"] = v
        for job, v in pool.map(_e10_band, [(tc, e) for tc in (0.01, 0.03, 0.1) for e in ("F", "AS", "AM", "AI")]):
            d[f"band_{job[0]}_{job[1]}"] = v
        bm = pool.map(_e10_bandmap, [(tc, j, sd) for tc in (0.01, 0.03, 0.1) for j in range(PERIODS.size) for sd in range(3)])
    for tc in (0.01, 0.03, 0.1):
        d[f"bandmap_{tc}"] = np.array([[row for (t_, j_, s_), row in bm if t_ == tc and j_ == j] for j in range(PERIODS.size)])
    d["sigmas"] = SIGMAS; d["periods"] = PERIODS; d["leaks"] = np.array(leaks)
    np.savez(os.path.join(OUT, "e10.npz"), **d)
    print("E10 done")


# ============================================================================ E11 regularization sensitivity
def e11_lambda():
    """Refit both learned decoders over a grid of ridge strengths; report test SNR."""
    ref = reference_params()
    lams = np.array([1e-9, 1e-7, 1e-6, 1e-5, 1e-3, 1e-1])
    pairs = {"AI_fixed": [], "AS": []}
    rich = {"AI_fixed": [], "AS": []}
    for s in np.geomspace(0.008, 1.0, 18):
        for k in range(2):
            x, xi, _ = inp(s, 20000 + int(s * 1e4) + 97 * k, T=12)
            for key in pairs:
                r = run_ref(x, xi, key, ref)
                L = int(round(r["lag"] / DT)); b = int(2 / DT)
                pairs[key].append((r["rx"][b:x.size - L], r["est"][b:x.size - L], x[b + L:]))
                rich[key].append((r["rx"], x[b:x.size - L], b + L))
    sig_test = SIGMAS[::3]
    tests = {key: [] for key in pairs}
    for s in sig_test:
        for sd in range(3):
            x, xi, _ = inp(s, SEEDS[sd])
            for key in pairs:
                tests[key].append((s, x, run_ref(x, xi, key, ref)))
    out = {"lams": lams, "sig": sig_test}
    for key in pairs:
        th0 = ref["theta"]["AI" if key.startswith("AI") else key]
        # rate-gain decoder
        g = GainDecoder()
        A = np.concatenate([g.features(rx, st)[::10] for rx, st, _ in pairs[key]])
        y = np.concatenate([t[::10] for _, _, t in pairs[key]])
        AtA, Aty, sc = A.T @ A, A.T @ y, np.trace(A.T @ A) / A.shape[1]
        # per-spike decoder
        rd = RichDecoder(theta0=th0)
        Ar = np.concatenate([rd.features(rx)[b::10][: t.size // 10] for rx, t, b in rich[key]])
        yr = np.concatenate([t[::10][: t.size // 10] for _, t, _ in rich[key]])
        RtR, Rty, scr = Ar.T @ Ar, Ar.T @ yr, np.trace(Ar.T @ Ar) / Ar.shape[1]
        Ftest = [(s, x, r, g.features(r["rx"], r["est"]), rd.features(r["rx"])) for s, x, r in tests[key]]
        for nm, M_, v_, c_, idx in (("rate", AtA, Aty, sc, 3), ("rich", RtR, Rty, scr, 4)):
            res = np.zeros((lams.size, sig_test.size))
            for li, lam in enumerate(lams):
                w = np.linalg.solve(M_ + lam * c_ * np.eye(M_.shape[0]), v_)
                for j, s in enumerate(sig_test):
                    vals = [snr_at(x, F_[idx - 3] @ w if False else (fr if idx == 3 else fp) @ w, r["lag"])
                            for (s2, x, r, fr, fp) in Ftest if s2 == s]
                    res[li, j] = np.mean(vals)
            out[f"{key}_{nm}"] = res
    np.savez(os.path.join(OUT, "e11.npz"), **out)
    print("E11 done", {k: v.max(1).round(1) for k, v in out.items() if k not in ("lams", "sig")})


ALL = dict(e1=e1_example, e2=e2_rasters, e3=e3_rate_distortion, e4=e4_contrast,
           e5=e5_scale, e6=e6_switching, e7=e7_flash, e8=e8_map, e9=e9_robustness, e10=e10_controls, e11=e11_lambda)

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    print("ref", reference_params(), flush=True)
    for nm in names:
        t = time.time()
        ALL[nm]()
        print(nm, f"{time.time() - t:.0f}s", flush=True)
