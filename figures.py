"""Figures (revision 2). Run: python figures.py [fig1 ...] -> figures/*.pdf, *.png

Layout rule: every figure uses matplotlib's constrained layout; panel letters are drawn
as left-aligned axes titles and legends sit outside the data area, so text never overlaps
data, ticks or neighbouring panels.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results")
FD = os.path.join(BASE, "figures")
os.makedirs(FD, exist_ok=True)
DT = 1e-4

COL = dict(U="#7a7974", F="#eb6834", AS="#2a78d6", AM="#6250d6", AI_gain="#1baf7a",
           AI_fixed="#1baf7a", AI_learned="#0d6e4a", AS_learned="#123f78",
           AI_rich="#0d6e4a", AS_rich="#123f78",
           F_split="#eb6834", AS_split="#2a78d6",
           ink="#0b0b0b", ink2="#52514e", muted="#a3a29c", grid="#e6e5e0", off="#52514e")
LAB = dict(U="Uniform ADC", F="Fixed threshold", AS="Spike-history (additive)",
           AM="Spike-history (multiplicative)", AI_gain="Input-driven, gain-informed decoder",
           AI_fixed="Input-driven, fixed-gain decoder", AI_learned="Input-driven, rate-gain decoder",
           AS_learned="Spike-history (additive), rate-gain decoder",
           AI_rich="Input-driven, learned per-spike decoder", AS_rich="Spike-history (additive), learned per-spike decoder",
           F_split="Fixed, separate ON/OFF units", AS_split="Spike-history, separate ON/OFF units")
LS = dict(AI_fixed="--", AI_learned="-.", AS_learned=":", F_split="--", AS_split="--",
          AI_rich=(0, (5, 1.5, 1, 1.5, 1, 1.5)), AS_rich=(0, (1, 1)))
TAU_SHADES = {0.1: "#9fc3ef", 0.3: "#2a78d6", 1.0: "#123f78"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 7.4,
    "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": COL["ink2"], "axes.linewidth": 0.6, "xtick.color": COL["ink2"],
    "ytick.color": COL["ink2"], "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": COL["ink"], "lines.linewidth": 1.3, "legend.frameon": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
})
MM = 1 / 25.4
W2 = 180 * MM


def title(ax, letter, text=""):
    ax.set_title(f"$\\bf{{{letter}}}$   {text}" if text else f"$\\bf{{{letter}}}$", loc="left",
                 color=COL["ink"], pad=4)


def ygrid(ax):
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FD, name + ".png"))
    fig.savefig(os.path.join(FD, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def raster(ax, trains, dt_bin, t0=0.0, on_col=COL["AS"], lw=0.5, gap=0.15):
    """trains: (ntrials, nbins) with +1/-1/0. ON ticks upper half of each row, OFF lower."""
    for k, tr in enumerate(trains):
        on = np.flatnonzero(tr == 1) * dt_bin + t0
        off = np.flatnonzero(tr == -1) * dt_bin + t0
        y = len(trains) - 1 - k
        ax.vlines(on, y + 0.5, y + 1 - gap, color=on_col, lw=lw)
        ax.vlines(off, y + gap, y + 0.5, color=COL["off"], lw=lw)
    ax.set_ylim(0, len(trains))


def binr(tr, k):
    """Downsample +1/-1/0 trains by k samples (ON wins ties), trimming the remainder."""
    n = (tr.shape[1] // k) * k
    tb = tr[:, :n].reshape(tr.shape[0], -1, k)
    return np.where((tb == 1).any(-1), 1, np.where((tb == -1).any(-1), -1, 0))


def load(name):
    return np.load(os.path.join(R, name), allow_pickle=True)


# ============================================================================ Figure 1
def _box(ax, x, y, w, h, text, fc="#f4f3ef", ec=None, fs=6.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=fc, ec=ec or COL["ink2"], lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=COL["ink"], linespacing=1.25)


def _arr(ax, p, q, color=None, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, lw=0.8,
                                 color=color or COL["ink2"], linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def schematic(ax):
    """Block diagram on a 0..1 x 0..1 canvas; boxes laid out on a grid with clear gaps."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y1, h1 = 0.74, 0.34       # main row
    y2, h2 = 0.17, 0.26       # feedback row
    cols = dict(c=0.065, k=0.205, cmp=0.385, spk=0.565, dec=0.745, out=0.925)
    wid = dict(c=0.10, k=0.10, cmp=0.17, spk=0.10, dec=0.15, out=0.07)
    _box(ax, cols["c"], y1, wid["c"], h1, "stimulus\n$c(t)$\n+ noise $\\eta(t)$")
    _box(ax, cols["k"], y1, wid["k"], h1, "temporal\nfilter\n$k(t)$")
    _box(ax, cols["cmp"], y1, wid["cmp"], h1, "comparator\n$|x - \\hat{x}| > \\theta(t)$\nrefractory 2 ms",
         fc="#eaf1fb", ec=COL["AS"])
    _box(ax, cols["spk"], y1, wid["spk"], h1, "ON / OFF\nspike times\n(1 ms grid)")
    _box(ax, cols["dec"], y1, wid["dec"], h1, "decoder\n$\\pm\\theta$ steps,\nleak, smoothing")
    _box(ax, cols["out"], y1, wid["out"], h1, "$\\hat{x}(t)$", fs=7)
    order = ["c", "k", "cmp", "spk", "dec", "out"]
    for a, b in zip(order[:-1], order[1:]):
        _arr(ax, (cols[a] + wid[a] / 2 + 0.004, y1), (cols[b] - wid[b] / 2 - 0.004, y1))
    xm = (cols["k"] + wid["k"] / 2 + cols["cmp"] - wid["cmp"] / 2) / 2
    ax.text(xm, y1 + 0.035, "$x(t)$", ha="center", va="bottom", fontsize=6.5, color=COL["ink2"])
    # feedback row: three boxes, each exactly under a column, gaps >= 0.02
    _box(ax, 0.205, y2, 0.15, h2, "input-driven gain (AI)\n$\\theta = \\theta_0\\,\\hat{s}(t)/s_0$",
         fc="#e8f6f0", ec=COL["AI_gain"], fs=6.0)
    _box(ax, 0.385, y2, 0.15, h2, "shared reference $\\hat{x}$\n(leaky, $\\tau_{leak}$ = 200 ms)", fs=6.0)
    _box(ax, 0.565, y2, 0.17, h2, "spike-history rate $r(t)$ (AS)\n$\\theta=\\theta_0(1+r/r_{1/2})$",
         fc="#eaf1fb", ec=COL["AS"], fs=6.0)
    yt, yb = y1 - h1 / 2, y2 + h2 / 2      # bottom of main row / top of feedback row
    cx = cols["cmp"]
    # comparator <-> shared reference (reference sits directly below the comparator)
    _arr(ax, (cx - 0.025, yt), (cx - 0.025, yb))          # spike: reference += +/- theta
    _arr(ax, (cx + 0.015, yb), (cx + 0.015, yt))          # reference -> comparator
    ax.text(cx - 0.032, (yt + yb) / 2, "$\\pm\\theta$", ha="right", va="center", fontsize=6, color=COL["ink2"])
    ax.text(cx + 0.022, (yt + yb) / 2, "$\\hat{x}$", ha="left", va="center", fontsize=6, color=COL["ink2"])
    # spikes -> rate (straight down), rate -> comparator (diagonal, right of the reference)
    _arr(ax, (cols["spk"], yt), (cols["spk"], yb), color=COL["AS"])
    _arr(ax, (cols["spk"] - 0.075, yb), (cx + 0.07, yt), color=COL["AS"])
    # x -> input RMS (down-left) and input RMS -> comparator (up-right); the two never cross
    _arr(ax, (xm - 0.012, y1 - 0.012), (0.205 + 0.03, yb), color=COL["AI_gain"], ls="--")
    _arr(ax, (0.205 + 0.075, yb), (cx - cols["cmp"] + cx - wid["cmp"] / 2 + 0.005, yt),
         color=COL["AI_gain"], ls="--")
    ax.text(0.84, y2, "The decoder recomputes the\nspike-history threshold from the\n"
            "spikes it receives; it cannot see\nthe input-driven gain $\\hat{s}(t)$.",
            ha="center", va="center", fontsize=5.9, color=COL["ink2"], style="italic", linespacing=1.3)


def fig1():
    d = load("e1.npz")
    snrs = json.loads(str(d["snrs"]))
    fig = plt.figure(figsize=(W2, 170 * MM))
    sf = fig.subfigures(2, 1, height_ratios=[1.0, 2.9], hspace=0.02)
    ax = sf[0].subplots()
    title(ax, "A", "Encoder and decoder")
    schematic(ax)
    gs = sf[1].add_gridspec(5, 2, height_ratios=[0.55, 1.25, 0.5, 0.5, 0.8], width_ratios=[3.1, 1])
    T = d["x"].size * DT
    t = np.arange(d["x"].size) * DT
    k = 10
    ax0 = sf[1].add_subplot(gs[0, 0])
    ax0.fill_between(t, 0, d["sig"], color=COL["muted"], alpha=0.35, lw=0)
    ax0.plot(t, d["sig"], color=COL["ink2"], lw=0.8)
    ax0.set_ylabel("RMS\ncontrast"); ax0.set_xlim(0, T); ax0.set_ylim(0, 0.78)
    ax0.set_yticks([0, 0.3])
    for (a, b), i in zip(((0, 3), (3, 6), (6, 9)), range(3)):
        ax0.text((a + b) / 2, 0.55, f"SNR: fixed {snrs['F'][i]:.1f} dB\nadaptive {snrs['AS'][i]:.1f} dB",
                 ha="center", va="center", fontsize=5.9, color=COL["ink"], linespacing=1.2)
    ax0.tick_params(labelbottom=False)
    title(ax0, "B", "Contrast steps")
    ax1 = sf[1].add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(t[::k], d["x"][::k], color=COL["ink"], lw=0.6)
    ax1.plot(t[::k], d["estF"][::k], color=COL["F"], lw=0.7, alpha=0.9)
    ax1.plot(t[::k], d["estA"][::k], color=COL["AS"], lw=0.7, alpha=0.9)
    z0, z1 = 4.0, 4.3
    ax1.axvspan(z0, z1, color=COL["muted"], alpha=0.25, lw=0)
    ax1.set_ylabel("signal")
    ax1.tick_params(labelbottom=False)
    title(ax1, "C", "Target $x(t)$ (black) and reconstructions")
    handles = [Line2D([], [], color=COL["ink"], lw=1, label="target"),
               Line2D([], [], color=COL["F"], lw=1, label=LAB["F"]),
               Line2D([], [], color=COL["AS"], lw=1, label=LAB["AS"])]
    for gi, (key, name, letter) in enumerate((("spkF", "F", "D"), ("spkA", "AS", "E"))):
        ax = sf[1].add_subplot(gs[2 + gi, 0], sharex=ax0)
        raster(ax, [d[key]], DT, on_col=COL[name], lw=0.3, gap=0.08)
        ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(["OFF", "ON"])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.tick_params(labelbottom=False)
        title(ax, letter, f"{LAB[name]} spikes ({np.count_nonzero(d[key]) / T:.0f} spikes/s)")
    ax = sf[1].add_subplot(gs[4, 0], sharex=ax0)
    ax.plot(t[::k], d["thA"][::k] * 1e3, color=COL["AS"], lw=1.0)
    ax.plot(t[::k], d["thF"][::k] * 1e3, color=COL["F"], lw=1.0)
    ax.set_ylabel("threshold\n($10^{-3}$)"); ax.set_xlabel("time (s)")
    title(ax, "F", "Thresholds")
    # zoom
    axz = sf[1].add_subplot(gs[1:4, 1])
    m = (t >= z0) & (t <= z1)
    tz = (t[m] - z0) * 1e3
    axz.plot(tz, d["x"][m], color=COL["ink"], lw=1.0)
    axz.step(tz, d["recF"][m], where="post", color=COL["F"], lw=0.8)
    axz.step(tz, d["recA"][m], where="post", color=COL["AS"], lw=0.8)
    lo = min(d["x"][m].min(), d["recF"][m].min(), d["recA"][m].min())
    hi = max(d["x"][m].max(), d["recF"][m].max(), d["recA"][m].max())
    span = hi - lo
    for key, frac, c in (("spkF", 0.10, COL["F"]), ("spkA", 0.20, COL["AS"])):
        sp = d[key][m]
        yy = lo - frac * span
        axz.vlines(tz[sp != 0], yy - 0.035 * span, yy + 0.035 * span, color=c, lw=0.5)
    axz.set_ylim(lo - 0.27 * span, hi + 0.05 * span)
    axz.set_xlim(0, 300)
    axz.set_yticks([])
    axz.spines["left"].set_visible(False)
    axz.set_xlabel("time from 4.0 s (ms)")
    title(axz, "G", "Zoom (shaded in C)")
    axl = sf[1].add_subplot(gs[4, 1]); axl.axis("off")
    axl.legend(handles=handles + [Line2D([], [], color=COL["ink2"], lw=0.8, label="D, E: ON (top) / OFF (bottom)"),
                         Line2D([], [], color=COL["ink2"], lw=0.8, label="G: spikes, fixed (upper row),\n    adaptive (lower row)")],
               loc="center", fontsize=6.0)
    save(fig, "Fig1_model_example")


# ============================================================================ Figure 2 rasters across contrasts
def fig2():
    d = load("e2.npz")
    contrasts = [0.02, 0.06, 0.18, 0.5]
    keys = ["F", "AS", "AI_fixed"]
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(len(keys) + 1, len(contrasts), height_ratios=[0.45] + [1] * len(keys))
    n = d["xz"].size
    t = np.arange(n) * DT
    win = (t >= 0.2) & (t < 1.2)
    letters = iter("ABCDEFGHIJKLMNOP")
    for j, s in enumerate(contrasts):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(t[win][::5] - 0.2, s * d["xz"][win][::5] / 0.1, color=COL["ink"], lw=0.6)
        ax.set_xlim(0, 1); ax.set_xticks([])
        lim = 0.5 * np.abs(d["xz"][win]).max() / 0.1 * 1.05
        ax.set_ylim(-lim, lim)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        title(ax, next(letters), f"contrast {s:g}")
        for i, key in enumerate(keys):
            ax = fig.add_subplot(gs[i + 1, j])
            tr = d[f"{key}_{s}"][:, win]
            tb = binr(tr, 10)
            raster(ax, tb, 1e-3, on_col=COL[key], lw=0.45, gap=0.1)
            rate = (tr != 0).sum(1).mean() / 1.0   # rate over the displayed 1 s window
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(LAB[key].replace(", ", "\n").replace("Input-driven", "Input-driven gain"),
                              fontsize=6.4)
            if i == len(keys) - 1:
                ax.set_xlabel("time (s)")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_title(f"{rate:.0f} spikes/s", loc="right", fontsize=6.2, color=COL["ink2"], pad=2)
    fig.suptitle("Top: filtered stimulus $x$ on a common scale. Rasters: 25 trials, independent input noise; ON coloured, OFF grey",
                 fontsize=6.6, color=COL["ink2"])
    save(fig, "Fig2_rasters_contrast")


# ============================================================================ Figure 3 rate-distortion
def pareto(xv, yv):
    o = np.argsort(xv)
    xs, ys = np.asarray(xv)[o], np.asarray(yv)[o]
    keep, best = [], -np.inf
    for i in range(xs.size):
        if ys[i] > best:
            keep.append(i); best = ys[i]
    return xs[keep], ys[keep]


def fig3():
    arr = load("e3.npy")
    sig = arr[:, 0].astype(float); enc = arr[:, 2]
    p1, p2 = arr[:, 3].astype(float), arr[:, 4].astype(float)
    snr, rate, bits, lag = (arr[:, k].astype(float) for k in (5, 6, 7, 8))
    dq = load("e3_dq.npy")
    fig = plt.figure(figsize=(W2, 125 * MM))
    gs = fig.add_gridspec(2, 4)
    letters = iter("ABCDEFGH")
    keys_ev = ["F", "AS", "AM", "AI_gain", "AI_fixed"]
    for j, s in enumerate((0.03, 0.1, 0.3)):
        for row in range(2):
            ax = fig.add_subplot(gs[row, j])
            xv = rate if row == 0 else bits
            keys = keys_ev + (["U_matched", "U_fixed"] if row == 1 else [])
            for key in keys:
                m = (sig == s) & (enc == key)
                params = np.unique(np.stack([p1[m], p2[m]], 1), axis=0)
                xs, ys = [], []
                for pr in params:
                    mm = m & (p1 == pr[0]) & (p2 == pr[1])
                    xs.append(xv[mm].mean()); ys.append(snr[mm].mean())
                xs, ys = np.array(xs), np.array(ys)
                ok = xs > 2
                ck = "U" if key.startswith("U") else key
                ax.scatter(xs[ok], ys[ok], s=1.6, color=COL[ck], alpha=0.25, lw=0, zorder=1)
                px, py = pareto(xs[ok], ys[ok])
                ls = ":" if key == "U_fixed" else LS.get(key, "-")
                ax.plot(px, py, color=COL[ck], ls=ls, lw=1.2, zorder=2)
            ax.set_xscale("log")
            ax.set_xlim((4, 600) if row == 0 else (30, 8000))
            ax.set_ylim(-3, 32)
            ygrid(ax)
            ax.set_xlabel("spike rate (spikes/s)" if row == 0 else "code rate (bits/s)")
            if j == 0:
                ax.set_ylabel("test SNR (dB)")
            title(ax, next(letters) if True else "", f"contrast {s:g}")
    # timing resolution
    ax = fig.add_subplot(gs[0, 3])
    for key in ("F", "AS"):
        m = dq[:, 1] == key
        dqs = np.unique(dq[m, 0].astype(float))
        mu = [dq[m & (dq[:, 0].astype(float) == v), 3].astype(float).mean() for v in dqs]
        ax.plot(dqs * 1e3, mu, color=COL[key], marker="o", ms=2.5)
    ax.set_xscale("log"); ax.set_xlabel("timing resolution $\\Delta q$ (ms)")
    ax.set_ylabel("test SNR (dB)"); ax.set_ylim(16.5, 18); ygrid(ax)
    title(ax, "G", "Timing resolution")
    ax2 = fig.add_subplot(gs[1, 3])
    for key in ("F", "AS"):
        m = dq[:, 1] == key
        dqs = np.unique(dq[m, 0].astype(float))
        mu = [dq[m & (dq[:, 0].astype(float) == v), 4].astype(float).mean() for v in dqs]
        ax2.plot(dqs * 1e3, mu, color=COL[key], marker="o", ms=2.5)
    ax2.set_xscale("log"); ax2.set_xlabel("timing resolution $\\Delta q$ (ms)")
    ax2.set_ylabel("bits per spike"); ygrid(ax2)
    title(ax2, "H", "Cost per spike")
    handles = [Line2D([], [], color=COL[k], ls=LS.get(k, "-"), lw=1.3, label=LAB[k]) for k in keys_ev] + \
              [Line2D([], [], color=COL["U"], lw=1.3, label="Uniform ADC, range matched"),
               Line2D([], [], color=COL["U"], ls=":", lw=1.3, label="Uniform ADC, fixed range"),
               Line2D([], [], color=COL["ink2"], marker="o", ls="", ms=2, alpha=0.5, label="all operating points")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "Fig3_rate_distortion")


# ============================================================================ Figure 4 dynamic range
def dyn_range(sig, mu, thr=10):
    ls_ = np.log10(sig)
    ok = mu >= thr
    if not ok.any():
        return np.nan, np.nan, False, False
    idx = np.flatnonzero(ok)
    a, b = idx[0], idx[-1]
    lo = ls_[a] if a == 0 else np.interp(thr, [mu[a - 1], mu[a]], [ls_[a - 1], ls_[a]])
    hi = ls_[b] if b == sig.size - 1 else np.interp(thr, [mu[b + 1], mu[b]], [ls_[b + 1], ls_[b]])
    return 10 ** lo, 10 ** hi, a == 0, b == sig.size - 1


DR_SHORT = lambda k: (LAB[k].replace("Input-driven, ", "AI, ").replace("Spike-history", "SH")
                      .replace(", rate-gain decoder", ", rate-gain dec.")
                      .replace(", learned per-spike decoder", ", per-spike dec.")
                      .replace(", separate ON/OFF units", ", split").replace("Fixed threshold", "Fixed"))


def draw_dr(ax, order, M, s, keys, letter):
    dr_out = {}
    for j, k in enumerate(order):
        i = keys.index(k)
        per = []
        for sd in range(M.shape[2]):
            lo, hi, ol, oh = dyn_range(s, M[i, :, sd, 0])
            per.append(np.log10(hi / lo) if np.isfinite(lo) else 0.0)
        mu = M[i, :, :, 0].mean(1)
        lo, hi, ol, oh = dyn_range(s, mu)
        dr_out[k] = dict(lo=lo, hi=hi, open_lo=bool(ol), open_hi=bool(oh), seeds=list(per))
        y = len(order) - 1 - j
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=COL[k], lw=4, solid_capstyle="butt")
            ax.text(1.25, y, ("≥" if (ol or oh) else "") + f"{np.log10(hi / lo):.2f}", va="center",
                    fontsize=6, color=COL["ink"])
        else:
            ax.text(0.1, y, "never reaches 10 dB", va="center", ha="center", fontsize=5.8,
                    color=COL["ink2"], style="italic")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DR_SHORT(k) for k in order[::-1]], fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(0.008, 3)
    ax.set_xlabel("contrast range with SNR ≥ 10 dB")
    ax.text(1.25, len(order) - 0.35, "decades", fontsize=6, color=COL["ink2"])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    title(ax, letter, "Dynamic range")
    return dr_out


ORDER_ALL = ["U", "F", "F_split", "AI_fixed", "AI_learned", "AI_rich", "AS_split", "AM", "AS", "AS_learned",
             "AS_rich", "AI_gain"]


def fig4():
    d = load("e4.npz")
    M, C, s, keys = d["M"], d["C"], d["sigmas"], list(d["keys"])
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 2)
    main = ["U", "F", "AS", "AM", "AI_gain", "AI_fixed"]
    ax = fig.add_subplot(gs[0, 0])
    for k in main:
        i = keys.index(k)
        mu = M[i, :, :, 0].mean(1); se = M[i, :, :, 0].std(1, ddof=1) / np.sqrt(M.shape[2])
        ax.plot(s, mu, color=COL[k], ls=LS.get(k, "-"), marker="o", ms=1.8)
        ax.fill_between(s, mu - 1.96 * se, mu + 1.96 * se, color=COL[k], alpha=0.15, lw=0)
    ax.plot(s, C.mean(1), color=COL["muted"], lw=0.9, ls=":")
    ax.axhline(10, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.axvline(0.1, color=COL["muted"], lw=0.6)
    ax.set_xscale("log"); ax.set_ylim(-2, 30)
    ax.set_xlabel("RMS contrast"); ax.set_ylabel("test SNR (dB)"); ygrid(ax)
    title(ax, "A", "Accuracy across contrast")
    ax = fig.add_subplot(gs[0, 1])
    for k in ["F", "AS", "AM", "AI_gain"]:
        ax.plot(s, M[keys.index(k), :, :, 1].mean(1), color=COL[k], marker="o", ms=1.8)
    ax.axhline(500, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(2, 700)
    ax.set_xlabel("RMS contrast"); ax.set_ylabel("spike rate (spikes/s)"); ygrid(ax)
    title(ax, "B", "Firing rate (dashed: refractory limit)")
    ax = fig.add_subplot(gs[1, 0])
    for k in ["AS", "AM"]:
        dif = M[keys.index(k), :, :, 0] - M[keys.index("F"), :, :, 0]
        mu = dif.mean(1); se = dif.std(1, ddof=1) / np.sqrt(dif.shape[1])
        ax.plot(s, mu, color=COL[k], marker="o", ms=1.8)
        ax.fill_between(s, mu - 2.262 * se, mu + 2.262 * se, color=COL[k], alpha=0.18, lw=0)
    ax.axhline(0, color=COL["ink2"], lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("RMS contrast"); ax.set_ylabel("paired ΔSNR vs fixed (dB)")
    ygrid(ax)
    title(ax, "C", "Paired difference from fixed threshold, 95% CI")
    ax = fig.add_subplot(gs[1, 1])
    draw_dr(ax, ["U", "F", "AI_fixed", "AM", "AS", "AI_gain"], M, s, keys, "D")
    handles = [Line2D([], [], color=COL[k], ls=LS.get(k, "-"), lw=1.3, label=LAB[k]) for k in main]
    handles.append(Line2D([], [], color=COL["muted"], ls=":", lw=1, label="SNR of the noisy encoder input"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "Fig4_dynamic_range")
    # all variants (used by stats and Supplementary Figure S1)
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    dr_out = draw_dr(ax2, ORDER_ALL, M, s, keys, "x")
    plt.close(fig2)
    return dr_out


def figS1():
    d = load("e4.npz")
    M, s, keys = d["M"], d["sigmas"], list(d["keys"])
    e11 = load("e11.npz")
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    for k in ["AI_gain", "AI_rich", "AI_learned", "AI_fixed", "AS", "AS_rich", "AS_learned"]:
        ax.plot(s, M[keys.index(k), :, :, 0].mean(1), color=COL[k], ls=LS.get(k, "-"), marker="o", ms=1.8)
    ax.axhline(10, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_ylim(-12, 30)
    ax.set_xlabel("RMS contrast"); ax.set_ylabel("test SNR (dB)"); ygrid(ax)
    title(ax, "A", "Decoders that see only spikes")
    ax = fig.add_subplot(gs[0, 1])
    lams = e11["lams"]
    for key, nm, c, ls in (("AS", "rich", COL["AS_rich"], (0, (1, 1))), ("AS", "rate", COL["AS_learned"], ":"),
                           ("AI_fixed", "rich", COL["AI_rich"], LS["AI_rich"]), ("AI_fixed", "rate", COL["AI_learned"], "-.")):
        v = e11[f"{key}_{nm}"].max(1)
        ax.plot(lams, v, color=c, ls=ls, marker="o", ms=2)
        ax.text(lams[2], v[2] + 0.7, {"AS_rich": "SH, per-spike", "AS_rate": "SH, rate-gain",
                                     "AI_fixed_rich": "AI, per-spike", "AI_fixed_rate": "AI, rate-gain"}[f"{key}_{nm}"],
                fontsize=5.8, color=COL["ink2"], ha="left")
    ax.axhline(10, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_xlabel("ridge strength λ (× mean diagonal)")
    ax.set_ylabel("best test SNR over contrasts (dB)"); ax.set_ylim(0, 25); ygrid(ax)
    ax.axvline(1e-6, color=COL["muted"], lw=0.6)
    title(ax, "B", "Sensitivity to regularization (line: value used)")
    ax = fig.add_subplot(gs[1, 0])
    for k in ["F", "AS", "F_split", "AS_split"]:
        ax.plot(s, M[keys.index(k), :, :, 0].mean(1), color=COL[k], ls=LS.get(k, "-"), marker="o", ms=1.8)
    ax.axhline(10, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_ylim(-2, 30)
    ax.set_xlabel("RMS contrast"); ax.set_ylabel("test SNR (dB)"); ygrid(ax)
    title(ax, "C", "Separate ON/OFF units (dashed)")
    ax = fig.add_subplot(gs[1, 1])
    draw_dr(ax, ORDER_ALL, M, s, keys, "D")
    keyset = ["F", "AS", "AI_gain", "AI_fixed", "AI_learned", "AI_rich", "AS_learned", "AS_rich", "F_split", "AS_split"]
    handles = [Line2D([], [], color=COL[k], ls=LS.get(k, "-"), lw=1.3, label=LAB[k]) for k in keyset]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "FigS1_decoders_architecture")


# ============================================================================ Figure 5 identifiability
def fig5():
    d = load("e5.npz")
    amps = d["amps"]
    fig = plt.figure(figsize=(W2, 115 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], width_ratios=[1, 1, 1, 1])
    kinds = [("AI_noiseless_scaled_init", "Input-driven (noiseless)", COL["AI_gain"]),
             ("AS", "Spike-history, additive", COL["AS"]),
             ("F", "Fixed threshold", COL["F"]),
             ("AM", "Spike-history, mult.", COL["AM"])]
    for j, (kind, lab, c) in enumerate(kinds):
        ax = fig.add_subplot(gs[0, j])
        tr = d["ras_" + kind]
        tb = binr(tr, 10)
        raster(ax, tb, 1e-3, on_col=c, lw=0.45, gap=0.08)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels([f"×{a:g}" for a in d["ras_amps"][::-1]] if j == 0 else [])
        if j == 0:
            ax.set_ylabel("input amplitude")
        ax.set_xlabel("time (s)"); ax.set_xlim(0, 0.6); ax.set_xticks([0, 0.2, 0.4])
        title(ax, "ABCD"[j], lab)
    ax = fig.add_subplot(gs[1, 0:2])
    for kind, lab, c, ls in (("AI_noiseless_scaled_init", "input-driven, noiseless, scaled initial state", COL["AI_gain"], "-"),
                             ("AI_noiseless_fixed_init", "input-driven, noiseless, fixed initial state", COL["AI_gain"], ":"),
                             ("AI_noisy", "input-driven, with input noise", COL["AI_gain"], "--"),
                             ("AS", LAB["AS"], COL["AS"], "-"), ("AM", LAB["AM"], COL["AM"], "-"),
                             ("F", LAB["F"], COL["F"], "-")):
        ax.plot(amps, d["rate_" + kind], color=c, ls=ls, marker="o", ms=1.6, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1, 700)
    ax.set_xlabel("input amplitude (× reference)"); ax.set_ylabel("spike rate (spikes/s)"); ygrid(ax)
    title(ax, "E", "Rate against a constant scale factor")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    ax = fig.add_subplot(gs[1, 2])
    e10 = load("e10.npz")
    ps_ai = e10["step_ai_0.04_0.3_AI_fixed"]; ps_as = e10["step_ai_0.04_0.3_AS"]
    tt = (np.arange(ps_ai.size) + 0.5) * 0.05
    ax.axvspan(4, 8, color=COL["grid"], alpha=0.6, lw=0)
    ax.plot(tt, ps_as, color=COL["AS"], lw=0.9)
    ax.plot(tt, ps_ai, color=COL["AI_gain"], lw=0.9)
    ax.set_xlim(0, 8); ax.set_xlabel("time in cycle (s)"); ax.set_ylabel("rate (spikes/s)"); ygrid(ax)
    title(ax, "F", "Switch 0.04 | 0.30")
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(amps, d["dis_AI_noiseless_scaled_init"], color=COL["AI_gain"], marker="o", ms=1.6)
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("amplitude (× reference)"); ax.set_ylabel("unmatched fraction")
    ygrid(ax)
    title(ax, "G", "Unmatched spikes (A)")
    save(fig, "Fig5_identifiability")


# ============================================================================ Figure 6 switching
def fig6():
    d = load("e6.npz")
    taus = [0.1, 0.3, 1.0]
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 0.8, 0.8])
    half = 4.0
    n1 = int(1 / DT)
    col = 0
    for key in ("F", "AS"):
        ras = d[f"{key}_0.3_raster"]
        up = ras[:, int(3 / DT):int(6 / DT)]                       # increase at t = 0
        down = np.concatenate([ras[:-1, int(7 / DT):], ras[1:, :int(2 / DT)]], axis=1)  # decrease
        up = up[:down.shape[0]]                                     # same 29 cycles in all panels
        for seg, lab in ((up, "contrast increase"), (down, "contrast decrease")):
            ax = fig.add_subplot(gs[0, col])
            tb = binr(seg, 20)
            raster(ax, tb, 0.002, t0=-1.0, on_col=COL[key], lw=0.45, gap=0.06)
            shade = (0, 2) if lab == "contrast increase" else (-1, 0)
            ax.axvspan(*shade, color=COL["grid"], alpha=0.6, lw=0, zorder=0)
            ax.axvline(0, color=COL["ink2"], lw=0.6)
            ax.set_xlim(-1, 2); ax.set_xticks([-1, 0, 1]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("cycle")
            ax.set_xlabel("time from switch (s)")
            title(ax, "ABCD"[col], f"{'Fixed' if key == 'F' else 'Spike-history'}, {lab.split()[1]}")
            col += 1
    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(half, 2 * half, color=COL["grid"], alpha=0.6, lw=0)
    for tau in taus:
        ps = d[f"AS_{tau}_psth"]; tt = (np.arange(ps.size) + 0.5) * 0.05
        ax.plot(tt, ps, color=TAU_SHADES[tau], lw=1.0)
    ps = d["F_0.3_psth"]
    ax.plot(tt, ps, color=COL["F"], lw=1.0)
    ax.set_xlim(0, 8); ax.set_ylabel("rate (spikes/s)"); ygrid(ax)
    ax.tick_params(labelbottom=False)
    title(ax, "E", "Cycle-averaged rate (shaded: contrast 0.30)")
    ax2 = fig.add_subplot(gs[2, :], sharex=ax)
    ax2.axvspan(half, 2 * half, color=COL["grid"], alpha=0.6, lw=0)
    for tau in taus:
        w = d[f"AS_{tau}_wsnr"]; tw = (np.arange(w.size) + 0.5) * 0.2
        ax2.plot(tw, w, color=TAU_SHADES[tau], lw=1.0, marker="o", ms=1.5)
    w = d["F_0.3_wsnr"]
    ax2.plot(tw, w, color=COL["F"], lw=1.0, marker="o", ms=1.5)
    ax2.set_xlabel("time in cycle (s)"); ax2.set_ylabel("SNR, 200 ms windows (dB)"); ygrid(ax2)
    title(ax2, "F", "Windowed reconstruction accuracy")
    handles = [Line2D([], [], color=TAU_SHADES[t], lw=1.3, label=f"spike-history, $\\tau_a$ = {t:g} s") for t in taus] + \
              [Line2D([], [], color=COL["F"], lw=1.3, label="fixed threshold")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    save(fig, "Fig6_switching")


def switching_stats():
    from scipy.optimize import curve_fit
    d = load("e6.npz")
    out = {}
    for tau in (0.1, 0.3, 1.0):
        ps = d[f"AS_{tau}_psth"]; tt = (np.arange(ps.size) + 0.5) * 0.05
        th = d[f"AS_{tau}_th"]
        res = {}
        for name, a, b in (("up", 4.0, 8.0), ("down", 0.0, 4.0)):
            # fit from the rate extremum (peak after an increase, trough after a decrease)
            m0 = (tt > a) & (tt < a + 0.6)
            ext = tt[m0][np.argmax(ps[m0]) if name == "up" else np.argmin(ps[m0])]
            m = (tt >= ext) & (tt < b)
            x, y = tt[m] - ext, ps[m]
            f = lambda t, A, k, c: A * np.exp(-t / k) + c
            try:
                p, _ = curve_fit(f, x, y, p0=(y[0] - y[-1], tau, y[-1]),
                                 bounds=([-1e4, 0.005, 0], [1e4, 5, 1e4]), maxfev=20000)
                res[name + "_tau"] = float(p[1])
            except Exception:
                res[name + "_tau"] = float("nan")
            res[name + "_t_extremum"] = float(ext - a)
            mm = (tt > a) & (tt < b)
            xv = th[mm]; tv = tt[mm] - a
            x0, x1 = xv[0], xv[-20:].mean()
            lvl = x0 + 0.632 * (x1 - x0)
            k = np.flatnonzero((xv >= lvl) if name == "up" else (xv <= lvl))
            res[name + "_theta_t63"] = float(tv[k[0]]) if k.size else float("nan")
        res["rate_low_min"] = float(ps[(tt > 0.1) & (tt < 1)].min())
        res["rate_low_steady"] = float(ps[(tt > 3) & (tt < 4)].mean())
        res["rate_high_peak"] = float(ps[(tt > 4) & (tt < 4.3)].max())
        res["rate_high_steady"] = float(ps[(tt > 7) & (tt < 8)].mean())
        w = d[f"AS_{tau}_wsnr"]
        res["snr_low_steady"] = float(w[10:20].mean()); res["snr_high_steady"] = float(w[30:40].mean())
        out[tau] = res
    w = d["F_0.3_wsnr"]; ps = d["F_0.3_psth"]
    out["F"] = dict(snr_low_steady=float(w[10:20].mean()), snr_high_steady=float(w[30:40].mean()),
                    rate_low_steady=float(ps[60:80].mean()), rate_high_steady=float(ps[140:160].mean()))
    return out


# ============================================================================ Figure 7 flash detection
def fig7():
    d = load("e7.npz")
    amps = d["amps"]
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.7, 1, 1])
    # rasters: rows = (F, AS) x background, flash 0.05 and 0.3
    col = 0
    letters = iter("ABCDEFGHIJ")
    for bg in (0.04, 0.3):
        for a_sel in (0.05, 0.3):
            sub = gs[0, col].subgridspec(2, 1, hspace=0.05)
            for r_, key in enumerate(("F", "AS")):
                ax = fig.add_subplot(sub[r_])
                ras = binr(d[f"{key}_{bg}_ras_{a_sel}"][:30], 10)
                raster(ax, ras, 1e-3, t0=-0.1, on_col=COL[key], lw=0.6, gap=0.08)
                ax.axvspan(0, 0.03, color="#f2d95c", alpha=0.35, lw=0, zorder=0)
                ax.set_xlim(-0.1, 0.2); ax.set_yticks([]); ax.set_xticks([-0.1, 0, 0.1])
                if col == 0:
                    ax.set_ylabel({"F": "fixed", "AS": "adaptive"}[key], fontsize=6.4)
                if r_ == 0:
                    title(ax, next(letters), f"bg {bg:g}, flash {a_sel:g}")
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("time from flash (s)")
            col += 1
    ax = fig.add_subplot(gs[1, 0:2])
    for bg, ls in ((0.04, "-"), (0.3, "--")):
        for key in ("F", "AS", "AI_fixed"):
            ax.plot(amps, d[f"{key}_{bg}_auc"], color=COL[key], ls=ls, marker="o", ms=1.8)
    ax.axhline(0.75, color=COL["ink2"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_xlabel("flash amplitude (contrast)")
    ax.set_ylabel("ROC area"); ax.set_ylim(0.45, 1.02); ygrid(ax)
    title(ax, next(letters), "Detection (solid: bg 0.04; dashed: bg 0.30)")
    ax = fig.add_subplot(gs[1, 2:4])
    for bg, ls in ((0.04, "-"), (0.3, "--")):
        dv = d[f"diff_AS_{bg}"]; lo = d[f"diff_AS_{bg}_lo"]; hi = d[f"diff_AS_{bg}_hi"]
        ax.plot(amps, dv, color=COL["AS"], ls=ls, marker="o", ms=1.8)
        ax.fill_between(amps, lo, hi, color=COL["AS"], alpha=0.13, lw=0)
    ax.axhspan(-0.05, 0.05, color=COL["grid"], alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=COL["ink2"], lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("flash amplitude (contrast)")
    ax.set_ylabel("ΔROC area (adaptive − fixed)"); ygrid(ax)
    title(ax, next(letters), "Paired ΔROC, 95% CI (grey: ±0.05)")
    ax = fig.add_subplot(gs[2, 0:2])
    de = d["delays"]
    ax.plot(de, d["ps_F_auc"], color=COL["F"], marker="o", ms=2)
    for k, tau in (("AS_0.1", 0.1), ("AS_0.3", 0.3), ("AS_1.0", 1.0)):
        ax.plot(de, d[f"ps_{k}_auc"], color=TAU_SHADES[tau], marker="o", ms=2)
    ax.set_xscale("log"); ax.set_xlabel("flash delay after 0.30 → 0.04 switch (s)")
    ax.set_ylabel("ROC area"); ygrid(ax)
    title(ax, next(letters), "Flash 0.05 after a contrast decrease")
    ax = fig.add_subplot(gs[2, 2:4])
    for k, tau in (("AS_0.1", 0.1), ("AS_0.3", 0.3), ("AS_1.0", 1.0)):
        ax.plot(de, d[f"ps_diff_{k}"], color=TAU_SHADES[tau], marker="o", ms=2)
        ax.fill_between(de, d[f"ps_diff_{k}_lo"], d[f"ps_diff_{k}_hi"], color=TAU_SHADES[tau], alpha=0.12, lw=0)
    ax.axhspan(-0.05, 0.05, color=COL["grid"], alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=COL["ink2"], lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("flash delay after 0.30 → 0.04 switch (s)")
    ax.set_ylabel("ΔROC area (adaptive − fixed)"); ygrid(ax)
    title(ax, next(letters), "Paired difference, 95% CI")
    handles = [Line2D([], [], color=COL["F"], lw=1.3, label="fixed threshold"),
               Line2D([], [], color=COL["AS"], lw=1.3, label="spike-history ($\\tau_a$ 0.3 s)"),
               Line2D([], [], color=COL["AI_fixed"], lw=1.3, label="input-driven gain"),
               Line2D([], [], color=TAU_SHADES[0.1], lw=1.3, label="spike-history $\\tau_a$ 0.1 s"),
               Line2D([], [], color=TAU_SHADES[1.0], lw=1.3, label="spike-history $\\tau_a$ 1 s"),
               Line2D([], [], color="#f2d95c", lw=5, alpha=0.5, label="flash (30 ms)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "Fig7_flash_detection")


# ============================================================================ Figure 8 map
def map_stats():
    d = load("e8.npz")
    Rr = d["R"]
    F_lo, F_hi, F_all, AS_lo, AS_hi, AS_all = (Rr[..., k] for k in (0, 1, 2, 4, 5, 6))
    epoch = (AS_lo + AS_hi) / 2 - (F_lo + F_hi) / 2
    overall = AS_all - F_all
    from scipy.stats import t as tdist
    n = Rr.shape[2]
    tc = tdist.ppf(0.975, n - 1)
    out = {}
    for nm, D in (("epoch", epoch), ("overall", overall), ("lo", AS_lo - F_lo), ("hi", AS_hi - F_hi)):
        mu = D.mean(2); se = D.std(2, ddof=1) / np.sqrt(n)
        tstat = mu / np.maximum(se, 1e-12)
        p = 2 * tdist.sf(np.abs(tstat), n - 1)
        out[nm] = dict(mu=mu, lo=mu - tc * se, hi=mu + tc * se, p=p, q=bh(p))
    return d, out


def bh(p):
    """Benjamini-Hochberg adjusted p-values (same shape)."""
    flat = p.ravel(); m = flat.size
    o = np.argsort(flat)
    adj = flat[o] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out.reshape(p.shape)


def fig8():
    d, st = map_stats()
    ratios, periods = d["ratios"], d["periods"]
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 2)
    cmap = LinearSegmentedColormap.from_list("div", ["#d4521f", "#f2b48f", "#f2f1ee", "#8db8ea", "#1a5fb4"])
    lim = max(np.abs(st["epoch"]["mu"]).max(), np.abs(st["overall"]["mu"]).max())
    for j, (nm, lab) in enumerate((("epoch", "Mean of epoch SNRs (equal weight)"),
                                   ("overall", "Overall SNR (power weighted)"))):
        ax = fig.add_subplot(gs[0, j])
        mu = st[nm]["mu"]
        im = ax.pcolormesh(np.arange(periods.size + 1), np.arange(ratios.size + 1), mu, cmap=cmap,
                           norm=TwoSlopeNorm(0, -lim, lim), shading="flat", edgecolors="white", lw=0.6)
        sig = st[nm]["q"] < 0.05
        for i in range(ratios.size):
            for k in range(periods.size):
                if sig[i, k]:
                    ax.plot(k + 0.5, i + 0.5, marker="o", ms=1.6, color=COL["ink"], lw=0)
        ax.set_xticks(np.arange(periods.size)[::2] + 0.5)
        ax.set_xticklabels([f"{p:.2g}" for p in periods[::2]])
        ax.set_yticks(np.arange(ratios.size)[::2] + 0.5)
        ax.set_yticklabels([f"{r:.2g}" for r in ratios[::2]])
        ax.set_xlabel("contrast switching period (s)")
        ax.set_ylabel("high / low contrast ratio")
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
        cb.set_label("ΔSNR, adaptive − fixed (dB)", fontsize=6.2); cb.outline.set_visible(False)
        title(ax, "AB"[j], lab)
    Rr = d["R"]
    for j, (ep, lab) in enumerate(((0, "Low-contrast epochs"), (1, "High-contrast epochs"))):
        ax = fig.add_subplot(gs[1, j])
        for k, ls in ((periods.size - 1, "-"), (0, ":")):
            ax.plot(ratios, Rr[:, k, :, ep].mean(1), color=COL["F"], ls=ls, marker="o", ms=1.8)
            ax.plot(ratios, Rr[:, k, :, 4 + ep].mean(1), color=COL["AS"], ls=ls, marker="o", ms=1.8)
        ax.set_xscale("log"); ax.set_xlabel("high / low contrast ratio")
        ax.set_ylabel("SNR within epoch (dB)"); ygrid(ax)
        title(ax, "CD"[j], lab)
    handles = [Line2D([], [], color=COL["F"], lw=1.3, label="fixed, period 20 s"),
               Line2D([], [], color=COL["AS"], lw=1.3, label="spike-history, period 20 s"),
               Line2D([], [], color=COL["F"], ls=":", lw=1.3, label="fixed, period 0.1 s"),
               Line2D([], [], color=COL["AS"], ls=":", lw=1.3, label="spike-history, period 0.1 s"),
               Line2D([], [], color=COL["ink"], marker="o", ls="", ms=2, label="dot: paired difference significant after FDR correction (q < 0.05, 5 seeds)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "Fig8_helps_hurts_map")


# ============================================================================ Figure 9 robustness
def fig9():
    d = load("e9.npz")
    fig = plt.figure(figsize=(W2 * 0.75, 62 * MM))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    pf = d["pf"]
    for key in ("F", "AS"):
        for leak, ls in ((0.2, "-"), (0.0, ":")):
            v = d[f"rob_{key}_leak{leak}"]
            ax.plot(pf, v.mean(1), color=COL[key], ls=ls, marker="o", ms=1.8)
    ax.set_xlabel("synaptic failure probability"); ax.set_ylabel("test SNR (dB)"); ygrid(ax)
    title(ax, "A", "Spike failures (dotted: no leak)")
    ax = fig.add_subplot(gs[0, 1])
    nz = d["nz"]
    for key in ("U", "F", "AS"):
        ax.plot(nz, d["noise_" + key].mean(1), color=COL[key], marker="o", ms=1.8)
    ax.plot(nz, d["noise_ceil"], color=COL["muted"], ls=":")
    ax.set_xlabel("input noise s.d. (contrast)"); ax.set_ylabel("test SNR (dB)"); ygrid(ax)
    title(ax, "B", "Input noise")
    handles = [Line2D([], [], color=COL[k], lw=1.3, label=LAB[k]) for k in ("U", "F", "AS")] + \
              [Line2D([], [], color=COL["muted"], ls=":", lw=1, label="SNR of noisy input")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    save(fig, "Fig9_robustness")


# ============================================================================ Figure 10 controls
def fit_tau(ps, a, b, guess, kind):
    from scipy.optimize import curve_fit
    tt = (np.arange(ps.size) + 0.5) * 0.05
    m0 = (tt > a) & (tt < a + 0.6)
    ext = tt[m0][np.argmax(ps[m0]) if kind == "up" else np.argmin(ps[m0])]
    m = (tt >= ext) & (tt < b)
    x, y = tt[m] - ext, ps[m]
    f = lambda t, A, k, c: A * np.exp(-t / k) + c
    try:
        p, _ = curve_fit(f, x, y, p0=(y[0] - y[-1], guess, y[-1]), bounds=([-1e4, 0.005, 0], [1e4, 5, 1e4]),
                         maxfev=20000)
        return float(p[1])
    except Exception:
        return float("nan")


def controls_stats():
    d = load("e10.npz")
    s = d["sigmas"]
    out = {}
    out["sync"] = d["sync"]
    out["steps_up"] = {h: fit_tau(d[f"step_as_0.04_{h}_AS"], 4, 8, 0.3, "up") for h in (0.08, 0.16, 0.32, 0.64)}
    out["steps_down"] = {h: fit_tau(d[f"step_as_0.04_{h}_AS"], 0, 4, 0.3, "down") for h in (0.08, 0.16, 0.32, 0.64)}
    out["low_down"] = {}
    out["low_rate"] = {}
    for l in (0.01, 0.02, 0.04, 0.08):
        key = f"step_as_{l}_0.32_AS" if l != 0.04 else "step_as_0.04_0.32_AS"
        ps = d[key]; tt = (np.arange(ps.size) + 0.5) * 0.05
        out["low_down"][l] = fit_tau(ps, 0, 4, 0.3, "down")
        out["low_rate"][l] = float(ps[(tt > 3) & (tt < 4)].mean())
    out["leak_dr"] = {}
    for a in ("shared", "split"):
        for e in ("F", "AS"):
            for l in d["leaks"]:
                mu = d[f"leak_{a}_{e}_{l}"].mean(1)
                lo, hi, ol, oh = dyn_range(s, mu)
                out["leak_dr"][(a, e, float(l))] = (np.log10(hi / lo) if np.isfinite(lo) else 0.0, oh, mu.max())
    out["band_dr"] = {}
    for tc in (0.01, 0.03, 0.1):
        for e in ("F", "AS", "AM", "AI"):
            mu = d[f"band_{tc}_{e}"].mean(1)
            lo, hi, ol, oh = dyn_range(s, mu)
            out["band_dr"][(tc, e)] = (np.log10(hi / lo) if np.isfinite(lo) else 0.0, oh, lo, hi)
    out["bandmap"] = {}
    for tc in (0.01, 0.03, 0.1):
        B = d[f"bandmap_{tc}"]           # periods x seeds x 6 (F_lo, F_hi, F_all, AS_lo, AS_hi, AS_all)
        ep = (B[..., 3] + B[..., 4]) / 2 - (B[..., 0] + B[..., 1]) / 2
        ov = B[..., 5] - B[..., 2]
        out["bandmap"][tc] = (ep.mean(1), ov.mean(1), ep.std(1, ddof=1) / np.sqrt(ep.shape[1]),
                              ov.std(1, ddof=1) / np.sqrt(ov.shape[1]))
    return d, out


def fig10():
    d, st = controls_stats()
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 3)
    # A sync error
    ax = fig.add_subplot(gs[0, 0])
    sy = d["sync"]
    for ei, key in enumerate(("F", "AS", "AM")):
        m = sy[:, 0] == ei
        ax.plot(sy[m, 1] * 1e3, sy[m, 2] * 100, color=COL[key], marker="o", ms=2)
    for ei, key in enumerate(("AS", "AM")):
        m = sy[:, 0] == ei + 1
        ax.plot(sy[m, 1] * 1e3, sy[m, 3] * 100, color=COL[key], marker="s", ms=2, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("timing resolution $\\Delta q$ (ms)"); ax.set_ylabel("error (%)"); ygrid(ax)
    title(ax, "A", "Quantization error")
    ax.text(0.03, 0.97, "solid: smoothed reference error / RMS(x)\ndotted: median threshold error",
            transform=ax.transAxes, va="top", fontsize=5.6, color=COL["ink2"])
    # B step size
    ax = fig.add_subplot(gs[0, 1])
    hs = [0.08, 0.16, 0.32, 0.64]
    hu = hs[1:]   # the 0.04 -> 0.08 step is too small for a reliable fit
    ax.plot(hu, [st["steps_up"][h] for h in hu], color=COL["AS"], marker="o", ms=2.5)
    ax.plot(hs, [st["steps_down"][h] for h in hs], color=COL["AS"], marker="s", ms=2.5, ls="--")
    ax.axhline(0.3, color=COL["ink2"], lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([f"{h:g}" for h in hs]); ax.minorticks_off()
    ax.set_xlabel("high contrast (low = 0.04)"); ax.set_ylabel("rate time constant (s)")
    ax.set_ylim(0, 0.5); ygrid(ax)
    title(ax, "B", "Step size (solid up, dashed down)")
    # C low contrast / firing
    ax = fig.add_subplot(gs[0, 2])
    ls_ = [0.01, 0.02, 0.04, 0.08]
    ax.plot([st["low_rate"][l] for l in ls_], [st["low_down"][l] for l in ls_], color=COL["AS"], marker="s", ms=2.5, ls="--")
    for l in ls_:
        ax.annotate(f"{l:g}", (st["low_rate"][l], st["low_down"][l]), textcoords="offset points", xytext=(-4, 5),
                    fontsize=5.6, color=COL["ink2"])
    ax.axhline(0.3, color=COL["ink2"], lw=0.6, ls=":")
    ax.set_xlabel("steady low-contrast rate (spikes/s)"); ax.set_ylabel("recovery time constant (s)")
    ax.set_ylim(0, 0.5); ygrid(ax)
    title(ax, "C", "Recovery vs low-contrast firing")
    # D leak x architecture
    ax = fig.add_subplot(gs[1, 0])
    leaks = d["leaks"]
    for a, ls in (("shared", "-"), ("split", "--")):
        for e in ("F", "AS"):
            ax.plot(leaks * 1e3, [st["leak_dr"][(a, e, float(l))][0] for l in leaks], color=COL[e], ls=ls,
                    marker="o", ms=2)
    ax.set_xscale("log"); ax.set_xticks(leaks * 1e3); ax.set_xticklabels([f"{l * 1e3:g}" for l in leaks])
    ax.minorticks_off(); ax.set_xlabel("reference leak $\\tau_{leak}$ (ms)")
    ax.set_ylabel("range with SNR ≥ 10 dB (decades)"); ygrid(ax)
    title(ax, "D", "Leak and architecture")
    # E bandwidth dynamic range
    ax = fig.add_subplot(gs[1, 1])
    tcs = [0.01, 0.03, 0.1]
    xs = np.arange(len(tcs))
    for k, e in enumerate(("F", "AS", "AM", "AI")):
        ck = "AI_gain" if e == "AI" else e
        v = [st["band_dr"][(tc, e)][0] for tc in tcs]
        ax.bar(xs + (k - 1.5) * 0.19, v, width=0.17, color=COL[ck], edgecolor="white", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"{tc * 1e3:g} ms" for tc in tcs])
    ax.set_xlabel("stimulus correlation time $\\tau_c$"); ax.set_ylabel("range (decades, lower bound)")
    ygrid(ax)
    title(ax, "E", "Stimulus bandwidth")
    # F bandwidth map row
    ax = fig.add_subplot(gs[1, 2])
    per = d["periods"]
    for tc, c in ((0.01, "#eda100"), (0.03, COL["AS"]), (0.1, "#123f78")):
        ep, ov, se_ep, se_ov = st["bandmap"][tc]
        ax.plot(per, ep, color=c, marker="o", ms=2)
        ax.plot(per, ov, color=c, marker="s", ms=2, ls="--")
    ax.axhline(0, color=COL["ink2"], lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("switching period (s)")
    ax.set_ylabel("ΔSNR adaptive − fixed (dB)"); ygrid(ax)
    title(ax, "F", "Map row, ratio 13.5")
    handles = [Line2D([], [], color=COL["F"], lw=1.3, label=LAB["F"]),
               Line2D([], [], color=COL["AS"], lw=1.3, label=LAB["AS"]),
               Line2D([], [], color=COL["AM"], lw=1.3, label=LAB["AM"]),
               Line2D([], [], color=COL["AI_gain"], lw=1.3, label=LAB["AI_gain"]),
               Line2D([], [], color="#eda100", lw=1.3, label="panel F: $\\tau_c$ = 10 ms"),
               Line2D([], [], color=COL["AS"], lw=1.3, label="panel F: $\\tau_c$ = 30 ms"),
               Line2D([], [], color="#123f78", lw=1.3, label="panel F: $\\tau_c$ = 100 ms"),
               Line2D([], [], color=COL["ink2"], ls="--", lw=1.3, label="dashed: separate ON/OFF (D), power-weighted (F)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "FigS2_controls")


ALL = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7,
           fig8=fig8, fig9=fig9, figS1=figS1, figS2=fig10)
if __name__ == "__main__":
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
