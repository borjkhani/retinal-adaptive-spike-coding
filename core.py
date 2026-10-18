"""
Core model: retinal-inspired analog-to-spike converters (single receptive field).

Stimulus contrast c(t) -> biphasic temporal filter -> x(t) (the analog signal to be
transmitted) -> encoder -> ON/OFF spike trains -> decoder -> x_hat(t).

Event encoders (shared architecture: one signed reference, ON and OFF events)
  F   fixed threshold
  AS  additive spike-history threshold   theta = theta0 * (1 + r/r_half)
  AM  multiplicative spike-history threshold (Bohte 2012 style):
      theta = theta0 + a ;  a decays with tau_a ;  at a spike a += m * theta
  AI  input-driven divisive gain         theta = theta0 * s_hat / s0

Split architecture (separate ON and OFF units): each unit encodes its rectified
input with its own leaky reference, refractory period and adaptation state.

Update order at every time step i (identical in encoder and decoder):
  1. decay the reference (leak) and the adaptation state;
  2. compute theta_i from the decayed state (before any spike at step i);
  3. if a spike is emitted/received: reference += +/- theta_i, then adaptation state
     is incremented (r += 1/tau_a for AS, a += m*theta_i for AM).

Spike times are transmitted at a finite resolution dq (default 1 ms): each spike is
reported at the next multiple of dq (causal). The decoder runs on these quantized
times, and the code-rate estimate uses the same resolution.
"""
import numpy as np
from numba import njit

DT = 1e-4  # s, 0.1 ms

MODE = dict(F=0, AS=0, AM=2, AI=1)


# ----------------------------------------------------------------------------- stimuli
def lowpass_noise(n, tau_c, rng, dt=DT, order=2):
    """Unit-variance Gaussian noise, cascade of `order` first-order low-pass filters."""
    z = rng.standard_normal(n + int(10 * tau_c / dt))
    a = np.exp(-dt / tau_c)
    for _ in range(order):
        z = _ar1(z, a)
    z = z[-n:]
    return (z - z.mean()) / z.std()


@njit(cache=True)
def _ar1(z, a):
    y = np.empty_like(z)
    acc = 0.0
    b = np.sqrt(1 - a * a)
    for i in range(z.size):
        acc = a * acc + b * z[i]
        y[i] = acc
    return y


def biphasic_kernel(dt=DT, t1=0.012, t2=0.030, w2=0.8, n=4, dur=0.25):
    """Difference of gamma-like kernels (ON-type, partially band-pass)."""
    t = np.arange(0, dur, dt)
    g1 = (t / t1) ** n * np.exp(-n * (t / t1 - 1))
    g2 = (t / t2) ** n * np.exp(-n * (t / t2 - 1))
    return g1 / g1.sum() - w2 * g2 / g2.sum()


_K = None


def front_end(c, dt=DT):
    """Filtered analog signal x = k * c (causal), FFT convolution."""
    global _K
    if _K is None:
        _K = biphasic_kernel(dt)
    from scipy.signal import fftconvolve
    return fftconvolve(c, _K)[: c.size]


def contrast_signal(n, sigma, tau_c, rng, dt=DT):
    """c(t) = sigma(t) * z(t); sigma may be scalar or array."""
    return sigma * lowpass_noise(n, tau_c, rng, dt)


def make_input(sigma, T, rng, tau_c=0.03, noise=0.01, tau_n=0.002):
    """Clean target x = k*c and encoder input x_in = k*(c + eta)."""
    n = int(round(T / DT))
    c = contrast_signal(n, sigma, tau_c, rng)
    eta = noise * lowpass_noise(n, tau_n, rng, order=1) if noise > 0 else 0.0
    return front_end(c), front_end(c + eta), c


# ----------------------------------------------------------------------------- shared-reference encoder
@njit(cache=True)
def encode_shared(x, dt, theta0, kappa, m, tau_a, tau_leak, t_ref, mode, tau_sig, sig0, s_init):
    """
    mode 0 : theta = theta0 + kappa * r            (F: kappa = 0; AS: kappa = theta0/r_half)
    mode 1 : theta = theta0 * s_hat / sig0         (AI)
    mode 2 : theta = theta0 + a, a += m*theta      (AM)
    s_init : initial value of the running RMS estimate (AI only)
    Returns spikes (+1/-1/0), theta trace.
    """
    n = x.size
    spk = np.zeros(n, np.int8)
    th = np.empty(n)
    xh = 0.0
    r = 0.0
    a = 0.0
    s2 = s_init * s_init
    da = np.exp(-dt / tau_a)
    dl = np.exp(-dt / tau_leak) if tau_leak > 0 else 1.0
    ds = np.exp(-dt / tau_sig)
    nref = int(round(t_ref / dt))
    last = -10 * nref - 1
    for i in range(n):
        xh *= dl
        r *= da
        a *= da
        s2 = ds * s2 + (1 - ds) * x[i] * x[i]
        if mode == 0:
            theta = theta0 + kappa * r
        elif mode == 1:
            theta = theta0 * np.sqrt(s2) / sig0
            if theta < 1e-6:
                theta = 1e-6
        else:
            theta = theta0 + a
        if i - last >= nref:
            e = x[i] - xh
            if e > theta:
                spk[i] = 1
            elif e < -theta:
                spk[i] = -1
            if spk[i] != 0:
                xh += spk[i] * theta
                r += 1.0 / tau_a
                a += m * theta
                last = i
        th[i] = theta
    return spk, th


@njit(cache=True)
def decode_shared(spk, dt, theta0, kappa, m, tau_a, tau_leak, mode, theta_trace):
    """
    mode 0 : recompute theta = theta0 + kappa*r from received spikes (F, AS)
    mode 2 : recompute theta = theta0 + a       from received spikes (AM)
    mode 1 : use the supplied theta_trace (gain-informed decoder for AI)
    mode 3 : constant theta0 (fixed-gain decoder for AI)
    """
    n = spk.size
    out = np.empty(n)
    xh = 0.0
    r = 0.0
    a = 0.0
    da = np.exp(-dt / tau_a)
    dl = np.exp(-dt / tau_leak) if tau_leak > 0 else 1.0
    for i in range(n):
        xh *= dl
        r *= da
        a *= da
        if mode == 0:
            theta = theta0 + kappa * r
        elif mode == 2:
            theta = theta0 + a
        elif mode == 1:
            theta = theta_trace[i]
        else:
            theta = theta0
        if spk[i] != 0:
            xh += spk[i] * theta
            r += 1.0 / tau_a
            a += m * theta
        out[i] = xh
    return out


# ----------------------------------------------------------------------------- split ON / OFF units
@njit(cache=True)
def encode_split(x, dt, theta0, kappa, tau_a, tau_leak, t_ref):
    """Separate ON and OFF units. Unit u encodes y_u = [+x]_+ (ON) or [-x]_+ (OFF) with its
    own reference (increments only, leaky), refractory period and adaptation rate r_u."""
    n = x.size
    spk = np.zeros(n, np.int8)
    th_on = np.empty(n)
    xh = np.zeros(2)
    r = np.zeros(2)
    last = np.array([-10 ** 9, -10 ** 9])
    da = np.exp(-dt / tau_a)
    dl = np.exp(-dt / tau_leak)
    nref = int(round(t_ref / dt))
    for i in range(n):
        for u in range(2):
            xh[u] *= dl
            r[u] *= da
            y = x[i] if u == 0 else -x[i]
            if y < 0:
                y = 0.0
            theta = theta0 + kappa * r[u]
            if i - last[u] >= nref and y - xh[u] > theta:
                xh[u] += theta
                r[u] += 1.0 / tau_a
                last[u] = i
                # if both fire in one step (cannot happen: rectified inputs are exclusive)
                spk[i] = 1 if u == 0 else -1
            if u == 0:
                th_on[i] = theta
    return spk, th_on


@njit(cache=True)
def decode_split(spk, dt, theta0, kappa, tau_a, tau_leak):
    n = spk.size
    out = np.empty(n)
    xh = np.zeros(2)
    r = np.zeros(2)
    da = np.exp(-dt / tau_a)
    dl = np.exp(-dt / tau_leak)
    for i in range(n):
        for u in range(2):
            xh[u] *= dl
            r[u] *= da
            theta = theta0 + kappa * r[u]
            if (u == 0 and spk[i] == 1) or (u == 1 and spk[i] == -1):
                xh[u] += theta
                r[u] += 1.0 / tau_a
        out[i] = xh[0] - xh[1]
    return out


# ----------------------------------------------------------------------------- channel
def quantize_times(spk, dq, dt=DT):
    """Report each spike at the next multiple of dq (causal). dq <= dt: unchanged."""
    if dq <= dt * 1.0001:
        return spk
    step = int(round(dq / dt))
    idx = np.flatnonzero(spk)
    q = ((idx + step - 1) // step) * step
    ok = q < spk.size
    out = np.zeros_like(spk)
    # collisions are impossible when dq <= t_ref for the shared encoder; for the split
    # encoder keep the first event of a tick (counted as a loss)
    _, first = np.unique(q[ok], return_index=True)
    out[q[ok][first]] = spk[idx[ok][first]]
    return out


def delete_spikes(spk, p_fail, rng):
    if p_fail <= 0:
        return spk
    s = spk.copy()
    idx = np.flatnonzero(s)
    s[idx[rng.random(idx.size) < p_fail]] = 0
    return s


def uniform_adc(x, dt, D, B, R):
    """Sample every D seconds, quantize to B bits in [-R, R] (mid-rise), zero-order hold."""
    step = int(round(D / dt))
    xs = x[::step]
    L = 2 ** B
    q = 2 * R / L
    codes = np.clip(np.floor((xs + R) / q), 0, L - 1).astype(np.int64)
    vals = -R + (codes + 0.5) * q
    return np.repeat(vals, step)[: x.size], codes


# ----------------------------------------------------------------------------- metrics
@njit(cache=True)
def causal_smooth(y, dt, tau):
    if tau <= 0:
        return y.copy()
    a = np.exp(-dt / tau)
    out = np.empty_like(y)
    acc = y[0]
    for i in range(y.size):
        acc = a * acc + (1 - a) * y[i]
        out[i] = acc
    return out


def best_lag(target, est, max_lag=0.03, dt=DT, burn=2.0):
    """Alignment lag (s) maximizing SNR; used only on calibration realizations."""
    b = int(burn / dt)
    t = target[b:]
    p = np.mean(t ** 2)
    best, bl = -np.inf, 0
    for L in range(0, int(max_lag / dt) + 1, 5):
        e = est[b + L:]
        s = p / max(np.mean((t[: e.size] - e) ** 2), 1e-30)
        if s > best:
            best, bl = s, L
    return bl * dt


def snr_at(target, est, lag, dt=DT, burn=2.0, mask=None):
    """SNR (dB) at a fixed lag. mask (bool, same length as target) restricts samples."""
    L = int(round(lag / dt))
    b = int(burn / dt)
    t = target[b: target.size - L]
    e = est[b + L:]
    if mask is not None:
        mk = mask[b: target.size - L]
        t, e = t[mk], e[mk]
    return 10 * np.log10(np.mean(t ** 2) / max(np.mean((t - e) ** 2), 1e-30))


def entropy_bits(sym):
    _, cnt = np.unique(sym, return_counts=True, axis=0)
    p = cnt / cnt.sum()
    return -(p * np.log2(p)).sum()


def event_code_rate(spk, T, dq=1e-3, dt=DT):
    """Per-event marginal entropy of (ISI in units of dq, polarity) x event rate (bits/s).
    `spk` should already be quantized at dq. Idealized cost; ignores serial dependence."""
    idx = np.flatnonzero(spk)
    if idx.size < 3:
        return 0.0, idx.size / T
    step = max(int(round(dq / dt)), 1)
    qi = np.diff(idx) // step
    pol = spk[idx[1:]].astype(np.int64)
    return entropy_bits(np.stack([qi, pol], 1)) * idx.size / T, idx.size / T


def adc_code_rate(codes, D):
    """Entropy of successive code differences per sample x sample rate (DPCM-style)."""
    return entropy_bits(np.diff(codes)[:, None]) / D, 1.0 / D


# ----------------------------------------------------------------------------- convenience
DEFAULTS = dict(t_ref=2e-3, tau_a=0.3, tau_leak=0.2, tau_leak_split=0.03, tau_sig=0.3,
                tau_s=3e-3, dq=1e-3, r_half=20.0, m=0.5)


def encode(x_in, enc, theta0, sig0=1.0, s_init=None, **kw):
    """enc in {'F','AS','AM','AI','F_split','AS_split'}; returns (spikes, theta, params)."""
    p = {**DEFAULTS, **kw}
    if enc.endswith("_split"):
        kappa = theta0 / p["r_half"] if enc.startswith("AS") else 0.0
        spk, th = encode_split(x_in, DT, theta0, kappa, p["tau_a"], p["tau_leak_split"], p["t_ref"])
        return spk, th, p
    kappa = theta0 / p["r_half"] if enc == "AS" else 0.0
    m = p["m"] if enc == "AM" else 0.0
    si = sig0 if s_init is None else s_init
    spk, th = encode_shared(x_in, DT, theta0, kappa, m, p["tau_a"], p["tau_leak"], p["t_ref"],
                            MODE[enc], p["tau_sig"], sig0, si)
    return spk, th, p


def decode(rx, enc, theta0, p, th=None, decoder="spikes"):
    """decoder: 'spikes' (recompute from spikes), 'gain' (AI with true theta),
    'fixed' (AI with constant theta0)."""
    if enc.endswith("_split"):
        kappa = theta0 / p["r_half"] if enc.startswith("AS") else 0.0
        return decode_split(rx, DT, theta0, kappa, p["tau_a"], p["tau_leak_split"])
    kappa = theta0 / p["r_half"] if enc == "AS" else 0.0
    m = p["m"] if enc == "AM" else 0.0
    if enc == "AI":
        dm = 1 if decoder == "gain" else 3
        # gain-informed decoder reads the encoder's theta at the (quantized) received spike time
        return decode_shared(rx, DT, theta0, 0.0, 0.0, p["tau_a"], p["tau_leak"], dm, th)
    return decode_shared(rx, DT, theta0, kappa, m, p["tau_a"], p["tau_leak"], MODE[enc], th)


def run_event(x_clean, x_in, enc, theta0, decoder="spikes", lag=None, p_fail=0.0, rng=None,
              sig0=1.0, s_init=None, burn=2.0, **kw):
    """Encode, transmit (time quantization + optional failures), decode, score.
    If lag is None it is chosen on this realization (use only for calibration runs)."""
    spk, th, p = encode(x_in, enc, theta0, sig0=sig0, s_init=s_init, **kw)
    tx = quantize_times(spk, p["dq"])
    rx = delete_spikes(tx, p_fail, rng or np.random.default_rng(0))
    rec = decode(rx, enc, theta0, p, th=th, decoder=decoder)
    est = causal_smooth(rec, DT, p["tau_s"])
    if lag is None:
        lag = best_lag(x_clean, est, burn=burn)
    T = x_clean.size * DT
    bits, rate = event_code_rate(tx, T, dq=p["dq"])
    return dict(snr=snr_at(x_clean, est, lag, burn=burn), lag=lag, bits=bits, rate=rate,
                spk=spk, rx=rx, th=th, est=est, rec=rec)


def run_adc(x_clean, x_in, D, B, R, tau_s=3e-3, lag=None, burn=2.0):
    rec, codes = uniform_adc(x_in, DT, D, B, R)
    est = causal_smooth(rec, DT, tau_s)
    if lag is None:
        lag = best_lag(x_clean, est, burn=burn)
    bits, fs = adc_code_rate(codes, D)
    return dict(snr=snr_at(x_clean, est, lag, burn=burn), lag=lag, bits=bits, rate=fs,
                est=est, rec=rec)


# ----------------------------------------------------------------------------- learned spike-history decoder
def rate_trace(rx, tau=0.3):
    """Exponentially filtered event rate (Hz) from received spikes (both polarities)."""
    s = (rx != 0).astype(float) / DT
    return causal_smooth(s, DT, tau)


class GainDecoder:
    """History-based decoder: x_hat = g(log(1 + r)) * s(t), where s is the staircase from a
    base decoder and r the received-spike rate; g is piecewise linear in log rate, fitted
    by least squares on training realizations that are independent of the test set."""

    def __init__(self, knots=np.linspace(0, np.log1p(500), 14), tau=0.3):
        self.knots, self.tau, self.w = knots, tau, None

    def _basis(self, r):
        z = np.clip(np.log1p(r), self.knots[0], self.knots[-1])
        Bm = np.zeros((z.size, self.knots.size))
        j = np.clip(np.searchsorted(self.knots, z) - 1, 0, self.knots.size - 2)
        f = (z - self.knots[j]) / (self.knots[j + 1] - self.knots[j])
        Bm[np.arange(z.size), j] = 1 - f
        Bm[np.arange(z.size), j + 1] = f
        return Bm

    def features(self, rx, staircase):
        return self._basis(rate_trace(rx, self.tau)) * staircase[:, None]

    def fit(self, pairs, sub=10):
        """pairs: list of (rx, staircase_smoothed, target_aligned)."""
        A = np.concatenate([self.features(rx, s)[::sub] for rx, s, _ in pairs])
        y = np.concatenate([t[::sub] for _, _, t in pairs])
        lam = 1e-6 * np.trace(A.T @ A) / A.shape[1]
        self.w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, rx, staircase):
        return self.features(rx, staircase) @ self.w


@njit(cache=True)
def _leaky_steps(idx, W, n, dt, tau_leak, tau_s):
    """For each column j: leaky integration of steps W[k, j] at sample idx[k], then the
    causal exponential smoothing used by all decoders. Returns (n, ncols)."""
    nc = W.shape[1]
    out = np.zeros((n, nc))
    dl = np.exp(-dt / tau_leak)
    a = np.exp(-dt / tau_s)
    xh = np.zeros(nc)
    sm = np.zeros(nc)
    k = 0
    for i in range(n):
        for j in range(nc):
            xh[j] *= dl
        while k < idx.size and idx[k] == i:
            for j in range(nc):
                xh[j] += W[k, j]
            k += 1
        for j in range(nc):
            sm[j] = a * sm[j] + (1 - a) * xh[j]
            out[i, j] = sm[j]
    return out


class RichDecoder:
    """Spike-history decoder with a learned per-spike step size. Each received spike k adds
    pol_k * theta0 * g(z_k) to a leaky reference (same leak as the encoder), where z_k are
    causal spike-history features evaluated just before the spike: log filtered event rate at
    30 ms, 100 ms, 300 ms and 1 s, log of the preceding inter-spike interval, and the recent
    ON/OFF balance. g is a sum of piecewise-linear functions of the features, so the output is
    linear in the weights and is fitted by regularized least squares on training data."""

    TAUS = (0.03, 0.1, 0.3, 1.0)

    def __init__(self, theta0=1.0, tau_leak=0.2, tau_s=3e-3, nk=8):
        self.theta0, self.tau_leak, self.tau_s, self.nk, self.w = theta0, tau_leak, tau_s, nk, None

    @staticmethod
    def _pl_basis(z, knots):
        z = np.clip(z, knots[0], knots[-1])
        Bm = np.zeros((z.size, knots.size))
        j = np.clip(np.searchsorted(knots, z) - 1, 0, knots.size - 2)
        f = (z - knots[j]) / (knots[j + 1] - knots[j])
        Bm[np.arange(z.size), j] = 1 - f
        Bm[np.arange(z.size), j + 1] = f
        return Bm

    def _spike_features(self, rx):
        idx = np.flatnonzero(rx)
        ev = (rx != 0).astype(float) / DT
        cols = []
        for tau in self.TAUS:
            r = causal_smooth(ev, DT, tau)
            zb = np.log1p(r[np.maximum(idx - 1, 0)])          # just before the spike
            cols.append(self._pl_basis(zb, np.linspace(0, np.log1p(600), self.nk)))
        isi = np.r_[1.0, np.diff(idx) * DT] if idx.size else np.zeros(0)
        cols.append(self._pl_basis(np.log10(np.clip(isi, 1e-4, 1)), np.linspace(-3, 0, self.nk)))
        pol = causal_smooth(rx.astype(float) / DT, DT, 0.3)
        bal = pol[np.maximum(idx - 1, 0)] / (causal_smooth(ev, DT, 0.3)[np.maximum(idx - 1, 0)] + 1.0)
        cols.append(self._pl_basis(bal, np.linspace(-1, 1, self.nk)))
        return idx, np.concatenate(cols, axis=1)

    def features(self, rx):
        idx, Bk = self._spike_features(rx)
        W = Bk * (rx[idx].astype(float) * self.theta0)[:, None]
        return _leaky_steps(idx.astype(np.int64), W, rx.size, DT, self.tau_leak, self.tau_s)

    def fit(self, pairs, sub=10):
        """pairs: list of (rx, target_aligned_full_length, start_index)."""
        A = np.concatenate([self.features(rx)[b::sub][: t.size // sub] for rx, t, b in pairs])
        y = np.concatenate([t[::sub][: t.size // sub] for _, t, _ in pairs])
        lam = 1e-6 * np.trace(A.T @ A) / A.shape[1]
        self.w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, rx):
        return self.features(rx) @ self.w
