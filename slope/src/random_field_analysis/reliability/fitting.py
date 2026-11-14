from __future__ import annotations

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import gennorm, kstest

MIN_SAMPLES_PER_SLOPE = 30
N_YHAT_BINS = 4
MIN_BIN_SAMPLES = 20
MIN_ANCHOR_SAMPLES = 35
FIX_LOC = None
EPS_PDF = 1e-300
EPS_CDF = 1e-12
BETA_MIN = 0.3
BETA_MAX = 8.0
SCALE_FLOOR_RATIO = 0.05
SCALE_CAP_RATIO = 5.0
SHRINK_COUNT = 40.0
BETA_SHRINK_COUNT = 200.0
LOC_BIAS_Z = 0.0
SIGMA_MONOTONE = True
SIGMA_INFLATION = 1.12
SPLINE_SMOOTH_FACTOR = 0.8

def clean_1d(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]

def fit_single_ged(x: np.ndarray):
    x = clean_1d(x)
    if x.size < 8:
        raise ValueError('Not enough data to fit GED.')
    if FIX_LOC is None:
        beta, loc, scale = gennorm.fit(x)
    else:
        beta, loc, scale = gennorm.fit(x, floc=float(FIX_LOC))
    return (float(beta), float(loc), float(scale))

def quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError('y_hat has too many duplicates to form bins.')
    return edges

def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)

def _pointwise_pdf(err: np.ndarray, beta, loc, scale) -> np.ndarray:
    e = np.asarray(err, dtype=float).reshape(-1)
    if np.ndim(beta) == 0 and np.ndim(loc) == 0 and (np.ndim(scale) == 0):
        return gennorm.pdf(e, float(beta), loc=float(loc), scale=float(scale))
    b = np.asarray(beta, dtype=float).reshape(-1)
    m = np.asarray(loc, dtype=float).reshape(-1)
    s = np.asarray(scale, dtype=float).reshape(-1)
    n = min(e.size, b.size, m.size, s.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        bi = float(np.clip(b[i], BETA_MIN, BETA_MAX))
        si = float(max(s[i], 1e-12))
        out[i] = gennorm.pdf(float(e[i]), bi, loc=float(m[i]), scale=si)
    return out

def _pointwise_cdf(err: np.ndarray, beta, loc, scale) -> np.ndarray:
    e = np.asarray(err, dtype=float).reshape(-1)
    if np.ndim(beta) == 0 and np.ndim(loc) == 0 and (np.ndim(scale) == 0):
        return gennorm.cdf(e, float(beta), loc=float(loc), scale=float(scale))
    b = np.asarray(beta, dtype=float).reshape(-1)
    m = np.asarray(loc, dtype=float).reshape(-1)
    s = np.asarray(scale, dtype=float).reshape(-1)
    n = min(e.size, b.size, m.size, s.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        bi = float(np.clip(b[i], BETA_MIN, BETA_MAX))
        si = float(max(s[i], 1e-12))
        out[i] = gennorm.cdf(float(e[i]), bi, loc=float(m[i]), scale=si)
    return out

def ged_nll(err: np.ndarray, beta, loc, scale) -> np.ndarray:
    pdf = _pointwise_pdf(err, beta, loc, scale)
    return -np.log(np.clip(pdf, EPS_PDF, None))

def conditional_cdf(err: np.ndarray, betas: np.ndarray, locs: np.ndarray, scales: np.ndarray):
    u = _pointwise_cdf(err, betas, locs, scales)
    return np.clip(u.astype(float), EPS_CDF, 1.0 - EPS_CDF)

def _weighted_unique_points(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n = min(x.size, y.size, w.size)
    x = x[:n]
    y = y[:n]
    w = w[:n]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if x.size == 0:
        raise ValueError('No valid points for spline fitting.')
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.zeros_like(ux, dtype=float)
    uw = np.zeros_like(ux, dtype=float)
    for i in range(ux.size):
        m = inv == i
        wi = w[m]
        yi = y[m]
        sw = float(np.sum(wi))
        if sw <= 0:
            sw = float(np.sum(np.ones_like(wi)))
            wi = np.ones_like(wi, dtype=float)
        uy[i] = float(np.sum(wi * yi) / sw)
        uw[i] = sw
    return (ux, uy, uw)

def fit_spline_tck(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, k: int | None=None, s: float | None=None):
    x, y, w = _weighted_unique_points(x, y, w)
    if x.size == 1:
        eps = max(1e-06, abs(float(x[0])) * 1e-06)
        x = np.asarray([x[0] - eps, x[0] + eps], dtype=float)
        y = np.asarray([y[0], y[0]], dtype=float)
        w = np.asarray([1.0, 1.0], dtype=float)
    if k is None:
        k = int(min(3, max(1, x.size - 1)))
    else:
        k = int(max(1, min(int(k), int(max(1, x.size - 1)))))
    if s is None:
        y_var = float(np.var(y))
        s = float(SPLINE_SMOOTH_FACTOR * max(y_var, 1e-06) * x.size)
    t, c, k = splrep(x, y, w=np.sqrt(np.clip(w, 1.0, None)), k=k, s=s)
    return {'t': np.asarray(t, dtype=float), 'c': np.asarray(c, dtype=float), 'k': int(k)}

def eval_spline_tck(tck: dict, x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    yy = splev(xx, (tck['t'], tck['c'], int(tck['k'])), ext=3)
    return np.asarray(yy, dtype=float)

def isotonic_regression(y: np.ndarray, w: np.ndarray, *, increasing: bool=True) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n = int(min(y.size, w.size))
    if n <= 1:
        return y[:n].copy()
    yy = y[:n].copy()
    ww = np.clip(w[:n].copy(), 1e-12, None)
    if not increasing:
        yy = -yy
    vals = []
    wts = []
    lens = []
    for i in range(n):
        vals.append(float(yy[i]))
        wts.append(float(ww[i]))
        lens.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v1, w1, l1 = (vals[-2], wts[-2], lens[-2])
            v2, w2, l2 = (vals[-1], wts[-1], lens[-1])
            new_w = w1 + w2
            if new_w <= 0:
                new_w = 1.0
            new_v = (v1 * w1 + v2 * w2) / new_w
            vals[-2] = float(new_v)
            wts[-2] = float(new_w)
            lens[-2] = int(l1 + l2)
            vals.pop()
            wts.pop()
            lens.pop()
    fitted = np.empty(n, dtype=float)
    pos = 0
    for v, l in zip(vals, lens):
        fitted[pos:pos + l] = float(v)
        pos += l
    if not increasing:
        fitted = -fitted
    return fitted

def fit_conditional_ged(err: np.ndarray, yhat: np.ndarray):
    err = clean_1d(err)
    yhat = clean_1d(yhat)
    n = min(len(err), len(yhat))
    err = err[:n]
    yhat = yhat[:n]
    if err.size < MIN_SAMPLES_PER_SLOPE:
        raise ValueError('Not enough samples for conditional fit.')
    global_beta_mle, global_loc_mle, global_scale_mle = fit_single_ged(err)
    n_all = int(err.size)
    mean_e = float(np.mean(err))
    median_e = float(np.median(err))
    std_e = float(np.std(err, ddof=1)) if n_all > 1 else float(np.std(err))
    se_mean = float(std_e / np.sqrt(max(n_all, 1)))
    loc_margin = float(LOC_BIAS_Z * se_mean) if mean_e > 0.0 and median_e > 0.0 else 0.0
    if FIX_LOC is None:
        global_loc = float(global_loc_mle + loc_margin)
    else:
        global_loc = float(FIX_LOC)
    try:
        global_beta, _, global_scale = gennorm.fit(err, floc=global_loc)
    except Exception:
        global_beta = float(global_beta_mle)
        global_scale = float(global_scale_mle)
    global_beta = float(np.clip(global_beta, BETA_MIN, BETA_MAX))
    global_scale = float(max(float(global_scale), 1e-12))
    n_bins_target = int(min(N_YHAT_BINS, max(2, err.size // MIN_ANCHOR_SAMPLES)))
    edges = quantile_edges(yhat, n_bins_target)
    idx = assign_bins(yhat, edges)
    n_eff_bins = len(edges) - 1
    anchor_x = np.full(n_eff_bins, np.nan, dtype=float)
    locs = np.full(n_eff_bins, global_loc, dtype=float)
    scales = np.full(n_eff_bins, global_scale, dtype=float)
    counts = np.zeros(n_eff_bins, dtype=int)
    scale_floor = float(max(global_scale * SCALE_FLOOR_RATIO, 1e-08))
    scale_cap = float(max(global_scale * SCALE_CAP_RATIO, scale_floor * 1.05))
    global_log_scale = float(np.log(global_scale))
    for j in range(n_eff_bins):
        in_bin = idx == j
        ej = err[in_bin]
        yj = yhat[in_bin]
        counts[j] = int(ej.size)
        if yj.size > 0:
            anchor_x[j] = float(np.median(yj))
        else:
            anchor_x[j] = float(0.5 * (edges[j] + edges[j + 1]))
        lj = global_loc
        sj = global_scale
        if ej.size >= MIN_BIN_SAMPLES:
            try:
                if FIX_LOC is None:
                    _, lj_fit, sj_fit = gennorm.fit(ej, f0=float(global_beta))
                else:
                    _, _, sj_fit = gennorm.fit(ej, f0=float(global_beta), floc=float(global_loc))
                    lj_fit = float(global_loc)
                if np.isfinite(lj_fit):
                    lj = float(lj_fit)
                if np.isfinite(sj_fit) and sj_fit > 0:
                    sj = float(sj_fit)
            except Exception:
                pass
        sj = float(np.clip(sj, scale_floor, scale_cap))
        wj = float(counts[j] / (counts[j] + SHRINK_COUNT))
        locs[j] = float((1.0 - wj) * global_loc + wj * lj)
        log_sj = float(np.log(max(sj, 1e-12)))
        scales[j] = float(np.clip(np.exp((1.0 - wj) * global_log_scale + wj * log_sj), scale_floor, scale_cap))
    anchor_w = np.clip(counts.astype(float), 1.0, None)
    loc_tck = fit_spline_tck(anchor_x, locs, anchor_w, k=1)
    loc_i = eval_spline_tck(loc_tck, yhat)
    log_scales = np.log(np.clip(scales, scale_floor, scale_cap))
    sigma_candidates = []
    log_scale_tck_free = fit_spline_tck(anchor_x, log_scales, anchor_w, k=1)
    sigma_i_free = np.clip(np.exp(eval_spline_tck(log_scale_tck_free, yhat)), scale_floor, scale_cap)
    nll_free = float(np.mean(ged_nll(err, global_beta, loc_i, sigma_i_free)))
    sigma_candidates.append((nll_free, None, log_scales, log_scale_tck_free, sigma_i_free))
    if SIGMA_MONOTONE and log_scales.size >= 2:
        for increasing in (True, False):
            log_iso = isotonic_regression(log_scales, anchor_w, increasing=increasing)
            tck = fit_spline_tck(anchor_x, log_iso, anchor_w, k=1, s=0.0)
            sigma_i = np.clip(np.exp(eval_spline_tck(tck, yhat)), scale_floor, scale_cap)
            nll = float(np.mean(ged_nll(err, global_beta, loc_i, sigma_i)))
            sigma_candidates.append((nll, bool(increasing), log_iso, tck, sigma_i))
    sigma_candidates.sort(key=lambda t: t[0])
    _, best_increasing, best_log_scales, log_scale_tck, sigma_i_raw = sigma_candidates[0]
    if best_increasing is None:
        sigma_monotone = False
        sigma_monotone_increasing = None
        scales = np.clip(np.exp(log_scales), scale_floor, scale_cap)
    else:
        sigma_monotone = True
        sigma_monotone_increasing = bool(best_increasing)
        scales = np.clip(np.exp(best_log_scales), scale_floor, scale_cap)
    z = (err - loc_i) / np.clip(sigma_i_raw, scale_floor, None)
    betas = np.full(n_eff_bins, global_beta, dtype=float)
    for j in range(n_eff_bins):
        in_bin = idx == j
        zj = z[in_bin]
        bj = global_beta
        if zj.size >= MIN_BIN_SAMPLES:
            try:
                bj_fit, _, _ = gennorm.fit(zj, floc=0.0, fscale=1.0)
                if np.isfinite(bj_fit) and bj_fit > 0:
                    bj = float(np.clip(bj_fit, BETA_MIN, BETA_MAX))
            except Exception:
                pass
        wj_beta = float(counts[j] / (counts[j] + BETA_SHRINK_COUNT))
        betas[j] = float(np.clip((1.0 - wj_beta) * global_beta + wj_beta * bj, BETA_MIN, BETA_MAX))
    beta_tck = fit_spline_tck(anchor_x, np.log(np.clip(betas, BETA_MIN, BETA_MAX)), anchor_w, k=1)
    beta_i = np.clip(np.exp(eval_spline_tck(beta_tck, yhat)), BETA_MIN, BETA_MAX)
    sigma_i = np.clip(sigma_i_raw * float(SIGMA_INFLATION), scale_floor * float(SIGMA_INFLATION), scale_cap * float(SIGMA_INFLATION))
    nll_cond = float(np.mean(ged_nll(err, beta_i, loc_i, sigma_i)))
    nll_base = float(np.mean(ged_nll(err, global_beta, global_loc, global_scale)))
    nll_gain = nll_base - nll_cond
    u = conditional_cdf(err, beta_i, loc_i, sigma_i)
    ks = kstest(u, 'uniform', alternative='two-sided', mode='auto')
    return {'param_form': 'spline', 'beta': global_beta, 'global_beta': float(global_beta), 'loc': global_loc, 'global_loc': float(global_loc), 'global_scale': float(global_scale), 'n_bins_target': int(n_bins_target), 'edges': edges, 'anchors_x': anchor_x, 'beta_tck': beta_tck, 'loc_tck': loc_tck, 'log_scale_tck': log_scale_tck, 'scale_floor': scale_floor, 'scale_cap': scale_cap, 'sigma_inflation': float(SIGMA_INFLATION), 'betas': betas, 'locs': locs, 'scales': scales, 'counts': counts, 'nll_base': nll_base, 'nll_cond': nll_cond, 'nll_gain': nll_gain, 'pit_ks_d': float(ks.statistic), 'pit_ks_p': float(ks.pvalue), 'mean_e': mean_e, 'std_e': std_e, 'se_mean': se_mean, 'loc_margin': loc_margin, 'sigma_monotone': bool(sigma_monotone), 'sigma_monotone_increasing': sigma_monotone_increasing}
