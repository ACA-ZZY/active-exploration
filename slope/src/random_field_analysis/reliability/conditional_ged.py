from __future__ import annotations
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import splev
from scipy.stats import gennorm
EPS_CDF = 1e-12
BETA_MIN_DEFAULT = 0.3
BETA_MAX_DEFAULT = 8.0
TEST_COUNT_DEFAULT = 500
MIN_SAMPLES_PER_SLOPE_DEFAULT = 30
PAIR_FILE = 'fs_eval_pairs.csv'
PRED_FILE = 'fs_pred.txt'
TRUE_FILE = 'fsV.txt'
ERR_FILE = 'fs_error.xlsx'
ERR_COL_PREFERRED = 'Loss Values'
ERR_COL_CANDIDATES = ['Loss Values', 'Loss', 'loss', 'Error', 'Errors', 'error', 'residual', 'Residual']

def _parse_float_list(text: Optional[str]) -> np.ndarray:
    if text is None:
        return np.asarray([], dtype=float)
    toks = [tok.strip() for tok in str(text).replace(';', ',').split(',') if tok.strip()]
    return np.asarray([float(tok) for tok in toks], dtype=float) if toks else np.asarray([], dtype=float)

def _parse_tck(kv: Dict[str, str], prefix: str):
    t = _parse_float_list(kv.get(f'{prefix}T'))
    c = _parse_float_list(kv.get(f'{prefix}C'))
    k_raw = kv.get(f'{prefix}K')
    if t.size < 2 or c.size < 2 or k_raw is None:
        return None
    return {'t': t, 'c': c, 'k': int(float(k_raw))}

def _eval_tck(tck: dict, x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    t = np.asarray(tck['t'], dtype=float)
    k = int(tck['k'])
    if t.size >= 2 * k + 2:
        xx = np.clip(xx, t[k], t[-k - 1])
    return np.asarray(splev(xx, (tck['t'], tck['c'], k), ext=3), dtype=float)

def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)

def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        span = max(float(np.std(x)), 1e-06) if x.size else 1e-06
        center = float(np.mean(x)) if x.size else 0.0
        edges = np.asarray([center - span, center + span], dtype=float)
    return edges

def _safe_scale(scale: float) -> float:
    return float(max(scale, 1e-06))

def _choose_error_col(df: pd.DataFrame) -> str:
    for col in ERR_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find error column in {ERR_FILE}. columns={list(df.columns)}')

def _read_float_txt(path: Path) -> np.ndarray:
    with path.open('r', encoding='utf-8') as handle:
        return np.asarray([float(line.strip()) for line in handle if line.strip()], dtype=float)

def load_error_yhat_fold(fold_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    fold_path = Path(fold_dir)
    pair_path = fold_path / PAIR_FILE
    if pair_path.exists():
        pair = pd.read_csv(pair_path)
        if not {'y_hat', 'error'}.issubset(pair.columns):
            raise ValueError(f'{pair_path} must contain y_hat and error')
        err = pd.to_numeric(pair['error'], errors='raise').to_numpy(dtype=float)
        yhat = pd.to_numeric(pair['y_hat'], errors='raise').to_numpy(dtype=float)
    else:
        err_df = pd.read_excel(fold_path / ERR_FILE)
        err = pd.to_numeric(err_df[_choose_error_col(err_df)], errors='raise').to_numpy(dtype=float)
        if (fold_path / PRED_FILE).exists():
            yhat = _read_float_txt(fold_path / PRED_FILE)
        elif (fold_path / TRUE_FILE).exists():
            ytrue = _read_float_txt(fold_path / TRUE_FILE)
            if len(err) != len(ytrue):
                raise ValueError('Error and true-response row counts differ')
            yhat = ytrue + err
        else:
            raise FileNotFoundError(f'Missing {PAIR_FILE}, {PRED_FILE}, or {TRUE_FILE} in {fold_path}')
    if err.shape != yhat.shape or not np.all(np.isfinite(err) & np.isfinite(yhat)):
        raise ValueError('Paired responses must have equal lengths and finite values')
    return (err, yhat)

def split_fit_test(err_all: np.ndarray, yhat_all: np.ndarray, test_count: int=TEST_COUNT_DEFAULT):
    if err_all.size < test_count + MIN_SAMPLES_PER_SLOPE_DEFAULT:
        raise ValueError(f'Not enough rows: total={err_all.size}, expected > {test_count}')
    return {'test': (err_all[:test_count], yhat_all[:test_count]), 'fit': (err_all[test_count:], yhat_all[test_count:])}

def bucket_by_slope(err: np.ndarray, yhat: np.ndarray, start_index: int, num_slopes: int):
    out_err = {sid: [] for sid in range(int(num_slopes))}
    out_yhat = {sid: [] for sid in range(int(num_slopes))}
    for local_idx, (e, yh) in enumerate(zip(err, yhat)):
        sid = int((start_index + local_idx) % int(num_slopes))
        out_err[sid].append(float(e))
        out_yhat[sid].append(float(yh))
    return {sid: (np.asarray(out_err[sid], dtype=float), np.asarray(out_yhat[sid], dtype=float)) for sid in range(int(num_slopes))}

@dataclass
class ConditionalGEDRuntime:
    mode: str
    param_form: str
    global_beta: float
    global_loc: float
    global_scale: float
    sigma_inflation: float = 1.0
    beta_min: float = BETA_MIN_DEFAULT
    beta_max: float = BETA_MAX_DEFAULT
    scale_floor: float = 1e-08
    scale_cap: float = 1.0
    beta_tck: Optional[dict] = None
    loc_tck: Optional[dict] = None
    log_scale_tck: Optional[dict] = None
    edges: Optional[np.ndarray] = None
    betas: Optional[np.ndarray] = None
    locs: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None

    @staticmethod
    def from_single(beta: float, loc: float, scale: float, sigma_inflation: float=1.0, beta_min: float=BETA_MIN_DEFAULT, beta_max: float=BETA_MAX_DEFAULT):
        sc = float(scale)
        floor = max(sc * 0.05, 1e-08)
        cap = max(sc * 5.0, floor * 1.05)
        return ConditionalGEDRuntime(mode='single', param_form='single', global_beta=float(beta), global_loc=float(loc), global_scale=sc, sigma_inflation=float(sigma_inflation), beta_min=float(beta_min), beta_max=float(beta_max), scale_floor=floor, scale_cap=cap)

    @staticmethod
    def from_params_file(path: str | Path):
        kv: Dict[str, str] = {}
        with Path(path).open('r', encoding='utf-8') as handle:
            for raw in handle:
                line = raw.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    kv[key.strip()] = value.strip()
        if 'PitX' in kv or 'PitY' in kv:
            raise ValueError('Recalibrated parameter files are not supported')
        mode = str(kv.get('Mode', 'single')).strip().lower()
        param_form = str(kv.get('ParamForm', 'single')).strip().lower()
        global_beta = float(kv.get('GlobalBeta', kv.get('Beta', 1.0)))
        global_loc = float(kv.get('GlobalLoc', kv.get('Loc', 0.0)))
        global_scale = float(kv.get('GlobalScale', kv.get('Scale', 1.0)))
        sigma_inflation = float(kv.get('SigmaInflation', 1.0))
        beta_min = float(kv.get('BetaMin', BETA_MIN_DEFAULT))
        beta_max = float(kv.get('BetaMax', BETA_MAX_DEFAULT))
        clip_raw = kv.get('BETA_CLIP')
        if clip_raw:
            try:
                clip_pair = ast.literal_eval(clip_raw)
                beta_min, beta_max = (float(clip_pair[0]), float(clip_pair[1]))
            except Exception:
                pass
        scale_floor = float(kv.get('SCALE_FLOOR', max(global_scale * 0.05, 1e-08)))
        scale_cap = float(kv.get('SCALE_CAP', max(global_scale * 5.0, scale_floor * 1.05)))
        if mode == 'conditional_yhat' and param_form == 'spline':
            beta_tck = _parse_tck(kv, 'SplineBeta')
            loc_tck = _parse_tck(kv, 'SplineLoc')
            log_scale_tck = _parse_tck(kv, 'SplineLogScale')
            if beta_tck is None or loc_tck is None or log_scale_tck is None:
                raise ValueError(f'Invalid spline tck in {path}')
            return ConditionalGEDRuntime(mode, param_form, global_beta, global_loc, global_scale, sigma_inflation, beta_min, beta_max, scale_floor, scale_cap, beta_tck, loc_tck, log_scale_tck)
        if mode == 'conditional_yhat':
            edges = _parse_float_list(kv.get('BinEdges'))
            betas = _parse_float_list(kv.get('BinBetas'))
            locs = _parse_float_list(kv.get('BinLocs'))
            scales = _parse_float_list(kv.get('BinScales'))
            if edges.size < 2 or scales.size < 1 or edges.size != scales.size + 1:
                raise ValueError(f'Invalid binned parameters in {path}')
            if betas.size == 0:
                betas = np.full_like(scales, global_beta, dtype=float)
            if locs.size == 0:
                locs = np.full_like(scales, global_loc, dtype=float)
            return ConditionalGEDRuntime(mode=mode, param_form='binned', global_beta=global_beta, global_loc=global_loc, global_scale=global_scale, sigma_inflation=sigma_inflation, beta_min=beta_min, beta_max=beta_max, scale_floor=scale_floor, scale_cap=scale_cap, edges=edges, betas=betas, locs=locs, scales=scales)
        return ConditionalGEDRuntime.from_single(global_beta, global_loc, global_scale, sigma_inflation, beta_min, beta_max)

    def conditional_params(self, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(y_hat, dtype=float)
        flat = y.reshape(-1)
        if self.mode == 'single' or self.param_form == 'single':
            beta = np.full_like(flat, self.global_beta, dtype=float)
            loc = np.full_like(flat, self.global_loc, dtype=float)
            sigma_raw = np.full_like(flat, self.global_scale, dtype=float)
        elif self.param_form == 'spline':
            beta = np.clip(np.exp(_eval_tck(self.beta_tck, flat)), self.beta_min, self.beta_max)
            loc = _eval_tck(self.loc_tck, flat)
            sigma_raw = np.exp(_eval_tck(self.log_scale_tck, flat))
        else:
            idx = _assign_bins(flat, self.edges)
            beta = np.asarray(self.betas[idx], dtype=float)
            loc = np.asarray(self.locs[idx], dtype=float)
            sigma_raw = np.asarray(self.scales[idx], dtype=float)
        sigma_raw = np.clip(sigma_raw, self.scale_floor, self.scale_cap)
        sigma = np.clip(sigma_raw * self.sigma_inflation, self.scale_floor * self.sigma_inflation, self.scale_cap * self.sigma_inflation)
        return (beta.reshape(y.shape), loc.reshape(y.shape), sigma.reshape(y.shape))

    def cdf_error(self, e: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        beta, loc, sigma = self.conditional_params(y_hat)
        u = np.asarray(gennorm.cdf(e, beta, loc=loc, scale=sigma), dtype=float)
        u = np.where(np.isfinite(u), u, 0.5)
        u = np.clip(u, EPS_CDF, 1.0 - EPS_CDF)
        return u

    def p_event_from_yhat(self, y_hat: np.ndarray, threshold: float=1.0, error_sign: int=1, event: str='le') -> np.ndarray:
        y = np.asarray(y_hat, dtype=float)
        sign = 1 if int(np.sign(error_sign)) >= 0 else -1
        ev = str(event).strip().lower()
        if ev not in {'le', 'ge'}:
            raise ValueError("event must be 'le' or 'ge'")
        if sign == 1:
            z = y - float(threshold)
            cdf = self.cdf_error(z, y)
            p = 1.0 - cdf if ev == 'le' else cdf
        else:
            z = float(threshold) - y
            cdf = self.cdf_error(z, y)
            p = cdf if ev == 'le' else 1.0 - cdf
        return np.clip(np.asarray(p, dtype=float), EPS_CDF, 1.0 - EPS_CDF)

    def p_fail(self, y_hat: np.ndarray, threshold: float=1.0, error_sign: int=1) -> np.ndarray:
        return self.p_event_from_yhat(y_hat=y_hat, threshold=threshold, error_sign=error_sign, event='le')
    fit_details: Optional[dict] = None

def load_conditional_ged(path: str | Path) -> ConditionalGEDRuntime:
    return ConditionalGEDRuntime.from_params_file(path)
_RE_FOLD_LINE = re.compile('^\\s*===\\s*Fold:\\s*([^\\s=]+)\\s*===\\s*$')
_RE_SLOPE_LINE = re.compile('^\\s*Slope\\s+(\\d+)\\s*:\\s*Mode=([A-Za-z_]+)\\s*,\\s*Beta=([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*,\\s*Loc=([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*,\\s*Scale=([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)')

def load_fold_slope_models_from_summary(summary_path: str | Path, fold_name: str, sigma_inflation: float=1.0):
    models: Dict[int, ConditionalGEDRuntime] = {}
    current_fold = None
    with Path(summary_path).open('r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            fold_match = _RE_FOLD_LINE.match(line)
            if fold_match:
                current_fold = fold_match.group(1).strip()
                continue
            if current_fold != str(fold_name):
                continue
            slope_match = _RE_SLOPE_LINE.match(line)
            if slope_match:
                models[int(slope_match.group(1))] = ConditionalGEDRuntime.from_single(beta=float(slope_match.group(3)), loc=float(slope_match.group(4)), scale=float(slope_match.group(5)), sigma_inflation=sigma_inflation)
    return models

def fit_conditional_ged_runtime(err: np.ndarray, yhat: np.ndarray, scale_inflation: float=1.12) -> ConditionalGEDRuntime:
    from .fitting import fit_conditional_ged
    errors = np.asarray(err, dtype=float).reshape(-1)
    predictions = np.asarray(yhat, dtype=float).reshape(-1)
    if errors.shape != predictions.shape or not np.all(np.isfinite(errors) & np.isfinite(predictions)):
        raise ValueError('Fit arrays must be paired and finite')
    if not np.isfinite(scale_inflation) or scale_inflation <= 0:
        raise ValueError('Scale multiplier must be positive')
    details = fit_conditional_ged(errors, predictions)
    return ConditionalGEDRuntime(mode='conditional_yhat', param_form='spline', global_beta=details['global_beta'], global_loc=details['global_loc'], global_scale=details['global_scale'], sigma_inflation=float(scale_inflation), scale_floor=details['scale_floor'], scale_cap=details['scale_cap'], beta_tck=details['beta_tck'], loc_tck=details['loc_tck'], log_scale_tck=details['log_scale_tck'], edges=details['edges'], betas=details['betas'], locs=details['locs'], scales=details['scales'], fit_details=details)
