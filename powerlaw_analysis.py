"""
powerlaw_analysis.py
====================
Automatic power-law region identification and fitting for SAXS .dat / .txt files.

Physics background
------------------
In the intermediate- and high-q regime, many scattering systems follow a
power law:

    I(q) ∝ q^{-α}

Taking the natural logarithm linearises the relation:

    ln I(q) = -α · ln q + ln A

Fitting ln(I) vs ln(q) by weighted linear regression gives:
    slope      = -α      →   α = -slope
    intercept  = ln(A)   →   A = exp(intercept)

The exponent α carries physical meaning:
    α = 4    → smooth interfaces (Porod law, sharp surface)
    α = 2    → Gaussian coil / thin disc
    α = 1    → thin rod
    α ∈ (1,3) → mass fractal
    α ∈ (3,4) → surface fractal

Automated region search
-----------------------
1.  Restrict candidate data to q > Q_MIN_FIT (exclude Guinier / transition
    regime) and q < Q_MAX_FIT if set.  If a guinier_results.csv is provided,
    the per-file qmax from that table is used as Q_MIN_FIT automatically.
2.  Exclude points where σ_i / I_i > MAX_RELATIVE_ERROR (noise-dominated).
3.  Enumerate every possible window (i_start, i_end) within the remaining
    points, subject to MIN_WINDOW_POINTS and MAX_WINDOW_POINTS.
4.  For each window fit ln(I) = -α·ln(q) + const by weighted least squares
    using weights w_i = I_i²/σ_i² (falls back to unweighted OLS if no errors).
5.  Apply hard rejection: R² ≥ MIN_R2.
6.  Assess stability: collect α values of all overlapping valid windows and
    compute CV = σ(α)/μ(α).  Small CV → stable power law.
7.  Select the longest window with CV ≤ MAX_ALPHA_CV; fall back to lowest CV.
8.  Two-regime check: split the valid q range at each interior point, fit two
    separate power laws, report both α₁ and α₂ if the two-regime χ²_r is
    lower than the single-regime χ²_r by more than TWO_REGIME_IMPROVEMENT.

Usage
-----
1.  Edit the USER SETTINGS block below.
2.  Run:   python powerlaw_analysis.py
3.  Results appear in OUTPUT_FOLDER as:
      powerlaw_results.csv          — one row per input file
      <sample>_powerlaw.png         — 4-panel diagnostic figure per file
"""

import re
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import f as f_dist


# ====================================================================
# USER SETTINGS  — edit here, nowhere else
# ====================================================================

# BASE_FOLDER   = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS\process\Tina ONLINE AF4 SAXS")

# # ↓ Only change this line when switching datasets
# SAMPLE_FOLDER = "F3C_mRNA_TRIS_scan-84558_shotsE_subtracted_50frame_blocks2"
# # F3_empty_TRIS_scan-84555_shotsE_subtracted_50frame_blocks2
# # F3_mRNA_TRIS_scan-84556_shotsE_subtracted_50frame_blocks2
# # F3C_emtpy_TRIS_scan-84557_shotsE_subtracted_50frame_blocks2
# # F3C_mRNA_TRIS_scan-84558_shotsE_subtracted_50frame_blocks2

BASE_FOLDER   = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS")

# ↓ Only change this line when switching datasets
SAMPLE_FOLDER = "Tina OFFLINE SAXS"


INPUT_FOLDER  = BASE_FOLDER / SAMPLE_FOLDER
OUTPUT_FOLDER = INPUT_FOLDER / f"powerlaw2_{date.today():%Y-%m-%d}"

FILE_EXTENSIONS = [".dat", ".txt"]   # case-insensitive; extend if needed

# ---- q-range boundaries -------------------------------------------
# Lower q boundary for the power-law fit (Å⁻¹).
# Points below this q are in the Guinier / transition regime and are excluded.
# Overridden per-file automatically if GUINIER_CSV is set.
Q_MIN_FIT = 0.02

# Upper q boundary (Å⁻¹).  None = use noise cutoff only.
Q_MAX_FIT = None

# ---- Noise cutoff --------------------------------------------------
# Exclude points where σ_i / I_i exceeds this value (relative error too large).
MAX_RELATIVE_ERROR = 0.5

# ---- Fitting window size -------------------------------------------
MIN_WINDOW_POINTS = 10
MAX_WINDOW_POINTS = 100

# ---- Linearity threshold -------------------------------------------
MIN_R2 = 0.990

# ---- Stability criterion -------------------------------------------
# CV = σ(α)/μ(α) across overlapping valid windows.
MAX_ALPHA_CV = 0.05

# ---- Two-regime detection ------------------------------------------
# A second power-law regime is reported when the F-test p-value is below
# this significance threshold (default 0.05 = 95 % confidence).
F_TEST_ALPHA = 0.05

# ---- Weighted fitting ----------------------------------------------
# If True and per-point errors are available, weight log-space residuals
# by σ(ln I) = σ(I)/I.  Falls back gracefully to unweighted OLS otherwise.
USE_WEIGHTED_FIT = True

# ---- Guinier CSV ---------------------------------------------------
# Path to guinier_results.csv produced by guinier_analysis_robust.py.
# When set, the per-file qmax column is used as the lower q boundary
# for that file (overrides Q_MIN_FIT).  Set to None to disable.
GUINIER_CSV = INPUT_FOLDER / "guinier_results.csv"

# ---- Plotting ------------------------------------------------------
SAVE_PLOTS = True
PLOT_DPI   = 150

# ====================================================================
# END OF USER SETTINGS
# ====================================================================


# --------------------------------------------------------------------
# Section 1: Data loading
# --------------------------------------------------------------------

def load_saxs_file(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read a SAXS data file with columns  q  I  [err].

    •  Lines starting with '#' or that cannot be parsed as floats are skipped.
    •  Handles files with only q and I (err returned as None).
    •  Data are sorted by ascending q before return.

    Returns
    -------
    q   : 1-D array of scattering vectors (Å⁻¹)
    I   : 1-D array of scattering intensities
    err : 1-D array of uncertainties, or None if absent / all-invalid
    """
    rows: List[List[float]] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", ".")
            parts = re.split(r"[\s;]+", line)
            try:
                nums = [float(p) for p in parts if p]
            except ValueError:
                continue
            if len(nums) >= 2:
                rows.append(nums[:3] if len(nums) >= 3 else nums[:2] + [float("nan")])

    if not rows:
        raise ValueError(f"No numeric data found in {filepath.name}")

    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]

    q   = arr[:, 0]
    I   = arr[:, 1]
    err = arr[:, 2] if arr.shape[1] >= 3 else None

    if err is not None:
        if np.all(~np.isfinite(err)) or np.all(err == 0):
            err = None

    return q, I, err


# --------------------------------------------------------------------
# Section 2: Weighted linear regression (closed-form, no scipy)
# --------------------------------------------------------------------

def _linfit(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    """
    Fit  y = slope·x + intercept  by (optionally weighted) least squares.

    Parameters
    ----------
    x, y  : data arrays (must have the same length ≥ 3)
    w     : weight array  w_i = 1/σ_i²  (proper inverse-variance weights).
            If None, ordinary unweighted least squares is used.

    Returns
    -------
    dict with keys: slope, slope_err, intercept, intercept_err, r2, chi2_red
    or None if the system is degenerate.

    Implementation notes
    --------------------
    Weighted normal equations (closed form):
        slope     = (Sw·Swxy − Swx·Swy) / D
        intercept = (Swy·Swx2 − Swx·Swxy) / D
        D = Sw·Swx2 − Swx²

    Covariance when w_i = 1/σ_i²:
        var(slope)     = Sw / D
        var(intercept) = Swx2 / D
    """
    n = len(x)
    if n < 3:
        return None

    if w is not None:
        Sw   = float(np.sum(w))
        Swx  = float(np.sum(w * x))
        Swy  = float(np.sum(w * y))
        Swx2 = float(np.sum(w * x**2))
        Swxy = float(np.sum(w * x * y))

        D = Sw * Swx2 - Swx**2
        if D <= 0.0:
            return None

        slope     = (Sw * Swxy - Swx * Swy) / D
        intercept = (Swy * Swx2 - Swx * Swxy) / D

        var_slope     = Sw   / D
        var_intercept = Swx2 / D

        resid    = y - (slope * x + intercept)
        chi2     = float(np.sum(resid**2 * w))
        chi2_red = chi2 / (n - 2)

    else:
        xm  = float(np.mean(x))
        ym  = float(np.mean(y))
        Sxx = float(np.sum((x - xm)**2))
        if Sxx == 0.0:
            return None

        slope     = float(np.sum((x - xm) * (y - ym))) / Sxx
        intercept = ym - slope * xm

        resid = y - (slope * x + intercept)
        s2    = float(np.sum(resid**2)) / (n - 2)

        var_slope     = s2 / Sxx
        var_intercept = s2 * float(np.sum(x**2)) / (n * Sxx)
        chi2_red      = float("nan")

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return dict(
        slope         = float(slope),
        slope_err     = float(np.sqrt(max(var_slope,     0.0))),
        intercept     = float(intercept),
        intercept_err = float(np.sqrt(max(var_intercept, 0.0))),
        r2            = float(r2),
        chi2_red      = float(chi2_red),
    )


# --------------------------------------------------------------------
# Section 3: Single-window power-law fit
# --------------------------------------------------------------------

def _fit_window(
    q_win:   np.ndarray,
    I_win:   np.ndarray,
    err_win: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """
    Fit I(q) = A · q^{-α} for a pre-selected data window in log-log space.

    Steps
    -----
    1.  Keep only finite, strictly positive (q, I) pairs.
    2.  Transform to log-space:  y = ln(I),  x = ln(q).
    3.  Propagate uncertainties:  σ(ln I) ≈ σ(I)/I.
    4.  Build weights w_i = I_i²/σ_i² as specified.
    5.  Run weighted or unweighted linear regression.
    6.  Extract α = -slope,  A = exp(intercept).
    7.  Apply hard rejection: R² ≥ MIN_R2.

    Returns a dict of fit diagnostics, or None if the window is invalid.
    """
    mask = np.isfinite(q_win) & (q_win > 0) & np.isfinite(I_win) & (I_win > 0)
    if err_win is not None:
        mask &= np.isfinite(err_win) & (err_win > 0)
    if int(mask.sum()) < 3:
        return None

    q_  = q_win[mask]
    I_  = I_win[mask]
    x   = np.log(q_)
    y   = np.log(I_)

    # Weights: w_i = I_i² / σ_i²  (as specified), equivalent to 1/σ(lnI)²
    w        = None
    weighted = False
    if USE_WEIGHTED_FIT and err_win is not None:
        sigma_lnI = err_win[mask] / I_
        if np.all(sigma_lnI > 0) and np.all(np.isfinite(sigma_lnI)):
            w        = 1.0 / sigma_lnI**2   # = I²/σ²
            weighted = True

    fit = _linfit(x, y, w)
    if fit is None:
        return None

    if fit["r2"] < MIN_R2:
        return None

    alpha     = -fit["slope"]
    alpha_err =  fit["slope_err"]
    prefactor     = float(np.exp(fit["intercept"]))
    prefactor_err = prefactor * fit["intercept_err"]

    n_pts = int(mask.sum())

    return dict(
        alpha         = alpha,
        alpha_err     = alpha_err,
        prefactor     = prefactor,
        prefactor_err = prefactor_err,
        intercept     = fit["intercept"],
        intercept_err = fit["intercept_err"],
        r2            = fit["r2"],
        chi2_red      = fit["chi2_red"],
        weighted      = weighted,
        n_points      = n_pts,
        qmin          = float(q_.min()),
        qmax          = float(q_.max()),
    )


# --------------------------------------------------------------------
# Section 4: Window selection (longest stable window)
# --------------------------------------------------------------------

def _select_best_window(
    valid_windows: List[Dict[str, Any]],
    cvs:           List[float],
) -> int:
    """
    Return the index of the best window using the longest-stable-window rule.

    Among all windows with CV ≤ MAX_ALPHA_CV, pick the one with the most
    points.  If no window passes the CV threshold, fall back to the window
    with the lowest CV (regardless of size).
    """
    stable_indices = [i for i, cv in enumerate(cvs) if cv <= MAX_ALPHA_CV]
    if stable_indices:
        return max(stable_indices, key=lambda i: valid_windows[i]["n_points"])
    return int(np.argmin(cvs))


# --------------------------------------------------------------------
# Section 5: Two-regime detection
# --------------------------------------------------------------------

def _fit_two_regimes(
    q_fit:   np.ndarray,
    I_fit:   np.ndarray,
    err_fit: Optional[np.ndarray],
    single_chi2_red: float,
) -> Optional[Dict[str, Any]]:
    """
    Try splitting the q range at each interior point and fitting two separate
    power laws.  Report the best split if the F-test p-value is below
    F_TEST_ALPHA, accounting for the 2 extra parameters introduced by splitting.

    F = [(RSS1 - RSS2) / Δp] / [RSS2 / (N - p2)]
    where RSS1 = single-regime residual SS, RSS2 = combined two-regime residual
    SS, Δp = 2 (one extra slope + one extra intercept), p2 = 4, N = n_points.

    Returns a dict with keys alpha1, alpha1_err, alpha2, alpha2_err,
    q_transition, chi2_red_two, f_statistic, f_pvalue, or None if no split
    passes the F-test.
    """
    N = len(q_fit)
    if not np.isfinite(single_chi2_red) or N < 2 * MIN_WINDOW_POINTS + 1:
        return None

    # Residual sum of squares for the single-regime fit (RSS1 = chi2_red * DOF)
    dof_single = N - 2
    rss_single = single_chi2_red * dof_single

    I_  = I_fit
    w_  = None
    if USE_WEIGHTED_FIT and err_fit is not None:
        sigma_lnI = err_fit / I_
        valid_w   = np.isfinite(sigma_lnI) & (sigma_lnI > 0)
        if np.all(valid_w):
            w_ = 1.0 / sigma_lnI**2

    best: Optional[Dict[str, Any]] = None
    best_pvalue = F_TEST_ALPHA   # only keep splits that beat this threshold

    for split in range(MIN_WINDOW_POINTS, N - MIN_WINDOW_POINTS):
        # Left segment
        q1 = q_fit[:split]
        e1 = err_fit[:split] if err_fit is not None else None
        r1 = _fit_window(q1, I_[:split], e1)
        if r1 is None:
            continue

        # Right segment
        q2 = q_fit[split:]
        e2 = err_fit[split:] if err_fit is not None else None
        r2 = _fit_window(q2, I_[split:], e2)
        if r2 is None:
            continue

        dof1 = r1["n_points"] - 2
        dof2 = r2["n_points"] - 2
        if dof1 <= 0 or dof2 <= 0:
            continue

        c1 = r1["chi2_red"] if np.isfinite(r1["chi2_red"]) else float("nan")
        c2 = r2["chi2_red"] if np.isfinite(r2["chi2_red"]) else float("nan")
        if not (np.isfinite(c1) and np.isfinite(c2)):
            continue

        # Combined RSS2 and F-statistic
        rss2      = c1 * dof1 + c2 * dof2   # = RSS_left + RSS_right
        delta_p   = 2                         # extra params: one slope + one intercept
        p2        = 4                         # total params in two-regime model
        dof_two   = N - p2

        if dof_two <= 0 or rss2 <= 0:
            continue

        f_stat  = ((rss_single - rss2) / delta_p) / (rss2 / dof_two)
        p_value = float(f_dist.sf(f_stat, delta_p, dof_two))

        if p_value < best_pvalue:
            best_pvalue = p_value
            best = dict(
                alpha1        = r1["alpha"],
                alpha1_err    = r1["alpha_err"],
                alpha2        = r2["alpha"],
                alpha2_err    = r2["alpha_err"],
                q_transition  = float(q_fit[split]),
                chi2_red_two  = rss2 / dof_two,
                r2_1          = r1["r2"],
                r2_2          = r2["r2"],
                f_statistic   = float(f_stat),
                f_pvalue      = p_value,
            )

    return best


# --------------------------------------------------------------------
# Section 6: Automatic power-law region search
# --------------------------------------------------------------------

def find_powerlaw_region(
    q:         np.ndarray,
    I:         np.ndarray,
    err:       Optional[np.ndarray],
    q_min_fit: float,
) -> Dict[str, Any]:
    """
    Identify the best power-law region in a SAXS curve.

    See module docstring for the full algorithm description.

    Parameters
    ----------
    q, I, err   : full SAXS data arrays
    q_min_fit   : lower q boundary (Å⁻¹); excludes Guinier / transition regime

    Returns a flat dict containing all fit results and internal arrays for
    plotting (prefixed with _ for internal use).
    """
    nan_result: Dict[str, Any] = dict(
        status                 = "no valid region found",
        alpha                  = float("nan"),
        alpha_err              = float("nan"),
        alpha2                 = float("nan"),
        alpha2_err             = float("nan"),
        q_transition           = float("nan"),
        prefactor              = float("nan"),
        prefactor_err          = float("nan"),
        qmin                   = float("nan"),
        qmax                   = float("nan"),
        n_points               = float("nan"),
        r2                     = float("nan"),
        chi2_red               = float("nan"),
        stability_cv           = float("nan"),
        two_regime             = False,
        valid_windows          = [],
        _q_fit                 = np.array([]),
        _I_fit                 = np.array([]),
        _err_fit               = None,
        candidate_window_count = 0,
        valid_window_count     = 0,
    )

    # ----------------------------------------------------------------
    # Step 1 — apply q boundaries and noise cutoff
    # ----------------------------------------------------------------
    mask_pos = np.isfinite(q) & (q > 0) & np.isfinite(I) & (I > 0)
    if int(mask_pos.sum()) < MIN_WINDOW_POINTS:
        nan_result["status"] = "too few points with positive intensity"
        return nan_result

    q_  = q[mask_pos]
    I_  = I[mask_pos]
    e_  = err[mask_pos] if err is not None else None

    order = np.argsort(q_)
    q_ = q_[order];  I_ = I_[order]
    if e_ is not None:
        e_ = e_[order]

    # Lower q boundary
    mask_lo = q_ >= q_min_fit
    q_ = q_[mask_lo];  I_ = I_[mask_lo]
    if e_ is not None:
        e_ = e_[mask_lo]

    # Upper q boundary
    if Q_MAX_FIT is not None:
        mask_hi = q_ <= Q_MAX_FIT
        q_ = q_[mask_hi];  I_ = I_[mask_hi]
        if e_ is not None:
            e_ = e_[mask_hi]

    # Noise cutoff: exclude points where σ/I > MAX_RELATIVE_ERROR
    if e_ is not None:
        rel_err = e_ / I_
        mask_noise = np.isfinite(rel_err) & (rel_err <= MAX_RELATIVE_ERROR)
        q_ = q_[mask_noise];  I_ = I_[mask_noise]
        e_ = e_[mask_noise]

    if len(q_) < MIN_WINDOW_POINTS:
        nan_result["status"] = (
            f"too few points after q-range and noise filtering (found {len(q_)})"
        )
        return nan_result

    nan_result["_q_fit"]   = q_
    nan_result["_I_fit"]   = I_
    nan_result["_err_fit"] = e_

    # ----------------------------------------------------------------
    # Step 2+3 — enumerate windows and apply hard criteria
    # ----------------------------------------------------------------
    n = len(q_)
    valid_windows: List[Dict[str, Any]] = []
    candidate_window_count = 0
    valid_window_count     = 0

    for i_start in range(n - MIN_WINDOW_POINTS + 1):
        for i_end in range(
            i_start + MIN_WINDOW_POINTS,
            min(i_start + MAX_WINDOW_POINTS + 1, n + 1),
        ):
            candidate_window_count += 1
            res = _fit_window(
                q_[i_start:i_end],
                I_[i_start:i_end],
                e_[i_start:i_end] if e_ is not None else None,
            )
            if res is None:
                continue

            valid_window_count += 1
            res["i_start"] = i_start
            res["i_end"]   = i_end
            valid_windows.append(res)

    nan_result["candidate_window_count"] = candidate_window_count
    nan_result["valid_window_count"]     = valid_window_count

    if not valid_windows:
        nan_result["status"] = _diagnose_failure(q_, I_, e_)
        return nan_result

    # ----------------------------------------------------------------
    # Step 4 — stability assessment (CV of α across overlapping windows)
    # ----------------------------------------------------------------
    cvs: List[float] = []
    for k, wk in enumerate(valid_windows):
        sk, ek = wk["i_start"], wk["i_end"]
        len_k  = ek - sk
        neighbor_alphas = [wk["alpha"]]

        for j, wj in enumerate(valid_windows):
            if j == k:
                continue
            sj, ej = wj["i_start"], wj["i_end"]
            overlap = max(0, min(ek, ej) - max(sk, sj))
            min_len = min(len_k, ej - sj)
            if min_len > 0 and overlap >= 0.5 * min_len:
                neighbor_alphas.append(wj["alpha"])

        arr_a   = np.array(neighbor_alphas)
        mean_a  = float(arr_a.mean())
        cv      = float(arr_a.std() / abs(mean_a)) if mean_a != 0.0 else float("inf")
        cvs.append(cv)

    # ----------------------------------------------------------------
    # Step 5 — select the best window (longest stable window)
    # ----------------------------------------------------------------
    best_idx = _select_best_window(valid_windows, cvs)
    best     = valid_windows[best_idx]
    best_cv  = cvs[best_idx]

    status = (
        f"unstable power-law fit (CV={best_cv:.3f}); result may be unreliable"
        if best_cv > MAX_ALPHA_CV
        else "ok"
    )

    # ----------------------------------------------------------------
    # Step 6 — two-regime detection
    # ----------------------------------------------------------------
    i0, i1 = best["i_start"], best["i_end"]
    q_fit   = q_[i0:i1]
    I_fit   = I_[i0:i1]
    err_fit = e_[i0:i1] if e_ is not None else None

    two_regime_result = _fit_two_regimes(
        q_fit, I_fit, err_fit, best["chi2_red"]
    )
    two_regime = two_regime_result is not None

    result = dict(
        nan_result,
        status                 = status,
        alpha                  = best["alpha"],
        alpha_err              = best["alpha_err"],
        alpha2                 = two_regime_result["alpha2"]      if two_regime else float("nan"),
        alpha2_err             = two_regime_result["alpha2_err"]  if two_regime else float("nan"),
        q_transition           = two_regime_result["q_transition"] if two_regime else float("nan"),
        prefactor              = best["prefactor"],
        prefactor_err          = best["prefactor_err"],
        qmin                   = best["qmin"],
        qmax                   = best["qmax"],
        n_points               = best["n_points"],
        r2                     = best["r2"],
        chi2_red               = best["chi2_red"],
        stability_cv           = best_cv,
        two_regime             = two_regime,
        valid_windows          = valid_windows,
        _q_fit                 = q_,
        _I_fit                 = I_,
        _err_fit               = e_,
        _best_i_start          = i0,
        _best_i_end            = i1,
        _two_regime_result     = two_regime_result,
        candidate_window_count = candidate_window_count,
        valid_window_count     = valid_window_count,
    )
    return result


def _diagnose_failure(
    q_: np.ndarray,
    I_: np.ndarray,
    e_: Optional[np.ndarray],
) -> str:
    """
    When no valid window is found, probe rejection criteria individually
    to return the most informative failure message.
    """
    any_candidate = False
    any_r2_ok     = False

    for i_start in range(len(q_) - MIN_WINDOW_POINTS + 1):
        for i_end in range(
            i_start + MIN_WINDOW_POINTS,
            min(i_start + MAX_WINDOW_POINTS + 1, len(q_) + 1),
        ):
            # Check without R² filter
            mask = (
                np.isfinite(q_[i_start:i_end]) & (q_[i_start:i_end] > 0) &
                np.isfinite(I_[i_start:i_end]) & (I_[i_start:i_end] > 0)
            )
            if int(mask.sum()) < 3:
                continue
            any_candidate = True
            res = _fit_window(
                q_[i_start:i_end],
                I_[i_start:i_end],
                e_[i_start:i_end] if e_ is not None else None,
            )
            if res is not None:
                any_r2_ok = True

    if not any_candidate:
        return "too few valid points to form any window"
    if not any_r2_ok:
        return "no window achieves R2 >= MIN_R2 - data may not follow a power law in this q range"
    return "no window satisfying all criteria simultaneously"


# --------------------------------------------------------------------
# Section 7: Load Guinier CSV for per-file q_min overrides
# --------------------------------------------------------------------

def _load_guinier_qmax(csv_path: Optional[Path]) -> Dict[str, float]:
    """
    Load the per-file qmax values from a guinier_results.csv.
    Returns a dict mapping normalised filename stem → qmax (Å⁻¹).
    Returns an empty dict if the file does not exist or cannot be read.
    """
    if csv_path is None or not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "file" not in df.columns or "qmax" not in df.columns:
            return {}
        result: Dict[str, float] = {}
        for _, row in df.iterrows():
            stem = Path(str(row["file"])).stem.lower()
            val  = float(row["qmax"])
            if np.isfinite(val) and val > 0:
                result[stem] = val
        return result
    except Exception:
        return {}


# --------------------------------------------------------------------
# Section 8: Diagnostic plots
# --------------------------------------------------------------------

def _save_plots(
    q_full:  np.ndarray,
    I_full:  np.ndarray,
    err_full: Optional[np.ndarray],
    res:     Dict[str, Any],
    name:    str,
    out_dir: Path,
) -> None:
    """
    Save a 4-panel diagnostic figure for one SAXS file:

    Panel 1 — SAXS curve (log-log):  I(q) vs q, fitted region highlighted,
               two-regime boundary marked if applicable
    Panel 2 — log-log plot:  ln I vs ln q, fit line(s) and annotation box
    Panel 3 — Residuals with error bars vs ln q
    Panel 4 — α across all valid windows vs q_max of window
    """
    is_ok       = res["status"].startswith("ok")
    has_fit     = np.isfinite(res["alpha"])   # True also for unstable fits
    q_fit       = res["_q_fit"]
    I_fit       = res["_I_fit"]
    err_fit     = res["_err_fit"]
    i0          = res.get("_best_i_start")
    i1          = res.get("_best_i_end")
    two_r       = res.get("_two_regime_result")
    two_regime  = res["two_regime"]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Power-law Analysis — {name}", fontsize=13, fontweight="bold")

    # ── Panel 1: SAXS curve (log-log) ───────────────────────────────
    ax1.set_title("SAXS curve", fontsize=11)
    mpos = (I_full > 0) & np.isfinite(I_full) & np.isfinite(q_full) & (q_full > 0)
    ax1.loglog(q_full[mpos], I_full[mpos],
               ".", color="steelblue", ms=3, alpha=0.7, label="data")

    if has_fit and i0 is not None:
        q_w = q_fit[i0:i1];  I_w = I_fit[i0:i1]
        dot_color = "tomato" if is_ok else "black"
        unstable_suffix = "" if is_ok else " [unstable]"
        if two_regime and two_r is not None:
            q_t = two_r["q_transition"]
            mask1 = q_w <= q_t
            mask2 = q_w >  q_t
            ax1.loglog(q_w[mask1], I_w[mask1], "o", color="tomato",  ms=5,
                       zorder=5, label=f"regime 1 (α={two_r['alpha1']:.2f}){unstable_suffix}")
            ax1.loglog(q_w[mask2], I_w[mask2], "o", color="darkorange", ms=5,
                       zorder=5, label=f"regime 2 (α={two_r['alpha2']:.2f}){unstable_suffix}")
            ax1.axvline(q_t, color="grey", lw=1.2, ls="--",
                        label=f"$q_t$ = {q_t:.4f} Å⁻¹")
        else:
            ax1.loglog(q_w, I_w, "o", color=dot_color, ms=5,
                       zorder=5, label=f"power-law region (α={res['alpha']:.2f}){unstable_suffix}")

    ax1.set_xlabel(r"$q\ (\mathrm{\AA}^{-1})$", fontsize=10)
    ax1.set_ylabel(r"$I(q)$ (a.u.)", fontsize=10)
    ax1.legend(fontsize=8)

    # ── Panel 2: ln I vs ln q ────────────────────────────────────────
    ax2.set_title(r"$\ln I$ vs $\ln q$", fontsize=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf   = (I_fit > 0) & np.isfinite(I_fit) & (q_fit > 0)
        lnq  = np.log(q_fit[mf])
        lnI  = np.log(I_fit[mf])

    ax2.plot(lnq, lnI, ".", color="steelblue", ms=4, alpha=0.7, label="fit-range data")

    if has_fit and i0 is not None:
        q_w  = q_fit[i0:i1];  I_w = I_fit[i0:i1]
        mw   = (I_w > 0) & np.isfinite(I_w) & (q_w > 0)
        lnq_w = np.log(q_w[mw]);  lnI_w = np.log(I_w[mw])

        slope     = -res["alpha"]
        intercept =  np.log(res["prefactor"])
        dot_color = "tomato" if is_ok else "black"
        line_color = "darkred" if is_ok else "black"

        if two_regime and two_r is not None:
            # Draw two fit lines
            q_t   = two_r["q_transition"]
            m1    = q_w[mw] <= q_t;  m2 = q_w[mw] > q_t
            for mask_seg, seg_color, seg_alpha, seg_key in [
                (m1, "tomato",     two_r["alpha1"], "alpha1"),
                (m2, "darkorange", two_r["alpha2"], "alpha2"),
            ]:
                if mask_seg.sum() < 2:
                    continue
                lnq_s = lnq_w[mask_seg];  lnI_s = lnI_w[mask_seg]
                ax2.plot(lnq_s, lnI_s, "o", color=seg_color, ms=5, zorder=5)
                a  = -seg_alpha
                b  = float(np.mean(lnI_s - a * lnq_s))
                x_line = np.linspace(lnq_s[0] * 0.98, lnq_s[-1] * 1.02, 200)
                ax2.plot(x_line, a * x_line + b, "-", color=seg_color, lw=1.8)
        else:
            ax2.plot(lnq_w, lnI_w, "o", color=dot_color, ms=5, zorder=5,
                     label="fit window")
            x_line = np.linspace(lnq_w[0] * 0.98, lnq_w[-1] * 1.02, 200)
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, "-", color=line_color, lw=1.8, label="fit")

        # Annotation box
        chi2_str = (
            f"\n$\\chi^2_r$ = {res['chi2_red']:.3f}"
            if np.isfinite(res["chi2_red"])
            else ""
        )
        status_str = "" if is_ok else f"\n⚠ {res['status']}"
        if two_regime and two_r is not None:
            ann = (
                f"$\\alpha_1$ = {two_r['alpha1']:.3f} ± {two_r['alpha1_err']:.3f}\n"
                f"$\\alpha_2$ = {two_r['alpha2']:.3f} ± {two_r['alpha2_err']:.3f}\n"
                f"$q_t$ = {two_r['q_transition']:.4f} Å⁻¹\n"
                f"$q_{{\\min}}$ = {res['qmin']:.4f} Å⁻¹\n"
                f"$q_{{\\max}}$ = {res['qmax']:.4f} Å⁻¹\n"
                f"$R^2$ = {res['r2']:.4f}\n"
                f"F-test $p$ = {two_r['f_pvalue']:.3g}"
                f"{chi2_str}{status_str}"
            )
        else:
            ann = (
                f"$\\alpha$ = {res['alpha']:.3f} ± {res['alpha_err']:.3f}\n"
                f"$A$ = {res['prefactor']:.3g} ± {res['prefactor_err']:.1g}\n"
                f"$q_{{\\min}}$ = {res['qmin']:.4f} Å⁻¹\n"
                f"$q_{{\\max}}$ = {res['qmax']:.4f} Å⁻¹\n"
                f"$R^2$ = {res['r2']:.4f}"
                f"{chi2_str}{status_str}"
            )
        ax2.text(0.97, 0.97, ann, transform=ax2.transAxes,
                 ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax2.set_xlabel(r"$\ln\,q$", fontsize=10)
    ax2.set_ylabel(r"$\ln\,I(q)$", fontsize=10)
    ax2.legend(fontsize=8)

    # ── Panel 3: residuals ──────────────────────────────────────────
    ax3.set_title("Power-law fit residuals", fontsize=11)
    if has_fit and i0 is not None:
        q_w   = q_fit[i0:i1];  I_w = I_fit[i0:i1]
        e_w   = err_fit[i0:i1] if err_fit is not None else None
        mw    = (I_w > 0) & np.isfinite(I_w) & (q_w > 0)
        lnq_w = np.log(q_w[mw])
        lnI_w = np.log(I_w[mw])
        slope = -res["alpha"]
        b     = np.log(res["prefactor"])
        resid = lnI_w - (slope * lnq_w + b)
        dot_color = "tomato" if is_ok else "black"

        ax3.axhline(0, color="k", lw=0.9, ls="--", alpha=0.6)
        if e_w is not None:
            sigma_lnI = e_w[mw] / I_w[mw]
            ax3.errorbar(lnq_w, resid, yerr=sigma_lnI,
                         fmt="o", color=dot_color, ms=5, capsize=3,
                         elinewidth=0.8)
        else:
            ax3.plot(lnq_w, resid, "o", color=dot_color, ms=5)

        if len(resid) >= 6:
            ax3.plot(lnq_w, resid, "-", color=dot_color, alpha=0.35, lw=1)

        if two_regime and two_r is not None:
            ax3.axvline(np.log(two_r["q_transition"]), color="grey",
                        lw=1.2, ls="--", label="$q_t$")
            ax3.legend(fontsize=8)

        ax3.set_xlabel(r"$\ln\,q$", fontsize=10)
        ax3.set_ylabel(r"$\ln I_{\rm obs} - \ln I_{\rm fit}$", fontsize=10)
    else:
        ax3.text(0.5, 0.5, f"No valid fit\n\n{res['status']}",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=9, color="grey")
        ax3.set_axis_off()

    # ── Panel 4: α across all valid windows ─────────────────────────
    ax4.set_title(r"$\alpha$ across valid windows", fontsize=11)
    vws = res.get("valid_windows", [])
    if vws:
        alpha_all = np.array([w["alpha"]  for w in vws])
        qmax_all  = np.array([w["qmax"]   for w in vws])
        ax4.scatter(qmax_all, alpha_all, c="steelblue", s=14, alpha=0.45,
                    label="valid windows")
        if has_fit:
            line_color = "tomato" if is_ok else "black"
            ax4.axhline(res["alpha"], color=line_color, lw=1.8, ls="--",
                        label=f"selected α = {res['alpha']:.3f}" + ("" if is_ok else " [unstable]"))
        ax4.set_xlabel(r"$q_{\rm max}$ of window (Å$^{-1}$)", fontsize=10)
        ax4.set_ylabel(r"$\alpha$", fontsize=10)
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No valid windows found",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=9, color="grey")
        ax4.set_axis_off()

    out_path = out_dir / f"{name}_powerlaw.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------
# Section 9: Main analysis pipeline
# --------------------------------------------------------------------

def analyze_folder(
    input_folder:  Path,
    output_folder: Path,
) -> pd.DataFrame:
    """
    Process all SAXS files in input_folder and write:
      •  <output_folder>/powerlaw_results.csv
      •  <output_folder>/<sample>_powerlaw.png  (one per file)

    Returns the results as a pandas DataFrame.
    """
    out_dir = output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Guinier qmax overrides
    guinier_qmax = _load_guinier_qmax(
        GUINIER_CSV if isinstance(GUINIER_CSV, Path) else (
            Path(GUINIER_CSV) if GUINIER_CSV is not None else None
        )
    )
    if guinier_qmax:
        print(f"Loaded Guinier qmax for {len(guinier_qmax)} file(s) from Guinier CSV.\n")

    # Collect and sort files
    files: List[Path] = []
    for ext in FILE_EXTENSIONS:
        files.extend(input_folder.glob(f"*{ext}"))
        files.extend(input_folder.glob(f"*{ext.upper()}"))

    def _natural_key(p: Path) -> list:
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", p.name)
        ]

    files = sorted(set(files), key=_natural_key)

    if not files:
        print(
            f"No files found in {input_folder} with extensions {FILE_EXTENSIONS}.\n"
            "Check INPUT_FOLDER and FILE_EXTENSIONS in USER SETTINGS."
        )
        return pd.DataFrame()

    print(f"Found {len(files)} file(s) in {input_folder}\n")

    rows: List[Dict[str, Any]] = []

    for fpath in files:
        name = fpath.stem
        print(f"  {fpath.name:<60}", end="")

        # --- load ---
        try:
            q, I, err = load_saxs_file(fpath)
        except Exception as exc:
            print(f"  LOAD ERROR: {exc}")
            rows.append({"file": fpath.name, "status": f"load error: {exc}"})
            continue

        # --- per-file q_min from Guinier CSV, or global default ---
        q_min_fit = max(guinier_qmax.get(name.lower(), Q_MIN_FIT), Q_MIN_FIT) #this has been changed based on batch samples result 13-4-2026

        # --- fit ---
        try:
            res = find_powerlaw_region(q, I, err, q_min_fit)
        except Exception as exc:
            print(f"  FIT ERROR: {exc}")
            rows.append({"file": fpath.name, "status": f"fit error: {exc}"})
            continue

        # --- report ---
        if np.isfinite(res["alpha"]):
            two_str = (
                f"  a2={res['alpha2']:.3f}  q_t={res['q_transition']:.4f}"
                if res["two_regime"]
                else ""
            )
            print(
                f"  a = {res['alpha']:.3f} +/- {res['alpha_err']:.3f}"
                f"  q=[{res['qmin']:.4f},{res['qmax']:.4f}]"
                f"  R2={res['r2']:.4f}"
                f"  CV={res['stability_cv']:.3f}"
                f"{two_str}"
                f"  [{res['status']}]"
            )
        else:
            print(f"  [{res['status']}]")

        # --- plots ---
        if SAVE_PLOTS:
            try:
                _save_plots(q, I, err, res, name, out_dir)
            except Exception as exc:
                print(f"    (plot error: {exc})")

        two_r = res.get("_two_regime_result")
        rows.append({
            "file":                     fpath.name,
            "alpha":                    res["alpha"],
            "alpha_err":                res["alpha_err"],
            "alpha1":                   two_r["alpha1"]     if two_r is not None else float("nan"),
            "alpha1_err":               two_r["alpha1_err"] if two_r is not None else float("nan"),
            "alpha2":                   res["alpha2"],
            "alpha2_err":               res["alpha2_err"],
            "q_transition":             res["q_transition"],
            "prefactor":                res["prefactor"],
            "prefactor_err":            res["prefactor_err"],
            "qmin":                     res["qmin"],
            "qmax":                     res["qmax"],
            "n_points":                 res["n_points"],
            "R2":                       res["r2"],
            "chi2_red":                 res["chi2_red"],
            "stability_cv":             res["stability_cv"],
            "two_regime":               res["two_regime"],
            "f_statistic":              two_r["f_statistic"] if two_r is not None else float("nan"),
            "f_pvalue":                 two_r["f_pvalue"]    if two_r is not None else float("nan"),
            "status":                   res["status"],
            "q_min_used":               q_min_fit,
            "candidate_window_count":   res["candidate_window_count"],
            "valid_window_count":       res["valid_window_count"],
        })

    df = pd.DataFrame(rows)

    float_cols = [
        "alpha", "alpha_err", "alpha2", "alpha2_err", "q_transition",
        "prefactor", "prefactor_err", "qmin", "qmax",
        "R2", "chi2_red", "stability_cv", "q_min_used",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(6)

    csv_path = out_dir / "powerlaw_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults -> {csv_path}")
    print(f"Plots   -> {out_dir}/")
    return df


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    df = analyze_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    if not df.empty:
        print("\nSummary of power-law fits:")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 140,
            "display.float_format", "{:.4g}".format,
        ):
            ok_mask = (
                df["status"].str.startswith("ok")
                if "status" in df.columns
                else pd.Series(True, index=df.index)
            )
            print(df[ok_mask].to_string(index=False))
