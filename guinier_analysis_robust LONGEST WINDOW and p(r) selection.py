"""
guinier_analysis_robust.py
==========================
Automatic, physically motivated Guinier region identification and fitting
for SAXS .dat / .txt files.

Physics background
------------------
The Guinier approximation is valid at low q (qRg ≲ 1.3) and states:

    I(q) ≈ I₀ · exp(−Rg² q² / 3)

Taking the natural logarithm linearises the relation:

    ln I(q) = ln I₀  −  (Rg² / 3) · q²

Fitting ln(I) vs q² by linear regression gives:
    slope      = −Rg² / 3   →   Rg = sqrt(−3 · slope)
    intercept  = ln(I₀)     →   I₀ = exp(intercept)

Because the approximation breaks down at high q, the fitting window must be
chosen carefully:  it must lie in the linear portion of the Guinier plot AND
satisfy the validity criterion  qmax · Rg ≤ 1.3.

Automated region search
-----------------------
1.  Restrict the candidate data to the low-q portion of each curve
    (configurable by max. point count and/or absolute q cutoff).
2.  Enumerate every possible window (start_index, window_size) within the
    low-q data, subject to min/max window size limits.
3.  For each window, fit ln(I) vs q² and apply HARD rejection criteria:
      • slope < 0           (Rg must be real)
      • qmax · Rg ≤ QRG_LIMIT  (Guinier validity)
      • R² ≥ MIN_R2        (approximate linearity)
4.  Assess STABILITY:  for every surviving window, collect the Rg values of
    all other surviving windows that overlap with it by ≥ 50 % of the
    shorter window's length.  Compute the coefficient of variation (CV) of
    those Rg values.  A small CV means Rg is insensitive to small boundary
    shifts — a hallmark of a genuine Guinier region.
5.  Select the best window using the longest-stable-window rule:
      • From all windows with CV ≤ MAX_RG_CV, pick the one with the most points.
      • If no window passes the CV threshold, fall back to the window with the
        lowest CV (regardless of size).
6.  Check for pathological low-q behaviour (upturn → possible aggregation).

Usage
-----
1.  Edit the USER SETTINGS block below.
2.  Run:   python guinier_analysis_robust.py
3.  Results appear in OUTPUT_FOLDER as:
      guinier_results.csv          — one row per input file
      <sample>_guinier.png         — 4-panel diagnostic figure per file
"""

import re
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ====================================================================
# USER SETTINGS  — edit here, nowhere else
# ====================================================================

INPUT_FOLDER  = r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS\Tina OFFLINE SAXS"
OUTPUT_FOLDER = str(Path(INPUT_FOLDER) / f"guinier_output_withp(r)_{date.today():%Y-%m-%d}")  # subfolder named by today's date

FILE_EXTENSIONS = [".dat", ".txt"]  # case-insensitive; extend if needed

# ---- Low-q search region -------------------------------------------
# Only the low-q tail of each curve is searched for the Guinier region.
# BOTH limits are applied simultaneously; whichever is more restrictive wins.
MAX_LOW_Q_POINTS  = 60              # use at most this many low-q points
MAX_LOW_Q_CUTOFF  = 0.10            # use only q < this value (Å⁻¹)
                                    # set to None to disable this cutoff

# ---- Fitting window size -------------------------------------------
MIN_WINDOW_POINTS = 5               # minimum data points in a fit window
MAX_WINDOW_POINTS = 30              # maximum data points in a fit window
#Below 5 you can't trust a line fit; above 30 you're likely leaving the Guinier regime.
# ---- Guinier validity criterion ------------------------------------
QRG_LIMIT = 1.3                     # hard upper bound on  qmax · Rg

# ---- Linearity threshold -------------------------------------------
MIN_R2 = 0.990                      # minimum R² (raise to be more selective;
                                    #   good data often yields R² > 0.999)

# ---- Stability criterion -------------------------------------------
# After collecting all valid windows, we require that Rg does not vary
# much when the window boundary shifts by a few points.  We measure this
# as the coefficient of variation (CV = σ/μ) of Rg across overlapping
# valid windows.  Windows with CV > MAX_RG_CV are flagged as unstable.
MAX_RG_CV = 0.08                    # 8 % → accept as stable

# ---- Window start position constraint ------------------------------
# The Guinier region must begin near the lowest available q points.
# A window starting far into the low-q array is fitting the wrong part
# of the curve (transition or Porod regime).
# Hard criterion:  i_start <= MAX_WINDOW_START  (index into low-q array).
# E.g. 10 means the fit window must begin within the first 10 low-q points.
MAX_WINDOW_START = 10               # raise if your data has many noisy first points

# ---- Weighted fitting ----------------------------------------------
# If True and per-point errors are available, weight log-space residuals
# by σ(ln I) = σ(I)/I.  Falls back gracefully to unweighted OLS otherwise.
USE_WEIGHTED_FIT = True

# ---- Plotting ------------------------------------------------------
SAVE_PLOTS    = True
PLOT_DPI      = 150
SAVE_COMBINED = True  # 4-panel summary figure per file

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
            # tolerate European decimal commas
            line = line.replace(",", ".")
            parts = re.split(r"[\s;]+", line) #"0.01  1.234  0.005" becomes ["0.01", "1.234", "0.005"]
            try:
                nums = [float(p) for p in parts if p]
            except ValueError:
                continue                          # header / text line
            if len(nums) >= 2:
                rows.append(nums[:3] if len(nums) >= 3 else nums[:2] + [float("nan")])

    if not rows:
        raise ValueError(f"No numeric data found in {filepath.name}")

    arr = np.array(rows, dtype=float)

    # Sort by q (always explicit, regardless of file order)
    arr = arr[np.argsort(arr[:, 0])]

    q   = arr[:, 0]
    I   = arr[:, 1]
    err = arr[:, 2] if arr.shape[1] >= 3 else None

    # Discard err column if all NaN or all zero
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
    dict with keys: slope, slope_err, intercept, intercept_err, r2
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

    For unweighted OLS:
        var(slope)     = s² / Sxx
        var(intercept) = s² · Σx² / (n · Sxx)
        s²             = Σresiduals² / (n − 2)
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

        # Exact covariance for proper inverse-variance weights
        var_slope     = Sw   / D
        var_intercept = Swx2 / D

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
        # var(intercept) = s² · Σx² / (n · Sxx)  [derived from (XᵀX)⁻¹]
        var_intercept = s2 * float(np.sum(x**2)) / (n * Sxx)

    # R² always computed on raw (unweighted) residuals for interpretability
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
    )


# --------------------------------------------------------------------
# Section 3: Single-window Guinier fit
# --------------------------------------------------------------------

def _fit_window(
    q_win:   np.ndarray,
    I_win:   np.ndarray,
    err_win: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """
    Fit the Guinier relation for a pre-selected data window.

    Steps
    -----
    1.  Keep only finite, strictly positive (q, I) pairs.
    2.  Transform to log-space:  y = ln(I),  x = q².
    3.  Propagate uncertainties:  σ(ln I) ≈ σ(I)/I  (first-order Taylor).
    4.  Run weighted or unweighted linear regression.
    5.  Derive Rg and I₀ with formal uncertainties.
    6.  Apply physical validity checks (slope < 0, Rg finite > 0, I₀ > 0).

    Returns a dict of fit diagnostics, or None if the window is invalid.
    """
    # --- validity mask ---
    mask = np.isfinite(q_win) & (q_win > 0) & np.isfinite(I_win) & (I_win > 0)
    if err_win is not None:
        mask &= np.isfinite(err_win) & (err_win > 0)
    if int(mask.sum()) < 3:
        return None
#Guinier transformation: x = q², y = ln I. Only keep valid points for fitting.
    q_  = q_win[mask]
    I_  = I_win[mask]
    x   = q_**2
    y   = np.log(I_)

    # --- weights in log-space ---
    w        = None
    weighted = False
    if USE_WEIGHTED_FIT and err_win is not None:
        sigma_lnI = err_win[mask] / I_
        if np.all(sigma_lnI > 0) and np.all(np.isfinite(sigma_lnI)):
            w        = 1.0 / sigma_lnI**2
            weighted = True

    fit = _linfit(x, y, w)
    if fit is None:
        return None

    slope     = fit["slope"]
    intercept = fit["intercept"]

    # --- physical checks ---
    if slope >= 0.0:
        return None           # positive slope → not a Guinier-like decay

    Rg_sq = -3.0 * slope
    if Rg_sq <= 0.0 or not np.isfinite(Rg_sq):
        return None

    Rg = float(np.sqrt(Rg_sq))
    I0 = float(np.exp(intercept))

    if not (np.isfinite(Rg) and Rg > 0.0 and np.isfinite(I0) and I0 > 0.0):
        return None

    # --- error propagation ---
    # Rg = sqrt(−3·slope)   →   σ_Rg = (3 / (2·Rg)) · σ_slope
    Rg_err = (3.0 / (2.0 * Rg)) * fit["slope_err"]
    # I₀ = exp(intercept)   →   σ_I₀ = I₀ · σ_intercept
    I0_err = I0 * fit["intercept_err"]

    # --- reduced chi-square (only with proper weights) ---
    chi2_red = float("nan")
    if weighted and w is not None:
        resid    = y - (slope * x + intercept)
        chi2     = float(np.sum(resid**2 * w))   # Σ (resid/σ)²
        chi2_red = chi2 / (int(mask.sum()) - 2)

    n_pts = int(mask.sum()) 
    #Returns everything about this window as a dictionary. qRg_max = qmax·Rg is the Guinier validity diagnostic — the hard criterion qRg_max ≤ 1.3 is applied later in the search loop.
    return dict(
        slope         = slope,
        slope_err     = fit["slope_err"],
        intercept     = intercept,
        intercept_err = fit["intercept_err"],
        Rg            = Rg,
        Rg_err        = Rg_err,
        I0            = I0,
        I0_err        = I0_err,
        r2            = fit["r2"],
        chi2_red      = chi2_red,
        weighted      = weighted,
        n_points      = n_pts,
        qmin          = float(q_.min()),
        qmax          = float(q_.max()),
        qRg_min       = float(q_.min() * Rg),
        qRg_max       = float(q_.max() * Rg),
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

    Among all windows with CV ≤ MAX_RG_CV, pick the one with the most points.
    If no window passes the CV threshold, fall back to the window with the
    lowest CV (regardless of size).
    """
    stable_indices = [i for i, cv in enumerate(cvs) if cv <= MAX_RG_CV] #Goes through all CV values and collects the indices of windows that are stable (CV ≤ 8%).
    if stable_indices:
        return max(stable_indices, key=lambda i: valid_windows[i]["n_points"]) #If any stable windows exist, return the index of the one with the most points.
    # fallback: lowest CV
    return int(np.argmin(cvs))


# --------------------------------------------------------------------
# Section 5: Automatic Guinier region search
# --------------------------------------------------------------------

def find_guinier_region(
    q:   np.ndarray,
    I:   np.ndarray,
    err: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Identify the best Guinier region in the low-q part of a SAXS curve.

    See module docstring for the full algorithm description.

    Returns a flat dict containing:
      status        — 'ok' or a descriptive failure/warning string
      Rg, Rg_err    — radius of gyration and its uncertainty (Å)
      I0, I0_err    — forward scattering and its uncertainty
      qmin, qmax    — q range of the selected window (Å⁻¹)
      n_points      — number of points in the selected window
      qRg_min/max   — Guinier validity diagnostic
      r2            — R² of the selected window
      chi2_red      — reduced chi-square (NaN if unweighted)
      stability_cv  — CV of Rg across overlapping valid windows
      slope/intercept and their errors
      _q_low/_I_low/_err_low — low-q arrays (for plotting; prefix _ = internal)
      valid_windows — list of all windows passing hard criteria (for plotting)
    """
    nan_result: Dict[str, Any] = dict(
        status                 = "no valid region found",
        Rg                     = float("nan"),
        Rg_err                 = float("nan"),
        I0                     = float("nan"),
        I0_err                 = float("nan"),
        qmin                   = float("nan"),
        qmax                   = float("nan"),
        n_points               = float("nan"),
        qRg_min                = float("nan"),
        qRg_max                = float("nan"),
        r2                     = float("nan"),
        chi2_red               = float("nan"),
        slope                  = float("nan"),
        slope_err              = float("nan"),
        intercept              = float("nan"),
        intercept_err          = float("nan"),
        stability_cv           = float("nan"),
        idx_start              = None,
        idx_end                = None,
        valid_windows          = [],
        _q_low                 = np.array([]),
        _I_low                 = np.array([]),
        _err_low               = None,
        # --- diagnostic counters (populated below) ---
        candidate_window_count = 0,
        valid_window_count     = 0,
    )

    # ----------------------------------------------------------------
    # Step 1 — restrict to finite, positive, sorted low-q data
    # ----------------------------------------------------------------
    mask_pos = np.isfinite(q) & (q > 0) & np.isfinite(I) & (I > 0)
    if int(mask_pos.sum()) < MIN_WINDOW_POINTS:
        nan_result["status"] = "too few points with positive intensity"
        return nan_result

    q_  = q[mask_pos]
    I_  = I[mask_pos]
    e_  = err[mask_pos] if err is not None else None

    # Re-sort (data from load_saxs_file are already sorted, but be safe)
    order = np.argsort(q_)
    q_ = q_[order];  I_ = I_[order]
    if e_ is not None:
        e_ = e_[order]

    # Apply low-q restrictions (both point count and q cutoff)
    n_low = len(q_)
    if MAX_LOW_Q_CUTOFF is not None:
        n_low = min(n_low, int(np.searchsorted(q_, MAX_LOW_Q_CUTOFF, side="right")))
    n_low = min(n_low, MAX_LOW_Q_POINTS)

    if n_low < MIN_WINDOW_POINTS:
        nan_result["status"] = f"too few low-q points (found {n_low})"
        nan_result["_q_low"] = q_[:n_low]
        nan_result["_I_low"] = I_[:n_low]
        nan_result["_err_low"] = e_[:n_low] if e_ is not None else None
        return nan_result

    q_low   = q_[:n_low]
    I_low   = I_[:n_low]
    err_low = e_[:n_low] if e_ is not None else None

    nan_result["_q_low"]   = q_low
    nan_result["_I_low"]   = I_low
    nan_result["_err_low"] = err_low

    # ----------------------------------------------------------------
    # Step 2+3 — enumerate windows and apply hard criteria
    # ----------------------------------------------------------------
    valid_windows: List[Dict[str, Any]] = []
    candidate_window_count = 0   # windows with physically plausible neg slope
    valid_window_count     = 0   # windows passing ALL hard criteria

    for i_start in range(n_low - MIN_WINDOW_POINTS + 1):
        for i_end in range(
            i_start + MIN_WINDOW_POINTS,
            min(i_start + MAX_WINDOW_POINTS + 1, n_low + 1),
        ):
            res = _fit_window(
                q_low[i_start:i_end],
                I_low[i_start:i_end],
                err_low[i_start:i_end] if err_low is not None else None,
            )
            if res is None:
                continue

            # Any window with a negative slope is a physical candidate
            candidate_window_count += 1

            # Hard criteria: Guinier validity + linearity + low-q start
            if res["qRg_max"] > QRG_LIMIT:
                continue
            if res["r2"] < MIN_R2:
                continue
            # The window must start near the beginning of the low-q data.
            # This prevents the algorithm from fitting a coincidentally linear
            # stretch in the transition or Porod regime further up the curve.
            if i_start > MAX_WINDOW_START:
                continue

            valid_window_count += 1
            res["i_start"] = i_start
            res["i_end"]   = i_end   # exclusive upper bound
            valid_windows.append(res)

    if not valid_windows:
        # Diagnose the reason for failure
        status = _diagnose_failure(q_low, I_low, err_low)
        nan_result["status"]                 = status
        nan_result["candidate_window_count"] = candidate_window_count
        nan_result["valid_window_count"]     = valid_window_count
        return nan_result

    # ----------------------------------------------------------------
    # Step 4 — stability assessment
    # Collect Rg values from all valid windows that overlap this one
    # by at least 50 % of the shorter window's length.
    # A low CV of those Rg values means the Guinier region is stable
    # against small changes in the fitting interval.
    # ----------------------------------------------------------------
    cvs: List[float] = []
    for k, wk in enumerate(valid_windows):
        sk, ek = wk["i_start"], wk["i_end"]
        len_k  = ek - sk
        neighbor_Rgs = [wk["Rg"]]          # include the window itself

        for j, wj in enumerate(valid_windows):
            if j == k:
                continue
            sj, ej = wj["i_start"], wj["i_end"]
            overlap = max(0, min(ek, ej) - max(sk, sj))
            min_len = min(len_k, ej - sj)
            if min_len > 0 and overlap >= 0.5 * min_len:
                neighbor_Rgs.append(wj["Rg"])

        arr_rg = np.array(neighbor_Rgs)
        mean_rg = float(arr_rg.mean())
        cv = float(arr_rg.std() / mean_rg) if mean_rg > 0 else float("inf")
        cvs.append(cv)

    # ----------------------------------------------------------------
    # Step 5 — select the best window (longest stable window)
    # ----------------------------------------------------------------
    best_idx = _select_best_window(valid_windows, cvs)
    best     = valid_windows[best_idx]
    best_cv  = cvs[best_idx]

    # ----------------------------------------------------------------
    # Step 6 — build result dict
    # ----------------------------------------------------------------
    if best_cv > MAX_RG_CV:
        status = (
            f"unstable Guinier fit (CV={best_cv:.2f}); result may be unreliable"
        )
    else:
        status = "ok"

    # Check for low-q upturn: positive slope in the very first few points
    # suggests inter-particle repulsion, aggregation, or beam artefacts.
    n_check = min(max(MIN_WINDOW_POINTS, 5), n_low)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnI_check = np.log(I_low[:n_check])
    x_check = q_low[:n_check]**2
    if np.all(np.isfinite(lnI_check)) and len(lnI_check) >= 2:
        slope_check = float(np.polyfit(x_check, lnI_check, 1)[0])
        if slope_check > 0.0 and status == "ok":
            status = "ok (possible low-q upturn — check for aggregation)"

    result = dict(
        nan_result,             # copy defaults
        status                 = status,
        Rg                     = best["Rg"],
        Rg_err                 = best["Rg_err"],
        I0                     = best["I0"],
        I0_err                 = best["I0_err"],
        qmin                   = best["qmin"],
        qmax                   = best["qmax"],
        n_points               = best["n_points"],
        qRg_min                = best["qRg_min"],
        qRg_max                = best["qRg_max"],
        r2                     = best["r2"],
        chi2_red      = best["chi2_red"],
        slope         = best["slope"],
        slope_err     = best["slope_err"],
        intercept     = best["intercept"],
        intercept_err = best["intercept_err"],
        stability_cv           = best_cv,
        idx_start              = best["i_start"],
        idx_end                = best["i_end"],
        valid_windows          = valid_windows,
        _q_low                 = q_low,
        _I_low                 = I_low,
        _err_low               = err_low,
        candidate_window_count = candidate_window_count,
        valid_window_count     = valid_window_count,
    )
    return result


# No negative slope at all → data not in Guinier regime
# No window with qRg ≤ 1.3 → Rg too large for your q range (need lower q)
# No window with R² ≥ MIN_R2 → data too curved/noisy
# No window starting early enough → all linear stretches are too far from q=0
# All flags True but still failed → the criteria are individually satisfiable but not simultaneously    
def _diagnose_failure(
    q_low:   np.ndarray,
    I_low:   np.ndarray,
    err_low: Optional[np.ndarray],
) -> str:
    """
    When no valid window is found, probe the rejection criteria individually
    to return the most informative failure message.
    """
    any_neg_slope = any_qrg_ok = any_r2_ok = any_start_ok = False

    for i_start in range(len(q_low) - MIN_WINDOW_POINTS + 1):
        for i_end in range(
            i_start + MIN_WINDOW_POINTS,
            min(i_start + MAX_WINDOW_POINTS + 1, len(q_low) + 1),
        ):
            res = _fit_window(
                q_low[i_start:i_end],
                I_low[i_start:i_end],
                err_low[i_start:i_end] if err_low is not None else None,
            )
            if res is None:
                continue
            any_neg_slope = True
            if res["qRg_max"] <= QRG_LIMIT:
                any_qrg_ok = True
            if res["r2"] >= MIN_R2:
                any_r2_ok = True
            if i_start <= MAX_WINDOW_START:
                any_start_ok = True

    if not any_neg_slope:
        return "no window with negative Guinier slope — data may not be in Guinier regime"
    if not any_qrg_ok:
        return "no window satisfying qmax·Rg ≤ 1.3 — Rg may be too large for accessible q range"
    if not any_r2_ok:
        return "no sufficiently linear Guinier window found (R² too low — check for aggregation or non-Guinier shape)"
    if not any_start_ok:
        return "no valid Guinier window found starting within the first low-q points"
    return "no window satisfying all criteria simultaneously"


# --------------------------------------------------------------------
# Section 5b: Diagnostic classification (added on top of existing logic)
# --------------------------------------------------------------------

def _check_lowq_upturn_detailed(res: Dict[str, Any]) -> bool:
    """
    Refined low-q upturn / aggregation detection.

    When a Guinier fit exists:
      • Take the low-q points that lie BEFORE the fit window (pre-window points).
      • Compare their ln(I) to the extrapolated Guinier line at the same q².
      • If at least 3 of those points (or ≥ 60 % of them when fewer than 5 are
        available) sit ABOVE the fit line by more than a small threshold (0.05 in
        ln-space ≈ 5 % in intensity), flag a low-q upturn.
      • If there are no pre-window points, check whether the first 3 points of
        the fit window itself show systematically positive residuals.

    When no Guinier fit exists:
      • Fit a simple line to the first n_check low-q points in Guinier coordinates
        (ln I vs q²).  A positive slope indicates the intensity rises as q
        increases — the hallmark of a low-q upturn.

    Returns True (flag raised) or False (no upturn detected).
    This is deliberately conservative: we require consistent systematic deviation,
    not just a single outlier point.
    """
    q_low  = res.get("_q_low",  np.array([]))
    I_low  = res.get("_I_low",  np.array([]))
    i_start = res.get("idx_start")
    slope   = res.get("slope",  float("nan"))
    intercept = res.get("intercept", float("nan"))
    fit_exists = (
        i_start is not None
        and np.isfinite(slope)
        and np.isfinite(intercept)
    )

    if fit_exists:
        # --- case A: compare pre-window points to the extrapolated fit ---
        n_pre = i_start  # number of points before the fit window
        if n_pre >= 3:
            q_pre  = q_low[:n_pre]
            I_pre  = I_low[:n_pre]
            valid  = (I_pre > 0) & np.isfinite(I_pre)
            if int(valid.sum()) >= 3:
                q_pre_v  = q_pre[valid]
                lnI_pre  = np.log(I_pre[valid])
                lnI_fit  = slope * q_pre_v**2 + intercept
                resid_pre = lnI_pre - lnI_fit          # positive = above fit
                n_above  = int(np.sum(resid_pre > 0.05))
                frac_above = n_above / len(resid_pre)
                # Flag if ≥60 % of pre-window points are above the fit line
                if frac_above >= 0.60 and n_above >= 3:
                    return True

        # --- case B: no pre-window points — check first 3 fit-window residuals ---
        else:
            i_end = res.get("idx_end", i_start)
            n_win = (i_end or i_start) - i_start
            n_check_inner = min(3, n_win)
            if n_check_inner >= 2:
                q_inner  = q_low[i_start : i_start + n_check_inner]
                I_inner  = I_low[i_start : i_start + n_check_inner]
                valid    = (I_inner > 0) & np.isfinite(I_inner)
                if int(valid.sum()) >= 2:
                    lnI_inner = np.log(I_inner[valid])
                    lnI_fit   = slope * q_inner[valid]**2 + intercept
                    resid_in  = lnI_inner - lnI_fit
                    # Flag if all first-window points are above the fit
                    if np.all(resid_in > 0.05):
                        return True
        return False

    else:
        # --- no fit: check slope of first few low-q points in Guinier coords ---
        n_check = min(max(MIN_WINDOW_POINTS, 5), len(q_low))
        if n_check < 3:
            return False
        I_check = I_low[:n_check]
        q_check = q_low[:n_check]
        valid = (I_check > 0) & np.isfinite(I_check) & np.isfinite(q_check)
        if int(valid.sum()) < 3:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lnI_check = np.log(I_check[valid])
        x_check = q_check[valid]**2
        slope_check = float(np.polyfit(x_check, lnI_check, 1)[0])
        return slope_check > 0.0


def _check_residual_trend(res: Dict[str, Any]) -> bool:
    """
    Check whether the Guinier fit residuals show a systematic curvature.

    Method: split the residuals into a lower half and upper half (by q²).
    Compute the mean residual in each half.  If the means have opposite signs
    AND the difference exceeds a threshold (0.03 in ln-space), a U- or arch-
    shaped pattern is likely — indicating the data curve more steeply or more
    shallowly than the fitted line.

    This is a minimal curvature proxy that avoids scipy dependencies.
    Returns True if a systematic trend is detected.
    """
    i_start = res.get("idx_start")
    i_end   = res.get("idx_end")
    slope   = res.get("slope",     float("nan"))
    intercept = res.get("intercept", float("nan"))
    q_low   = res.get("_q_low", np.array([]))
    I_low   = res.get("_I_low", np.array([]))

    if i_start is None or i_end is None:
        return False
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return False

    q_win = q_low[i_start:i_end]
    I_win = I_low[i_start:i_end]
    valid = (I_win > 0) & np.isfinite(I_win)
    if int(valid.sum()) < 6:
        return False   # too few points to split meaningfully

    q_v   = q_win[valid]
    lnI_v = np.log(I_win[valid])
    resid = lnI_v - (slope * q_v**2 + intercept)

    mid   = len(resid) // 2
    mean_lo = float(np.mean(resid[:mid]))
    mean_hi = float(np.mean(resid[mid:]))

    # Systematic curvature: halves have opposite signs with meaningful magnitude
    opposite_signs = (mean_lo > 0) != (mean_hi > 0)
    large_enough   = abs(mean_lo - mean_hi) > 0.03
    return bool(opposite_signs and large_enough)


def _assign_diagnosis(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the free-text status + quantitative metrics from find_guinier_region
    into a structured diagnosis code and ancillary flags.

    Returns a dict with keys:
      diagnosis            — one of the canonical codes below
      failure_reason       — short human-readable string (empty if ok)
      lowq_upturn_flag     — bool: low-q upturn / aggregation suspected
      residual_trend_flag  — bool: systematic curvature in fit residuals
      no_accessible_guinier_flag — bool: q-range too high-q for Rg

    Diagnosis codes
    ---------------
    ok                                — good Guinier fit
    ok_possible_lowq_upturn           — good fit but upturn detected below window
    fail_no_accessible_guinier        — Rg too large for accessible q range, or
                                        no negative-slope window at all
    fail_lowq_upturn_or_aggregation   — upturn dominates, no clean Guinier regime
    fail_unstable_guinier             — windows exist but Rg varies too much
    fail_poor_linearity               — windows exist but R² / curvature too bad
    fail_too_noisy_or_insufficient_points — too few usable low-q points
    fail_no_window_satisfying_all_criteria — catch-all simultaneous failure
    """
    status   = res.get("status", "")
    is_ok    = status == "ok" or status.startswith("ok (")
    cand_cnt = res.get("candidate_window_count", 0)

    # ---- refined upturn check ----
    lowq_flag   = _check_lowq_upturn_detailed(res)
    resid_flag  = _check_residual_trend(res)

    # ---- no-accessible-guinier flag ----
    # Either: no window satisfies qRg ≤ 1.3 (Rg too large for q range)
    # Or:     no window has a negative slope at all (no Guinier decay visible)
    no_access_flag = (
        "qmax·Rg ≤ 1.3" in status
        or "too large for accessible q range" in status
        or ("no window with negative" in status and cand_cnt == 0)
    )

    failure_reason = ""

    if is_ok:
        if lowq_flag or "low-q upturn" in status:
            diagnosis = "ok_possible_lowq_upturn"
        else:
            diagnosis = "ok"

    elif "too few" in status or "insufficient" in status:
        diagnosis      = "fail_too_noisy_or_insufficient_points"
        failure_reason = "Too few valid low-q points for fitting"

    elif no_access_flag:
        diagnosis      = "fail_no_accessible_guinier"
        failure_reason = "No window satisfies qmax·Rg ≤ 1.3 — particle likely too large for q range"

    elif "R² too low" in status or "linearity" in status.lower() or "non-Guinier" in status:
        if lowq_flag:
            diagnosis      = "fail_lowq_upturn_or_aggregation"
            failure_reason = "Poor linearity combined with low-q upturn — likely aggregation"
        else:
            diagnosis      = "fail_poor_linearity"
            failure_reason = "Candidate windows exist but R² is too low (non-linear Guinier plot)"

    elif "unstable" in status:
        diagnosis      = "fail_unstable_guinier"
        failure_reason = f"Rg varies too much across overlapping windows (CV = {res.get('stability_cv', float('nan')):.2f})"

    elif "no valid Guinier window found starting" in status:
        # All linear stretches are too far from q=0; might be dominated by
        # transitions or power-law at the very lowest q (possible upturn)
        if lowq_flag:
            diagnosis      = "fail_lowq_upturn_or_aggregation"
            failure_reason = "No Guinier window near q=0; low-q upturn detected"
        else:
            diagnosis      = "fail_no_window_satisfying_all_criteria"
            failure_reason = "No Guinier window starts within MAX_WINDOW_START points from q=0"

    else:
        # Catch-all: "no window satisfying all criteria simultaneously"
        if lowq_flag:
            diagnosis      = "fail_lowq_upturn_or_aggregation"
            failure_reason = "Low-q upturn prevents simultaneous satisfaction of all Guinier criteria"
        else:
            diagnosis      = "fail_no_window_satisfying_all_criteria"
            failure_reason = status   # preserve the original message

    return dict(
        diagnosis                  = diagnosis,
        failure_reason             = failure_reason,
        lowq_upturn_flag           = lowq_flag,
        residual_trend_flag        = resid_flag,
        no_accessible_guinier_flag = no_access_flag,
    )


def _assign_next_step(diagnosis: str, lowq_upturn_flag: bool) -> str:
    """
    Recommend the most appropriate follow-up analysis based on the Guinier
    diagnosis and the low-q upturn flag.

    Decision rules
    --------------
    ok                                → p(r)
        A clean Guinier fit is the green light for IFT / p(r) analysis.

    ok_possible_lowq_upturn           → inspect_then_pr
        The Guinier result is numerically good, but the low-q upturn hint means
        the user should verify the p(r) output does not show artefacts at large r.

    fail_no_accessible_guinier        → inspect_then_pr
        The particle may simply be too large for the q range, but the full
        scattering curve could still yield a p(r) — worth trying with caution.

    fail_lowq_upturn_or_aggregation   → inspect_then_powerlaw
        Likely aggregation or inter-particle effects dominate the low-q region;
        power-law analysis on the mid-/high-q range may be more informative.

    fail_unstable_guinier             → inspect_then_pr
        Some candidate windows exist but Rg is ill-defined; the overall curve
        shape could still support a p(r) with careful truncation.

    fail_poor_linearity               → inspect_then_powerlaw
        The Guinier plot is not linear — the sample may be polydisperse, have
        a complex shape, or show power-law/fractal scattering.

    fail_too_noisy_or_insufficient_points → qualitative_only
        Data quality is too low for quantitative analysis.

    fail_no_window_satisfying_all_criteria → inspect_then_pr
        Ambiguous failure; inspect the curve manually before deciding.

    If lowq_upturn_flag is True and the result would otherwise be p(r),
    we downgrade to inspect_then_pr (cautious).
    """
    mapping = {
        "ok":                                    "p(r)",
        "ok_possible_lowq_upturn":               "inspect_then_pr",
        "fail_no_accessible_guinier":            "inspect_then_pr",
        "fail_lowq_upturn_or_aggregation":       "inspect_then_powerlaw",
        "fail_unstable_guinier":                 "inspect_then_pr",
        "fail_poor_linearity":                   "inspect_then_powerlaw",
        "fail_too_noisy_or_insufficient_points": "qualitative_only",
        "fail_no_window_satisfying_all_criteria":"inspect_then_pr",
    }
    step = mapping.get(diagnosis, "qualitative_only")

    # Conservative downgrade: if a clean "ok" has an upturn flag, add caution
    if step == "p(r)" and lowq_upturn_flag:
        step = "inspect_then_pr"

    return step


# --------------------------------------------------------------------
# Section 6: Diagnostic plots
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

    Panel 1 — SAXS curve (log-log):  I(q) vs q,  Guinier region highlighted
    Panel 2 — Guinier plot:           ln I vs q²,  fit line + annotations
    Panel 3 — Guinier fit residuals
    Panel 4 — Rg scatter across all valid windows (stability overview)
    """
    is_ok    = res["status"] == "ok" or res["status"].startswith("ok (")
    q_low    = res["_q_low"]
    I_low    = res["_I_low"]
    err_low  = res["_err_low"]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Guinier Analysis — {name}", fontsize=13, fontweight="bold")

    # ── Panel 1: SAXS curve ─────────────────────────────────────────
    ax1.set_title("SAXS curve", fontsize=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpos = (I_full > 0) & np.isfinite(I_full) & np.isfinite(q_full) & (q_full > 0)

    ax1.loglog(q_full[mpos], I_full[mpos],
               ".", color="steelblue", ms=3, alpha=0.7, label="data")

    if is_ok and res["idx_start"] is not None:
        i0, i1 = res["idx_start"], res["idx_end"]
        ax1.loglog(q_low[i0:i1], I_low[i0:i1],
                   "o", color="tomato", ms=5, zorder=5, label="Guinier region")

    ax1.set_xlabel(r"$q\ (\mathrm{\AA}^{-1})$", fontsize=10)
    ax1.set_ylabel(r"$I(q)$ (a.u.)", fontsize=10)
    ax1.legend(fontsize=8)

    # ── Panel 2: Guinier plot ────────────────────────────────────────
    ax2.set_title("Guinier plot", fontsize=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnI_low = np.log(np.where(I_low > 0, I_low, np.nan))
    q2_low  = q_low**2

    ax2.plot(q2_low, lnI_low, ".", color="steelblue", ms=4, alpha=0.7,
             label="low-q data")

    if is_ok and res["idx_start"] is not None:
        i0, i1 = res["idx_start"], res["idx_end"]
        q2_g   = q_low[i0:i1]**2
        lnI_g  = np.log(I_low[i0:i1])

        ax2.plot(q2_g, lnI_g, "o", color="tomato", ms=5, zorder=5,
                 label="fit window")

        x_line = np.linspace(q2_g[0] * 0.95, q2_g[-1] * 1.05, 200)
        y_line = res["slope"] * x_line + res["intercept"]
        ax2.plot(x_line, y_line, "-", color="darkred", lw=1.8, label="fit")

        chi2_str = (
            f"\n$\\chi^2_r$ = {res['chi2_red']:.2f}"
            if np.isfinite(res["chi2_red"])
            else ""
        )
        ann = (
            f"$R_g$ = {res['Rg']:.2f} ± {res['Rg_err']:.2f} Å\n"
            f"$I_0$ = {res['I0']:.3g} ± {res['I0_err']:.1g}\n"
            f"$qR_g^{{\\max}}$ = {res['qRg_max']:.3f}\n"
            f"$R^2$ = {res['r2']:.4f}"
            f"{chi2_str}"
        )
        ax2.text(0.97, 0.97, ann, transform=ax2.transAxes,
                 ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax2.set_xlabel(r"$q^2\ (\mathrm{\AA}^{-2})$", fontsize=10)
    ax2.set_ylabel(r"$\ln\,I(q)$", fontsize=10)
    ax2.legend(fontsize=8)

    # ── Panel 3: residuals ──────────────────────────────────────────
    ax3.set_title("Guinier fit residuals", fontsize=11)
    if is_ok and res["idx_start"] is not None:
        i0, i1 = res["idx_start"], res["idx_end"]
        q2_g   = q_low[i0:i1]**2
        lnI_g  = np.log(I_low[i0:i1])
        resid  = lnI_g - (res["slope"] * q2_g + res["intercept"])

        ax3.axhline(0, color="k", lw=0.9, ls="--", alpha=0.6)
        if err_low is not None:
            sigma_lnI = err_low[i0:i1] / I_low[i0:i1]
            ax3.errorbar(q2_g, resid, yerr=sigma_lnI,
                         fmt="o", color="tomato", ms=5, capsize=3,
                         elinewidth=0.8)
        else:
            ax3.plot(q2_g, resid, "o", color="tomato", ms=5)

        # Highlight systematic trends by a smoothed guide (≥6 points)
        if len(resid) >= 6:
            ax3.plot(q2_g, resid, "-", color="tomato", alpha=0.35, lw=1)

        ax3.set_xlabel(r"$q^2\ (\mathrm{\AA}^{-2})$", fontsize=10)
        ax3.set_ylabel(r"$\ln I_{\rm obs} - \ln I_{\rm fit}$", fontsize=10)
    else:
        ax3.text(0.5, 0.5, f"No valid fit\n\n{res['status']}",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=9, color="grey")
        ax3.set_axis_off()

    # ── Panel 4: Rg stability across all valid windows ──────────────
    ax4.set_title("$R_g$ across valid windows", fontsize=11)
    vws = res.get("valid_windows", [])
    if vws:
        Rg_all   = np.array([w["Rg"]   for w in vws])
        qmax_all = np.array([w["qmax"] for w in vws])
        ax4.scatter(qmax_all, Rg_all, c="steelblue", s=14, alpha=0.45,
                    label="valid windows")
        if is_ok:
            ax4.axhline(res["Rg"], color="tomato", lw=1.8, ls="--",
                        label=f"selected $R_g$ = {res['Rg']:.2f} Å")
        ax4.set_xlabel(r"$q_{\rm max}$ of window (Å$^{-1}$)", fontsize=10)
        ax4.set_ylabel(r"$R_g$ (Å)", fontsize=10)
        ax4.legend(fontsize=8)
        # If Rg is fairly constant, the scatter should be a horizontal band —
        # large vertical spread here signals genuine instability.
    else:
        ax4.text(0.5, 0.5, "No valid windows found",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=9, color="grey")
        ax4.set_axis_off()

    out_path = out_dir / f"{name}_guinier.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------
# Section 7: Main analysis pipeline
# --------------------------------------------------------------------

def analyze_folder(
    input_folder:  str,
    output_folder: str,
) -> pd.DataFrame:
    """
    Process all SAXS files in input_folder and write:
      •  <output_folder>/guinier_results.csv
      •  <output_folder>/<sample>_guinier.png  (one per file)

    Returns the results as a pandas DataFrame.
    """
    in_dir  = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files (case-insensitive, deduplicated)
    files: List[Path] = []
    for ext in FILE_EXTENSIONS:
        files.extend(in_dir.glob(f"*{ext}"))
        files.extend(in_dir.glob(f"*{ext.upper()}"))

    # Sort in natural order so frame numbers come out right
    def _natural_key(p: Path) -> list:
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", p.name)
        ]

    files = sorted(set(files), key=_natural_key)

    if not files:
        print(
            f"No files found in {in_dir} with extensions {FILE_EXTENSIONS}.\n"
            "Check INPUT_FOLDER and FILE_EXTENSIONS in USER SETTINGS."
        )
        return pd.DataFrame()

    print(f"Found {len(files)} file(s) in {in_dir}\n")

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

        # --- fit ---
        try:
            res = find_guinier_region(q, I, err)
        except Exception as exc:
            print(f"  FIT ERROR: {exc}")
            rows.append({"file": fpath.name, "status": f"fit error: {exc}"})
            continue

        # --- report ---
        if np.isfinite(res["Rg"]):
            print(
                f"  Rg = {res['Rg']:6.2f} ± {res['Rg_err']:.2f} Å"
                f"  qRg_max = {res['qRg_max']:.3f}"
                f"  R² = {res['r2']:.4f}"
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

        # --- diagnostic classification (new layer, does not alter fit results) ---
        diag = _assign_diagnosis(res)
        next_step = _assign_next_step(
            diag["diagnosis"], diag["lowq_upturn_flag"]
        )

        rows.append({
            "file":                     fpath.name,
            "Rg":                       res["Rg"],
            "Rg_err":                   res["Rg_err"],
            "I0":                       res["I0"],
            "I0_err":                   res["I0_err"],
            "qmin":                     res["qmin"],
            "qmax":                     res["qmax"],
            "n_points":                 res["n_points"],
            "qRg_min":                  res["qRg_min"],
            "qRg_max":                  res["qRg_max"],
            "R2":                       res["r2"],
            "chi2_red":                 res["chi2_red"],
            "stability_cv":             res.get("stability_cv", float("nan")),
            "status":                   res["status"],
            # --- new diagnostic columns ---
            "diagnosis":                diag["diagnosis"],
            "next_step":                next_step,
            "lowq_upturn_flag":         diag["lowq_upturn_flag"],
            "residual_trend_flag":      diag["residual_trend_flag"],
            "no_accessible_guinier_flag": diag["no_accessible_guinier_flag"],
            "failure_reason":           diag["failure_reason"],
            "candidate_window_count":   res.get("candidate_window_count", 0),
            "valid_window_count":       res.get("valid_window_count", 0),
        })

    df = pd.DataFrame(rows)

    # Round float columns for readability
    float_cols = ["Rg", "Rg_err", "I0", "I0_err",
                  "qmin", "qmax", "qRg_min", "qRg_max",
                  "R2", "chi2_red", "stability_cv"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(6)

    csv_path = out_dir / "guinier_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")
    print(f"Plots   → {out_dir}/")
    return df


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    df = analyze_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    if not df.empty:
        print("\nSummary of Guinier fits:")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 120,
            "display.float_format", "{:.4g}".format,
        ):
            ok_mask = df["status"].str.startswith("ok") if "status" in df.columns else pd.Series(True, index=df.index)
            print(df[ok_mask].to_string(index=False))
