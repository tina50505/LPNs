from pathlib import Path
import re
import numpy as np
import pandas as pd

# =========================================================
# CHANGE THESE
# =========================================================
FOLDER = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\20260306 CoSAXS\process\Tina ONLINE AF4 SAXS\F3_empty_TRIS_scan-84555_shotsE_subtracted_50frame_blocks")

EXTENSIONS = [".dat", ".txt", ""]

# automatic Guinier settings
MIN_POINTS = 6
MAX_POINTS = 30
QRG_MAX_LIMIT = 1.30       # strict Guinier criterion
LOWQ_FRACTION = 0.25       # only search the lowest 25% of q-range
MIN_R2 = 0.995            # minimum linearity

# output
OUTFILE = FOLDER / "guinier_results.csv"
# =========================================================


def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


def get_all_files(folder, extensions):
    files = []
    for f in folder.iterdir():
        if f.is_file():
            if f.suffix in extensions or (f.suffix == "" and "" in extensions):
                files.append(f)
    return sorted(files, key=natural_key)


def read_saxs_dat(filepath):
    q_vals, I_vals, err_vals = [], [], []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                q = float(parts[0])
                I = float(parts[1])
                err = float(parts[2]) if len(parts) > 2 else np.nan

                q_vals.append(q)
                I_vals.append(I)
                err_vals.append(err)
            except ValueError:
                continue

    q = np.array(q_vals)
    I = np.array(I_vals)
    err = np.array(err_vals)

    # Guinier needs positive intensities
    mask = np.isfinite(q) & np.isfinite(I) & (I > 0)
    return q[mask], I[mask], err[mask]


def weighted_linear_fit(x, y, sigma=None):
    """
    Fit y = m*x + b
    Returns slope, intercept, covariance matrix
    """
    if sigma is not None and np.all(np.isfinite(sigma)) and np.all(sigma > 0):
        # np.polyfit expects weights = 1/sigma
        p, cov = np.polyfit(x, y, 1, w=1.0/sigma, cov=True)
    else:
        p, cov = np.polyfit(x, y, 1, cov=True)

    slope, intercept = p
    return slope, intercept, cov


def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def auto_guinier_one_file(q, I, err,
                          min_points=6,
                          max_points=30,
                          qrg_max_limit=1.30,
                          lowq_fraction=0.25,
                          min_r2=0.995):
    """
    Automatic Guinier range search.

    Returns best fit dict.
    Raises ValueError if no valid window is found.
    """

    if len(q) < min_points:
        raise ValueError("Too few valid data points.")

    # search only the low-q part
    q_limit = q.min() + lowq_fraction * (q.max() - q.min())
    lowq_mask = q <= q_limit

    q_low = q[lowq_mask]
    I_low = I[lowq_mask]
    err_low = err[lowq_mask]

    n = len(q_low)
    if n < min_points:
        raise ValueError("Too few low-q points for Guinier search.")

    best = None

    for start in range(0, n - min_points + 1):
        for end in range(start + min_points, min(start + max_points, n) + 1):
            q_fit = q_low[start:end]
            I_fit = I_low[start:end]
            err_fit = err_low[start:end]

            x = q_fit**2
            y = np.log(I_fit)

            # propagate error: sigma[ln I] = sigma[I]/I
            if np.all(np.isfinite(err_fit)) and np.all(err_fit > 0):
                sigma_y = err_fit / I_fit
            else:
                sigma_y = None

            try:
                slope, intercept, cov = weighted_linear_fit(x, y, sigma=sigma_y)
            except Exception:
                continue

            # slope must be negative in Guinier
            if slope >= 0:
                continue

            Rg = np.sqrt(-3.0 * slope)
            I0 = np.exp(intercept)

            if not np.isfinite(Rg) or not np.isfinite(I0) or I0 <= 0:
                continue

            qRg_min = q_fit.min() * Rg
            qRg_max = q_fit.max() * Rg

            # strict Guinier limit
            if qRg_max > qrg_max_limit:
                continue

            y_pred = slope * x + intercept
            R2 = r_squared(y, y_pred)

            if not np.isfinite(R2) or R2 < min_r2:
                continue

            # errors from covariance
            try:
                slope_err = float(np.sqrt(cov[0, 0]))
                intercept_err = float(np.sqrt(cov[1, 1]))
            except Exception:
                slope_err = np.nan
                intercept_err = np.nan

            # Rg = sqrt(-3*slope)
            # dRg/dslope = -3/(2*Rg)
            if np.isfinite(slope_err) and Rg > 0:
                Rg_err = abs(3.0 / (2.0 * Rg)) * slope_err
            else:
                Rg_err = np.nan

            # prefer:
            # - high R2
            # - more points
            # - qRg_max close to 1.0 but not above 1.3
            score = 5.0 * R2 + 0.03 * len(q_fit) - 0.8 * abs(qRg_max - 1.0)

            result = {
                "qmin": float(q_fit.min()),
                "qmax": float(q_fit.max()),
                "n_points": int(len(q_fit)),
                "Rg_A": float(Rg),
                "Rg_err_A": float(Rg_err),
                "I0": float(I0),
                "I0_err": float(I0 * intercept_err) if np.isfinite(intercept_err) else np.nan,
                "qRg_min": float(qRg_min),
                "qRg_max": float(qRg_max),
                "R2": float(R2),
                "score": float(score),
                "slope": float(slope),
                "intercept": float(intercept),
            }

            if best is None or result["score"] > best["score"]:
                best = result

    if best is None:
        raise ValueError("No valid Guinier window found.")

    return best


# =========================================================
# MAIN
# =========================================================
files = get_all_files(FOLDER, EXTENSIONS)

if len(files) == 0:
    raise ValueError("No files found in the folder.")

results = []

for fp in files:
    try:
        q, I, err = read_saxs_dat(fp)

        best = auto_guinier_one_file(
            q, I, err,
            min_points=MIN_POINTS,
            max_points=MAX_POINTS,
            qrg_max_limit=QRG_MAX_LIMIT,
            lowq_fraction=LOWQ_FRACTION,
            min_r2=MIN_R2
        )

        results.append({
            "file": fp.name,
            "Rg_A": best["Rg_A"],
            "Rg_err_A": best["Rg_err_A"],
            "qmin_A^-1": best["qmin"],
            "qmax_A^-1": best["qmax"],
            "n_points": best["n_points"],
            "qRg_min": best["qRg_min"],
            "qRg_max": best["qRg_max"],
            "I0": best["I0"],
            "I0_err": best["I0_err"],
            "R2": best["R2"],
            "status": "ok"
        })

        print(f"[OK] {fp.name} | Rg = {best['Rg_A']:.2f} ± {best['Rg_err_A']:.2f} Å | "
              f"q = {best['qmin']:.5f} to {best['qmax']:.5f} Å^-1 | qRg_max = {best['qRg_max']:.2f}")

    except Exception as e:
        results.append({
            "file": fp.name,
            "Rg_A": np.nan,
            "Rg_err_A": np.nan,
            "qmin_A^-1": np.nan,
            "qmax_A^-1": np.nan,
            "n_points": np.nan,
            "qRg_min": np.nan,
            "qRg_max": np.nan,
            "I0": np.nan,
            "I0_err": np.nan,
            "R2": np.nan,
            "status": f"failed: {str(e)}"
        })

        print(f"[FAILED] {fp.name} | {e}")

df = pd.DataFrame(results)
df.to_csv(OUTFILE, index=False)

print("\nDone.")
print(f"Saved table to:\n{OUTFILE}")

# optional: print compact summary
print("\nSummary:")
print(df[["file", "Rg_A", "Rg_err_A", "qmin_A^-1", "qmax_A^-1", "qRg_max", "status"]].to_string(index=False))