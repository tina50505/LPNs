from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# USER SETTINGS
# =========================================================
FILES = [
    Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\Ribogreen on fractions\TE_TX_combined_F3_F3C.xlsx"),
    Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\Ribogreen on fractions\Test_plate_TE_TX_F3_F3C.xlsx"),
    Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\Ribogreen on fractions\Another_F3_TE_only.xlsx"),
    Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\Ribogreen on fractions\Another_F3_TX_only.xlsx"),
]

SHEET = "Data_for_Python"
OUTDIR = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\Ribogreen on fractions\ribogreen_clean_output7")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Fit settings
FIT_INCLUDE_BLANK = True
AVERAGE_STANDARDS_BY_CONC = True
MIN_REPLICATE_R2_FOR_AVG = 0.99
SATURATION_RFU = 260000
CLIP_NEGATIVE_CONC_TO_ZERO = True

# Plot settings
ANNOTATE_SAMPLES = True
SAVE_PLOTS = True
SHOW_PLOTS = False

# EE plot: exclude fractions where EE% falls outside this range (physically non-logical)
EE_VALID_MIN = 0 # % — drop points below this (e.g. negative EE makes no sense)
EE_VALID_MAX = 100.0   # % — drop points above this (e.g. >100% makes no sense)

# =========================================================
# MANUAL STANDARD CURVE OVERRIDE
# target plate -> donor plate whose fitted average std curve will be used
# =========================================================
ENABLE_MANUAL_CURVE_OVERRIDE = True

MANUAL_CURVE_OVERRIDES = {
    # target:                        donor:
    #("TE_TX_combined_F3_F3C.xlsx", "TE"): ("Another_F3_TE_only.xlsx", "TE"),

    # examples:
    # ("Some_other_file.xlsx", "TE"): ("Another_F3_TE_only.xlsx", "TE"),
    # ("Some_file.xlsx", "TX"): ("Another_F3_TX_only.xlsx", "TX"),
}

# =========================================================
# HELPERS
# =========================================================
def linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan #r2 is calculated as 1 - (residual sum of squares / total sum of squares), with a check to avoid division by zero, residual sum is the sum of squared differences between observed and fitted values, total sum is the sum of squared differences between observed values and their mean.
    return slope, intercept, r2


def conc_from_rfu(rfu, slope, intercept):
    conc = (rfu - intercept) / slope
    if CLIP_NEGATIVE_CONC_TO_ZERO:
        conc = max(conc, 0.0)
    return conc


def get_well_row(well):
    if pd.isna(well):
        return pd.NA
    m = re.match(r"([A-Za-z]+)", str(well).strip())
    return m.group(1).upper() if m else str(well)


def safe_max(series, default=0):
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return default
    return float(series.max())


def clean_dataframe(df, source_file):
    required_input_cols = [
        "plate", "well", "kind", "std_conc_ng_ml", "dilution_x",
        "rfu", "sample", "fraction", "buffer", "note"
    ]
    missing = [c for c in required_input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source_file.name}: missing required columns: {missing}")

    df = df.copy()
    df = df[df["kind"].isin(["standard", "blank", "sample", "exclude"])].copy()

    for col in ["fraction", "std_conc_ng_ml", "dilution_x", "rfu"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["plate", "well", "sample", "buffer", "kind", "note"]:
        df[col] = df[col].astype("string")

    df["source_file"] = source_file.name
    return df


def build_sample_label(row):
    parts = []
    if pd.notna(row.get("sample")) and str(row["sample"]).strip():
        parts.append(str(row["sample"]).strip())
    if pd.notna(row.get("fraction")):
        try:
            parts.append(f"F{int(row['fraction'])}")
        except Exception:
            parts.append(f"F{row['fraction']}")
    if pd.notna(row.get("dilution_x")):
        parts.append(f"{float(row['dilution_x']):g}x")
    if pd.notna(row.get("plate")) and str(row["plate"]).strip():
        parts.append(str(row["plate"]).strip())
    return " | ".join(parts) if parts else str(row.get("well", ""))


def make_flag_text(row):
    flags = []
    if bool(row.get("flag_below_curve", False)):
        flags.append("below_curve")
    if bool(row.get("flag_above_curve", False)):
        flags.append("above_curve")
    if bool(row.get("flag_saturated", False)):
        flags.append("saturated")
    note = row.get("note", None)
    if pd.notna(note) and str(note).strip():
        flags.append(str(note).strip())
    return "; ".join(flags)


def point_legend_label(row):
    sample = str(row["sample"]).strip() if pd.notna(row["sample"]) else "sample"
    frac = f"F{int(row['fraction'])}" if pd.notna(row["fraction"]) else ""
    dil = f"{float(row['dilution_x']):g}x" if pd.notna(row["dilution_x"]) else ""
    well = f" ({row['well']})" if pd.notna(row["well"]) else ""
    flag = f" [{row['flag_text']}]" if pd.notna(row["flag_text"]) and str(row["flag_text"]).strip() else ""
    return " ".join([x for x in [sample, frac, dil] if x]) + well + flag


def fit_curve(std_raw, source_file, plate_name):
    std = std_raw.copy()

    # blanks count as 0 ng/mL
    std.loc[std["kind"] == "blank", "std_conc_ng_ml"] = 0.0

    if not FIT_INCLUDE_BLANK:
        std = std[std["kind"] == "standard"].copy()

    std = std.dropna(subset=["std_conc_ng_ml", "rfu"]).sort_values(["std_conc_ng_ml", "well"])

    if len(std) < 2:
        raise ValueError(f"{source_file.name} / {plate_name}: not enough standard points to fit a curve.")

    std["replicate_id"] = std["well"].apply(get_well_row)

    # -----------------------------------------------------
    # Fit each replicate row separately (A/B/C/...)
    # -----------------------------------------------------
    replicate_rows = []
    for rep_id in sorted(std["replicate_id"].dropna().unique()):
        rep = std[std["replicate_id"] == rep_id].copy().sort_values("std_conc_ng_ml")
        rep = rep.dropna(subset=["std_conc_ng_ml", "rfu"])

        if len(rep) < 2:
            continue

        x_rep = rep["std_conc_ng_ml"].to_numpy(dtype=float)
        y_rep = rep["rfu"].to_numpy(dtype=float)

        rep_slope, rep_intercept, rep_r2 = linear_fit(x_rep, y_rep)

        replicate_rows.append({
            "source_file": source_file.name,
            "curve_id": f"{source_file.stem}__{plate_name}",
            "plate": plate_name,
            "replicate_id": rep_id,
            "rep_slope": rep_slope,
            "rep_intercept": rep_intercept,
            "rep_r2": rep_r2,
            "include_in_avg": rep_r2 >= MIN_REPLICATE_R2_FOR_AVG,
            "n_points": len(rep),
        })

    if not replicate_rows:
        raise ValueError(f"{source_file.name} / {plate_name}: could not fit any replicate standard curves.")

    replicate_stats = pd.DataFrame(replicate_rows)

    good_reps = replicate_stats.loc[
        replicate_stats["include_in_avg"], "replicate_id"
    ].tolist()

    if len(good_reps) == 0:
        raise ValueError(
            f"{source_file.name} / {plate_name}: no replicate standard curves passed "
            f"R² >= {MIN_REPLICATE_R2_FOR_AVG:.2f}, so no average curve was built."
        )

    # keep all raw points for plotting, but mark whether their replicate is used
    raw_std_points = std.copy()
    raw_std_points["curve_id"] = f"{source_file.stem}__{plate_name}"
    raw_std_points = raw_std_points.merge(
        replicate_stats[["replicate_id", "rep_r2", "include_in_avg"]],
        on="replicate_id",
        how="left"
    )

    # -----------------------------------------------------
    # Build average standard points ONLY from good replicates
    # -----------------------------------------------------
    std_used = std[std["replicate_id"].isin(good_reps)].copy()

    if AVERAGE_STANDARDS_BY_CONC:
        fit_points = (
            std_used.groupby("std_conc_ng_ml", dropna=False)
               .agg(
                   rfu=("rfu", "mean"),
                   rfu_sd=("rfu", "std"),
                   n_reps=("rfu", "size"),
               )
               .reset_index()
               .sort_values("std_conc_ng_ml")
        )
    else:
        fit_points = std_used[["std_conc_ng_ml", "rfu"]].copy()
        fit_points["rfu_sd"] = np.nan
        fit_points["n_reps"] = 1

    x = fit_points["std_conc_ng_ml"].to_numpy(dtype=float)
    y = fit_points["rfu"].to_numpy(dtype=float)

    slope, intercept, r2 = linear_fit(x, y)

    curve_row = {
        "source_file": source_file.name,
        "curve_id": f"{source_file.stem}__{plate_name}",
        "plate": plate_name,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "std_conc_min_ng_ml": float(np.min(x)),
        "std_conc_max_ng_ml": float(np.max(x)),
        "std_rfu_min": float(np.min(y)),
        "std_rfu_max": float(np.max(y)),
        "n_raw_std_wells": len(raw_std_points),
        "n_fit_points": len(fit_points),
        "fit_mode": "avg_by_concentration_filtered_replicates" if AVERAGE_STANDARDS_BY_CONC else "all_points_filtered_replicates",
        "n_replicates_total": int(replicate_stats["replicate_id"].nunique()),
        "n_replicates_used": int(len(good_reps)),
        "used_replicates": ",".join(map(str, good_reps)),
    }

    fit_points = fit_points.copy()
    fit_points["source_file"] = source_file.name
    fit_points["curve_id"] = f"{source_file.stem}__{plate_name}"
    fit_points["plate"] = plate_name
    fit_points["kind"] = np.where(fit_points["std_conc_ng_ml"] == 0, "blank", "standard")

    return curve_row, fit_points, raw_std_points, replicate_stats


# =========================================================
# LOAD ALL FILES
# =========================================================
all_rows = []
all_excluded = []

for file in FILES:
    df_tmp = pd.read_excel(file, sheet_name=SHEET)
    df_tmp = clean_dataframe(df_tmp, file)

    excluded = df_tmp[df_tmp["kind"] == "exclude"].copy()
    kept = df_tmp[df_tmp["kind"] != "exclude"].copy()

    all_rows.append(kept)
    if len(excluded):
        all_excluded.append(excluded)

df = pd.concat(all_rows, ignore_index=True)
exclude_rows = pd.concat(all_excluded, ignore_index=True) if all_excluded else pd.DataFrame(columns=df.columns)

# =========================================================
# FIT CURVES FILE-BY-FILE, PLATE-BY-PLATE
# average curve is built only from replicate std curves with R² >= MIN_REPLICATE_R2_FOR_AVG
# =========================================================
curve_rows = []
fit_points_rows = []
raw_std_rows = []
replicate_fit_rows = []

for source_file_name in df["source_file"].dropna().unique():
    source_mask = df["source_file"] == source_file_name
    source_path = next(p for p in FILES if p.name == source_file_name)

    for plate_name in sorted(df.loc[source_mask, "plate"].dropna().unique()):
        std_raw = df[
            source_mask
            & (df["plate"] == plate_name)
            & (df["kind"].isin(["standard", "blank"]))
        ].copy()

        if len(std_raw) == 0:
            continue

        try:
            curve_row, fit_points, raw_std_points, replicate_stats = fit_curve(
                std_raw, source_path, plate_name
            )
        except ValueError as e:
            print(f"[WARNING] {e}")
            continue

        curve_rows.append(curve_row)
        fit_points_rows.append(fit_points)
        raw_std_rows.append(raw_std_points)
        replicate_fit_rows.append(replicate_stats)

curve_df = pd.DataFrame(curve_rows).sort_values(["source_file", "plate"]).reset_index(drop=True)
fit_points_df = pd.concat(fit_points_rows, ignore_index=True) if fit_points_rows else pd.DataFrame()
raw_std_df = pd.concat(raw_std_rows, ignore_index=True) if raw_std_rows else pd.DataFrame()
replicate_fit_df = pd.concat(replicate_fit_rows, ignore_index=True) if replicate_fit_rows else pd.DataFrame()

# =========================================================
# RESOLVE WHICH CURVE EACH PLATE SHOULD USE
# default = use its own fitted curve
# if manual override is defined, use donor plate's fitted curve instead
# =========================================================
curve_lookup = {}
if not curve_df.empty:
    for _, r in curve_df.iterrows():
        curve_lookup[(r["source_file"], r["plate"])] = r.to_dict()

effective_curve_key_map = {}

all_source_plate_keys = (
    df[["source_file", "plate"]]
    .dropna()
    .drop_duplicates()
    .itertuples(index=False, name=None)
)

for key in all_source_plate_keys:
    source_file, plate = key

    if ENABLE_MANUAL_CURVE_OVERRIDE and key in MANUAL_CURVE_OVERRIDES:
        donor_key = MANUAL_CURVE_OVERRIDES[key]

        if donor_key not in curve_lookup:
            print(
                f"[WARNING] Manual override requested for {key}, "
                f"but donor curve {donor_key} was not found. Falling back to native curve."
            )
            effective_curve_key_map[key] = key
        else:
            effective_curve_key_map[key] = donor_key
    else:
        effective_curve_key_map[key] = key

curve_assignment_rows = []
for target_key, effective_key in sorted(effective_curve_key_map.items()):
    target_file, target_plate = target_key
    curve_file, curve_plate = effective_key
    curve_assignment_rows.append({
        "target_source_file": target_file,
        "target_plate": target_plate,
        "curve_source_file": curve_file,
        "curve_source_plate": curve_plate,
        "manual_override": bool(target_key != effective_key),
        "note": (
            f"using manual override from {curve_file} | {curve_plate}"
            if target_key != effective_key else
            "native curve"
        )
    })

curve_assignment_df = pd.DataFrame(curve_assignment_rows)

# =========================================================
# CALCULATE SAMPLE CONCENTRATIONS
# uses manual curve override if defined
# =========================================================
samples = df[df["kind"] == "sample"].copy().reset_index(drop=True)

# columns describing which curve was actually used
samples["curve_source_file"] = pd.NA
samples["curve_source_plate"] = pd.NA
samples["curve_id"] = pd.NA
samples["manual_curve_override"] = False
samples["curve_override_note"] = pd.NA

for col in ["slope", "intercept", "r2", "std_rfu_min", "std_rfu_max"]:
    samples[col] = np.nan

for idx, row in samples.iterrows():
    target_key = (row["source_file"], row["plate"])
    effective_key = effective_curve_key_map.get(target_key, target_key)

    params = curve_lookup.get(effective_key)
    if params is None:
        continue

    samples.at[idx, "curve_source_file"] = effective_key[0]
    samples.at[idx, "curve_source_plate"] = effective_key[1]
    samples.at[idx, "curve_id"] = params["curve_id"]

    for k in ["slope", "intercept", "r2", "std_rfu_min", "std_rfu_max"]:
        samples.at[idx, k] = params[k]

    if effective_key != target_key:
        samples.at[idx, "manual_curve_override"] = True
        samples.at[idx, "curve_override_note"] = (
            f"used curve from {effective_key[0]} | {effective_key[1]}"
        )
    else:
        samples.at[idx, "curve_override_note"] = "native curve"

samples["conc_diluted_ng_ml"] = [
    conc_from_rfu(rfu, slope, intercept)
    if pd.notna(rfu) and pd.notna(slope) and pd.notna(intercept)
    else np.nan
    for rfu, slope, intercept in zip(samples["rfu"], samples["slope"], samples["intercept"])
]

samples["conc_original_ng_ml"] = samples["conc_diluted_ng_ml"] * samples["dilution_x"]

samples["flag_below_curve"] = samples["rfu"] < samples["std_rfu_min"]
samples["flag_above_curve"] = samples["rfu"] > samples["std_rfu_max"]
samples["flag_saturated"] = samples["rfu"] >= SATURATION_RFU

samples["is_flagged"] = (
    samples["flag_below_curve"].fillna(False)
    | samples["flag_above_curve"].fillna(False)
    | samples["flag_saturated"].fillna(False)
)

samples["flag_short"] = samples.apply(
    lambda r: ",".join([
        name for name, cond in [
            ("below", r["flag_below_curve"]),
            ("above", r["flag_above_curve"]),
            ("sat",   r["flag_saturated"]),
        ] if bool(cond)
    ]),
    axis=1
)

samples["flag_text"] = samples.apply(make_flag_text, axis=1)
samples["sample_label"] = samples.apply(build_sample_label, axis=1)

samples = samples.sort_values(
    ["source_file", "plate", "sample", "fraction", "dilution_x", "well"]
).reset_index(drop=True)

# =========================================================
# PLOTS: STANDARD CURVES WITH SAMPLE POINTS
# if manual override is active for a target plate, the donor curve is shown
# =========================================================
sample_marker_map = {
    "F3": "o",
    "F3C": "s",
}

target_keys_for_plot = (
    df[["source_file", "plate"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["source_file", "plate"])
    .itertuples(index=False, name=None)
)

for source_file, plate in target_keys_for_plot:
    sub = samples[
        (samples["source_file"] == source_file) & (samples["plate"] == plate)
    ].copy()

    if sub.empty:
        continue

    effective_key = effective_curve_key_map.get((source_file, plate), (source_file, plate))
    curve = curve_lookup.get(effective_key)
    if curve is None:
        continue

    curve_id = curve["curve_id"]

    std_raw = raw_std_df[raw_std_df["curve_id"] == curve_id].copy()
    fit_pts = fit_points_df[fit_points_df["curve_id"] == curve_id].copy()

    sub = sub.sort_values(
        ["sample", "fraction", "dilution_x", "well"],
        na_position="last"
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    h_std = ax.scatter(
        std_raw["std_conc_ng_ml"], std_raw["rfu"],
        s=40, alpha=0.6, color="0.7", label="standard wells"
    )

    h_fit = ax.scatter(
        fit_pts["std_conc_ng_ml"], fit_pts["rfu"],
        s=90, marker="D", color="black", label="fit points (avg)"
    )

    x_max = max(
        safe_max(fit_pts["std_conc_ng_ml"], default=0) * 1.1,
        safe_max(sub["conc_diluted_ng_ml"], default=0) * 1.1,
        110
    )
    x_line = np.linspace(0, x_max, 300)
    y_line = curve["slope"] * x_line + curve["intercept"]
    h_line, = ax.plot(
        x_line, y_line,
        linestyle="--", color="black", linewidth=1.5, label="linear fit"
    )

    point_handles = []
    point_cmap = plt.cm.get_cmap("tab20", max(len(sub), 1))

    for i, (_, row) in enumerate(sub.iterrows()):
        color = point_cmap(i % point_cmap.N)
        marker = sample_marker_map.get(str(row["sample"]), "^")

        edgecolor = "black"
        linewidth = 0.8
        if bool(row["is_flagged"]):
            edgecolor = "red"
            linewidth = 1.8

        ax.scatter(
            row["conc_diluted_ng_ml"],
            row["rfu"],
            s=90,
            marker=marker,
            color=color,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=3,
        )

        point_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=edgecolor,
                markeredgewidth=linewidth,
                color=color,
                markersize=8,
                label=point_legend_label(row)
            )
        )

    override_text = ""
    if effective_key != (source_file, plate):
        override_text = (
            f"\nmanual override: using curve from "
            f"{effective_key[0]} | {effective_key[1]}"
        )

    ax.set_title(f"{source_file} | {plate} standard curve{override_text}")
    ax.set_xlabel("Concentration in diluted assay sample (ng/mL)")
    ax.set_ylabel("Fluorescence (RFU)")

    used_reps_text = curve.get("used_replicates", "")
    ax.text(
        0.98, 0.97,
        f"y = {curve['slope']:.2f}x + {curve['intercept']:.2f}\n"
        f"R² = {curve['r2']:.4f}\n"
        f"used reps: {used_reps_text}",
        transform=ax.transAxes,
        ha="right", va="top"
    )

    legend_main = ax.legend(
        handles=[
            h_std,
            h_fit,
            h_line,
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markerfacecolor='white',
                markeredgecolor='red',
                markeredgewidth=1.8,
                color='red',
                markersize=8,
                label='flagged sample point'
            )
        ],
        loc="upper left",
        fontsize=8
    )
    ax.add_artist(legend_main)

    shape_handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='black', markersize=8, label='F3'),
        Line2D([0], [0], marker='s', linestyle='None', color='black', markersize=8, label='F3C'),
    ]
    legend_shape = ax.legend(
        handles=shape_handles,
        title="Sample type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=8
    )
    ax.add_artist(legend_shape)

    if point_handles:
        ax.legend(
            handles=point_handles,
            title="Sample points",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.62),
            fontsize=7
        )

    fig.tight_layout()

    if SAVE_PLOTS:
        # Create a shorter filename to avoid path length issues
        stem = Path(source_file).stem.replace(" ", "_")
        safe_name = f"{stem}_{plate}_std_curve.png"
        filepath = str(OUTDIR / safe_name).replace("\\", "/")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# EXTRA PLOTS: REPLICATE STANDARD CURVES WITHIN EACH FILE + PLATE
# - replicate standard points are NOT connected
# - fit a linear line to each replicate separately
# - show ONLY the average fit line (no average fit points)
# - print each replicate fit equation on the right in matching color
# - if R² < threshold, that replicate is excluded from average-curve building
# =========================================================
for _, curve in curve_df.iterrows():
    source_file = curve["source_file"]
    plate = curve["plate"]
    curve_id = curve["curve_id"]

    std_raw = raw_std_df[raw_std_df["curve_id"] == curve_id].copy()
    rep_stats = replicate_fit_df[replicate_fit_df["curve_id"] == curve_id].copy()

    if std_raw.empty or rep_stats.empty:
        continue

    std_raw["replicate_id"] = std_raw["well"].apply(get_well_row)
    std_raw = std_raw.sort_values(["replicate_id", "std_conc_ng_ml", "well"])

    rep_ids = [x for x in std_raw["replicate_id"].dropna().unique()]
    rep_ids = sorted(rep_ids)

    if len(rep_ids) == 0:
        continue

    rep_stats_map = rep_stats.set_index("replicate_id").to_dict("index")

    fig, ax = plt.subplots(figsize=(11, 6.5))

    rep_cmap = plt.cm.get_cmap("tab10", max(len(rep_ids), 1))
    rep_point_handles = []
    rep_fit_handles = []
    rep_fit_texts = []

    x_max_data = safe_max(std_raw["std_conc_ng_ml"], default=100)
    x_line = np.linspace(0, max(x_max_data * 1.1, 110), 300)

    for i, rep_id in enumerate(rep_ids):
        rep = std_raw[std_raw["replicate_id"] == rep_id].copy().sort_values("std_conc_ng_ml")
        rep = rep.dropna(subset=["std_conc_ng_ml", "rfu"])
        if len(rep) < 2:
            continue

        color = rep_cmap(i % rep_cmap.N)

        ax.scatter(
            rep["std_conc_ng_ml"],
            rep["rfu"],
            s=55,
            color=color,
            marker="o",
            zorder=3,
        )

        rep_info = rep_stats_map.get(rep_id)
        if rep_info is None:
            continue

        rep_slope = rep_info["rep_slope"]
        rep_intercept = rep_info["rep_intercept"]
        rep_r2 = rep_info["rep_r2"]
        rep_used = bool(rep_info["include_in_avg"])

        rep_y_line = rep_slope * x_line + rep_intercept
        ax.plot(
            x_line,
            rep_y_line,
            linestyle="-",
            color=color,
            linewidth=1.8,
            alpha=0.95,
            zorder=2,
        )

        rep_point_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color=color,
                markerfacecolor=color,
                markersize=7,
                label=f"replicate {rep_id} points"
            )
        )

        rep_fit_handles.append(
            Line2D(
                [0], [0],
                linestyle="-",
                color=color,
                linewidth=1.8,
                label=f"replicate {rep_id} fit"
            )
        )

        status_text = "used" if rep_used else "excluded"
        rep_fit_texts.append(
            (
                color,
                f"{rep_id}: y = {rep_slope:.2f}x + {rep_intercept:.2f}, R² = {rep_r2:.4f} [{status_text}]"
            )
        )

    avg_y_line = curve["slope"] * x_line + curve["intercept"]
    avg_handle, = ax.plot(
        x_line,
        avg_y_line,
        linestyle="--",
        color="black",
        linewidth=2.4,
        label="average fit line"
    )

    ax.set_title(f"{Path(source_file).stem} | {plate} replicate standard curves")
    ax.set_xlabel("Standard concentration (ng/mL)")
    ax.set_ylabel("Fluorescence (RFU)")

    fig.subplots_adjust(right=0.72)

    ax.text(
        1.02, 0.97,
        f"avg: y = {curve['slope']:.2f}x + {curve['intercept']:.2f}\n"
        f"R² = {curve['r2']:.4f}\n"
        f"used reps: {curve['used_replicates']}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        color="black"
    )

    y0 = 0.84
    dy = 0.08
    for j, (txt_color, txt) in enumerate(rep_fit_texts):
        ax.text(
            1.02, y0 - j * dy,
            txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            color=txt_color
        )

    legend1 = ax.legend(
        handles=rep_point_handles,
        title="Replicate points",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.42),
        fontsize=8
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=rep_fit_handles + [avg_handle],
        title="Fit lines",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.12),
        fontsize=8
    )
    ax.add_artist(legend2)

    fig.tight_layout()

    if SAVE_PLOTS:
        filepath = str(OUTDIR / f"{Path(source_file).stem}_{plate}_replicate_standard_curves_with_fits.png")
        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# ADDITIONAL PLOTS: OVERLAY OF AVERAGE STANDARD CURVES
# one plot for all TE average curves
# one plot for all TX average curves
# =========================================================
for plate_name in ["TE", "TX"]:
    curve_sub = curve_df[curve_df["plate"] == plate_name].copy()
    fit_sub = fit_points_df[fit_points_df["plate"] == plate_name].copy()

    if curve_sub.empty or fit_sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    curve_sub = curve_sub.sort_values("source_file").reset_index(drop=True)
    cmap = plt.cm.get_cmap("tab10", max(len(curve_sub), 1))

    point_handles = []
    line_handles = []

    x_max_data = safe_max(fit_sub["std_conc_ng_ml"], default=100)
    x_line = np.linspace(0, max(x_max_data * 1.1, 110), 300)

    for i, (_, curve) in enumerate(curve_sub.iterrows()):
        color = cmap(i % cmap.N)
        curve_id = curve["curve_id"]
        label_base = Path(curve["source_file"]).stem

        pts = fit_sub[fit_sub["curve_id"] == curve_id].copy().sort_values("std_conc_ng_ml")
        if pts.empty:
            continue

        # average fit points used for this curve
        ax.scatter(
            pts["std_conc_ng_ml"],
            pts["rfu"],
            s=60,
            color=color,
            marker="o",
            zorder=3,
        )

        # average fit line
        y_line = curve["slope"] * x_line + curve["intercept"]
        ax.plot(
            x_line,
            y_line,
            linestyle="-",
            color=color,
            linewidth=2.0,
            zorder=2,
        )

        used_reps_text = curve["used_replicates"] if pd.notna(curve["used_replicates"]) else ""

        point_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color=color,
                markerfacecolor=color,
                markersize=7,
                label=f"{label_base} points"
            )
        )

        line_handles.append(
            Line2D(
                [0], [0],
                linestyle="-",
                color=color,
                linewidth=2.0,
                label=(
                    f"{label_base} fit: "
                    f"y = {curve['slope']:.2f}x + {curve['intercept']:.2f}, "
                    f"R² = {curve['r2']:.4f}"
                    + (f", reps: {used_reps_text}" if used_reps_text else "")
                )
            )
        )

    ax.set_title(f"{plate_name}: average standard curves from all plates")
    ax.set_xlabel("Standard concentration (ng/mL)")
    ax.set_ylabel("Fluorescence (RFU)")

    legend1 = ax.legend(
        handles=point_handles,
        title="Average standard points",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=8
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=line_handles,
        title="Average fit lines",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        fontsize=8
    )

    fig.tight_layout()

    if SAVE_PLOTS:
        filepath = str(OUTDIR / f"{plate_name}_all_average_standard_curves.png")
        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# PLOT ORIGINAL CONCENTRATIONS - SEPARATE PLOTS FOR TE AND TX
# x-axis = sample name + fraction (e.g., "F3 F15", "F3 F16")
# y-axis = original concentration
# marker shape = sample type (F3 vs F3C)
# label = dilution factor
# flagged points = hollow red outline
# =========================================================
samples["point_label"] = samples.apply(
    lambda r: (f"{float(r['dilution_x']):g}x" if pd.notna(r["dilution_x"]) else ""),
    axis=1
)

# Color for each plate
plate_color_map = {
    "TE": "#1f77b4",  # blue
    "TX": "#ff7f0e",  # orange
}

# Marker map by sample type
sample_marker_map = {
    "F3": "o", # circle
    "F3C": "s", # square
}

# Create x-axis labels: "sample fraction" (e.g., "F3 F15")
samples["x_label"] = samples.apply(
    lambda r: (
        str(r["sample"])
        + (" F" + str(int(r["fraction"])) if pd.notna(r["fraction"]) else "")
    ),
    axis=1
)

# Loop over each plate (TE and TX)
for plate_name in ["TE", "TX"]:
    plate_samples = samples[samples["plate"] == plate_name].copy()
    if plate_samples.empty:
        continue

    # Get unique x-axis labels for this plate in sorted order
    unique_x_labels = sorted(plate_samples["x_label"].dropna().unique())

    # Create a mapping from x-label to x-position
    x_pos_map = {label: i for i, label in enumerate(unique_x_labels)}

    fig, ax = plt.subplots(figsize=(max(12, 1.0 * len(unique_x_labels)), 7))

    # Plot all samples for this plate, grouped by (sample, fraction)
    for x_label in unique_x_labels:
        sub = plate_samples[plate_samples["x_label"] == x_label].copy()
        if sub.empty:
            continue

        x = x_pos_map[x_label]

        # For each point (dilution) in this sample+fraction, plot it
        for _, row in sub.iterrows():
            y = row["conc_original_ng_ml"]
            marker = sample_marker_map.get(str(row["sample"]), "^")
            color = plate_color_map.get(plate_name, "gray")

            if bool(row["is_flagged"]):
                # Flagged: hollow with red outline
                ax.scatter(
                    x, y,
                    s=130,
                    marker=marker,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.8,
                    zorder=3,
                )
            else:
                # Trusted: filled
                ax.scatter(
                    x, y,
                    s=85,
                    marker=marker,
                    color=color,
                    zorder=3,
                )

            # Label with dilution
            if ANNOTATE_SAMPLES:
                extra = f" *({row['flag_short']})" if row["is_flagged"] else ""
                ax.annotate(
                    row["point_label"] + extra,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    rotation=45,
                )

    ax.set_title(f"{plate_name}: Original sample concentrations")
    ax.set_xlabel("Sample Fraction")
    ax.set_ylabel("Original concentration (ng/mL)")
    ax.set_xticks(range(len(unique_x_labels)))
    ax.set_xticklabels(unique_x_labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Create legend handles
    marker_handles = [
        Line2D([0], [0], marker=sample_marker_map.get(s, "^"), linestyle='None',
               color='black', markerfacecolor='black', markersize=8, label=s)
        for s in ["F3", "F3C"] if s in plate_samples["sample"].unique()
    ]

    flag_handle = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markerfacecolor='none',
        markeredgecolor='red',
        markeredgewidth=1.8,
        markersize=8,
        label='Flagged / not trusted'
    )

    legend1 = ax.legend(
        handles=marker_handles,
        title="Sample type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=8
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=[flag_handle],
        title="Status",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65),
        fontsize=8
    )

    fig.tight_layout()

    if SAVE_PLOTS:
        filepath = str(OUTDIR / f"{plate_name}_original_concentration_grouped_by_fraction.png")
        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# PLOT ORIGINAL CONCENTRATIONS - SEPARATE PLOTS FOR EACH (SOURCE_FILE, PLATE) COMBINATION
# =========================================================
for source_file, plate_name in (
    samples[["source_file", "plate"]]
    .drop_duplicates()
    .sort_values(["plate", "source_file"])
    .itertuples(index=False, name=None)
):
    plate_samples = samples[
        (samples["source_file"] == source_file) & (samples["plate"] == plate_name)
    ].copy()
    
    if plate_samples.empty:
        continue

    # Get unique x-axis labels for this plate in sorted order
    unique_x_labels = sorted(plate_samples["x_label"].dropna().unique())

    # Create a mapping from x-label to x-position
    x_pos_map = {label: i for i, label in enumerate(unique_x_labels)}

    fig, ax = plt.subplots(figsize=(max(12, 1.0 * len(unique_x_labels)), 7))

    # Plot all samples for this plate, grouped by (sample, fraction)
    for x_label in unique_x_labels:
        sub = plate_samples[plate_samples["x_label"] == x_label].copy()
        if sub.empty:
            continue

        x = x_pos_map[x_label]

        # For each point (dilution) in this sample+fraction, plot it
        for _, row in sub.iterrows():
            y = row["conc_original_ng_ml"]
            marker = sample_marker_map.get(str(row["sample"]), "^")
            color = plate_color_map.get(plate_name, "gray")

            if bool(row["is_flagged"]):
                # Flagged: hollow with red outline
                ax.scatter(
                    x, y,
                    s=130,
                    marker=marker,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.8,
                    zorder=3,
                )
            else:
                # Trusted: filled
                ax.scatter(
                    x, y,
                    s=85,
                    marker=marker,
                    color=color,
                    zorder=3,
                )

            # Label with dilution
            if ANNOTATE_SAMPLES:
                extra = f" *({row['flag_short']})" if row["is_flagged"] else ""
                ax.annotate(
                    row["point_label"] + extra,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    rotation=45,
                )

    ax.set_title(f"{source_file} | {plate_name}: Original sample concentrations")
    ax.set_xlabel("Sample Fraction")
    ax.set_ylabel("Original concentration (ng/mL)")
    ax.set_xticks(range(len(unique_x_labels)))
    ax.set_xticklabels(unique_x_labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Create legend handles
    marker_handles = [
        Line2D([0], [0], marker=sample_marker_map.get(s, "^"), linestyle='None',
               color='black', markerfacecolor='black', markersize=8, label=s)
        for s in ["F3", "F3C"] if s in plate_samples["sample"].unique()
    ]

    flag_handle = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markerfacecolor='none',
        markeredgecolor='red',
        markeredgewidth=1.8,
        markersize=8,
        label='Flagged / not trusted'
    )

    legend1 = ax.legend(
        handles=marker_handles,
        title="Sample type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=8
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=[flag_handle],
        title="Status",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65),
        fontsize=8
    )

    fig.tight_layout()

    if SAVE_PLOTS:
        safe_source_name = Path(source_file).stem.replace(" ", "_")
        filepath = str(OUTDIR / f"{safe_source_name}_{plate_name}_original_concentration.png")
        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# ADDITIONAL PLOTS: AVERAGE CONCENTRATION CURVES BY SAMPLE TYPE
# one plot for TE: curve through average F3 conc + curve for F3C
# one plot for TX: same
# x-axis = fraction number (numeric)
# only non-flagged points used for the average curves
# individual scatter points shown for context
# =========================================================
avg_curve_colors = {
    "F3":  "#1f77b4",  # blue
    "F3C": "#d62728",  # red
}

for plate_name in ["TE", "TX"]:
    plate_samples = samples[samples["plate"] == plate_name].copy()
    if plate_samples.empty:
        continue

    all_fracs = plate_samples["fraction"].dropna().unique()
    fig_width = max(12, 0.6 * len(all_fracs))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    legend_handles = []

    for sample_type in ["F3", "F3C"]:
        sub = plate_samples[plate_samples["sample"] == sample_type].copy()
        if sub.empty:
            continue

        col = avg_curve_colors.get(sample_type, "gray")
        marker = sample_marker_map.get(sample_type, "^")

        # Individual scatter points (all, flagged in red outline)
        for _, row in sub.iterrows():
            y = row["conc_original_ng_ml"]
            if pd.isna(y):
                continue
            x = row["fraction"]
            if bool(row["is_flagged"]):
                ax.scatter(x, y, s=60, marker=marker, facecolors="none",
                           edgecolors="red", linewidths=1.5, zorder=3, alpha=0.7)
            else:
                ax.scatter(x, y, s=50, marker=marker, color=col,
                           zorder=3, alpha=0.4)

        # Average curve from non-flagged points only
        trusted = sub[~sub["is_flagged"]].dropna(subset=["fraction", "conc_original_ng_ml"])
        avg_df = (
            trusted
            .groupby("fraction", dropna=True)["conc_original_ng_ml"]
            .mean()
            .reset_index()
            .sort_values("fraction")
        )

        if len(avg_df) >= 2:
            ax.plot(
                avg_df["fraction"],
                avg_df["conc_original_ng_ml"],
                linestyle="-",
                color=col,
                linewidth=2.0,
                marker=marker,
                markersize=7,
                zorder=4,
            )

        legend_handles.append(
            Line2D([0], [0], marker=marker, linestyle="-", color=col,
                   markerfacecolor=col, markersize=8,
                   label=f"{sample_type} avg (non-flagged)")
        )

    flag_handle = Line2D(
        [0], [0], marker="o", linestyle="None",
        markerfacecolor="none", markeredgecolor="red",
        markeredgewidth=1.8, markersize=8, label="Flagged / not trusted"
    )

    ax.set_title(f"{plate_name}: Average concentration by fraction")
    ax.set_xlabel("Fraction number")
    ax.set_ylabel("Original concentration (ng/mL)")
    ax.grid(axis="y", alpha=0.3)

    legend1 = ax.legend(
        handles=legend_handles,
        title="Sample type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=8,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=[flag_handle],
        title="Status",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        fontsize=8,
    )

    fig.tight_layout()

    if SAVE_PLOTS:
        filepath = str(OUTDIR / f"{plate_name}_average_concentration_curves.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# FINAL PLOT: ENCAPSULATION EFFICIENCY (EE%)
# EE% = 100 * (TX - TE) / TX
# computed from average non-flagged concentrations per (sample_type, fraction)
# two curves: F3 and F3C
# =========================================================

# Build average non-flagged conc per (plate, sample_type, fraction)
trusted_all = samples[~samples["is_flagged"]].dropna(subset=["fraction", "conc_original_ng_ml"])
avg_by_group = (
    trusted_all
    .groupby(["plate", "sample", "fraction"], dropna=True)["conc_original_ng_ml"]
    .mean()
    .reset_index()
    .rename(columns={"conc_original_ng_ml": "avg_conc"})
)

te_avg = avg_by_group[avg_by_group["plate"] == "TE"][["sample", "fraction", "avg_conc"]].rename(columns={"avg_conc": "conc_TE"})
tx_avg = avg_by_group[avg_by_group["plate"] == "TX"][["sample", "fraction", "avg_conc"]].rename(columns={"avg_conc": "conc_TX"})

ee_df = pd.merge(te_avg, tx_avg, on=["sample", "fraction"], how="inner")
ee_df = ee_df[ee_df["conc_TX"] > 0].copy()
ee_df["EE_pct"] = 100.0 * (ee_df["conc_TX"] - ee_df["conc_TE"]) / ee_df["conc_TX"]
ee_df = ee_df[ee_df["EE_pct"].between(EE_VALID_MIN, EE_VALID_MAX)].copy()
ee_df = ee_df.sort_values(["sample", "fraction"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(max(10, 0.6 * ee_df["fraction"].nunique()), 6))

for sample_type in ["F3", "F3C"]:
    sub_ee = ee_df[ee_df["sample"] == sample_type].sort_values("fraction")
    if sub_ee.empty:
        continue
    col = avg_curve_colors.get(sample_type, "gray")
    marker = sample_marker_map.get(sample_type, "^")
    ax.plot(
        sub_ee["fraction"],
        sub_ee["EE_pct"],
        linestyle="-",
        color=col,
        linewidth=2.0,
        marker=marker,
        markersize=7,
        label=f"{sample_type}  (mean EE = {sub_ee['EE_pct'].mean():.1f}%)",
    )

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("Encapsulation efficiency: EE% = 100 × (TX − TE) / TX")
ax.set_xlabel("Fraction number")
ax.set_ylabel("EE (%)")
ax.grid(axis="y", alpha=0.3)
ax.legend(title="Sample type", fontsize=9)

fig.tight_layout()

if SAVE_PLOTS:
    filepath = str(OUTDIR / "EE_encapsulation_efficiency_by_fraction.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =========================================================
# BUILD ONE CLEAN FINAL OUTPUT TABLE
# std_raw = all standard wells
# std_avg = averaged fit points actually used for fitting
# sample = samples with fitted concentrations and comments
# =========================================================
std_raw_table = raw_std_df.copy()
std_raw_table["row_type"] = "std_raw"
std_raw_table["sample"] = pd.NA
std_raw_table["fraction"] = np.nan
std_raw_table["conc_diluted_ng_ml"] = np.nan
std_raw_table["conc_original_ng_ml"] = np.nan
std_raw_table["rfu_sd"] = np.nan
std_raw_table["n_reps"] = np.nan
std_raw_table["curve_source_file"] = std_raw_table["source_file"]
std_raw_table["curve_source_plate"] = std_raw_table["plate"]
std_raw_table["manual_curve_override"] = False

std_raw_table["comment"] = std_raw_table.apply(
    lambda r: (
        f"replicate={r['replicate_id']}; "
        f"rep_r2={r['rep_r2']:.4f}; "
        f"{'used_in_avg' if bool(r['include_in_avg']) else 'excluded_from_avg'}"
    ) if pd.notna(r.get("rep_r2")) else "",
    axis=1
)

std_raw_table = std_raw_table[[
    "source_file", "plate", "curve_id", "row_type",
    "curve_source_file", "curve_source_plate", "manual_curve_override",
    "well", "kind", "sample", "fraction",
    "std_conc_ng_ml", "dilution_x", "rfu", "rfu_sd", "n_reps",
    "conc_diluted_ng_ml", "conc_original_ng_ml", "comment"
]].copy()

std_avg_table = fit_points_df.copy()
std_avg_table["row_type"] = "std_avg"
std_avg_table["well"] = pd.NA
std_avg_table["sample"] = pd.NA
std_avg_table["fraction"] = np.nan
std_avg_table["dilution_x"] = 1.0
std_avg_table["conc_diluted_ng_ml"] = np.nan
std_avg_table["conc_original_ng_ml"] = np.nan
std_avg_table["curve_source_file"] = std_avg_table["source_file"]
std_avg_table["curve_source_plate"] = std_avg_table["plate"]
std_avg_table["manual_curve_override"] = False
std_avg_table["comment"] = std_avg_table.apply(
    lambda r: f"average_of_{int(r['n_reps'])}_reps" if pd.notna(r["n_reps"]) else "average_fit_point",
    axis=1
)

std_avg_table = std_avg_table[[
    "source_file", "plate", "curve_id", "row_type",
    "curve_source_file", "curve_source_plate", "manual_curve_override",
    "well", "kind", "sample", "fraction",
    "std_conc_ng_ml", "dilution_x", "rfu", "rfu_sd", "n_reps",
    "conc_diluted_ng_ml", "conc_original_ng_ml", "comment"
]].copy()

sample_table = samples.copy()
sample_table["row_type"] = "sample"
sample_table["std_conc_ng_ml"] = np.nan
sample_table["rfu_sd"] = np.nan
sample_table["n_reps"] = np.nan
sample_table["comment"] = sample_table.apply(
    lambda r: (
        (str(r["flag_text"]).strip() if pd.notna(r["flag_text"]) else "")
        + (" | " if pd.notna(r["flag_text"]) and str(r["flag_text"]).strip() and pd.notna(r["curve_override_note"]) else "")
        + (str(r["curve_override_note"]).strip() if pd.notna(r["curve_override_note"]) else "")
    ),
    axis=1
)

sample_table = sample_table[[
    "source_file", "plate", "curve_id", "row_type",
    "curve_source_file", "curve_source_plate", "manual_curve_override",
    "well", "kind", "sample", "fraction",
    "std_conc_ng_ml", "dilution_x", "rfu", "rfu_sd", "n_reps",
    "conc_diluted_ng_ml", "conc_original_ng_ml", "comment"
]].copy()

final_table = pd.concat(
    [std_raw_table, std_avg_table, sample_table],
    ignore_index=True
)

plate_order = {"TE": 0, "TX": 1}
row_type_order = {"std_raw": 0, "std_avg": 1, "sample": 2}

final_table["plate_sort"] = final_table["plate"].map(plate_order).fillna(99)
final_table["row_type_sort"] = final_table["row_type"].map(row_type_order).fillna(99)
final_table["std_sort"] = final_table["std_conc_ng_ml"].fillna(999999)
final_table["fraction_sort"] = final_table["fraction"].fillna(999999)
final_table["dilution_sort"] = final_table["dilution_x"].fillna(999999)

final_table = final_table.sort_values(
    ["source_file", "plate_sort", "row_type_sort", "std_sort", "fraction_sort", "dilution_sort", "well"]
).reset_index(drop=True)

final_table = final_table.drop(
    columns=["plate_sort", "row_type_sort", "std_sort", "fraction_sort", "dilution_sort"]
)

# =========================================================
# SAVE TABLES
# =========================================================
final_table.to_csv(OUTDIR / "ribogreen_clean_output_table.csv", index=False)

if not curve_df.empty:
    curve_df.to_csv(OUTDIR / "curve_summary.csv", index=False)

if not replicate_fit_df.empty:
    replicate_fit_df.to_csv(OUTDIR / "replicate_fit_summary.csv", index=False)

if not curve_assignment_df.empty:
    curve_assignment_df.to_csv(OUTDIR / "curve_assignment_summary.csv", index=False)

with pd.ExcelWriter(OUTDIR / "ribogreen_clean_output_table.xlsx", engine="openpyxl") as writer:
    final_table.to_excel(writer, sheet_name="Clean_output", index=False)

    if not curve_df.empty:
        curve_df.to_excel(writer, sheet_name="Curve_summary", index=False)

    if not replicate_fit_df.empty:
        replicate_fit_df.to_excel(writer, sheet_name="Replicate_fit_summary", index=False)

    if not curve_assignment_df.empty:
        curve_assignment_df.to_excel(writer, sheet_name="Curve_assignment", index=False)

# =========================================================
# PRINT KEY RESULTS
# =========================================================
print("\n=== Curve summary ===")
if not curve_df.empty:
    print(curve_df.to_string(index=False))
else:
    print("No valid curves.")

print("\n=== Replicate fit summary ===")
if not replicate_fit_df.empty:
    print(replicate_fit_df.to_string(index=False))
else:
    print("No replicate fits.")

print("\n=== Curve assignment summary ===")
if not curve_assignment_df.empty:
    print(curve_assignment_df.to_string(index=False))
else:
    print("No curve assignments.")

print("\n=== Clean final output table ===")
print(final_table.to_string(index=False))

print(f"\nSaved results to: {OUTDIR}")