from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import subprocess
import re
import shutil
import tempfile

# ── Paths for online samples ───────────────────────────────────────────────────────────────────
# BASE_FOLDER   = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS\process\Tina ONLINE AF4 SAXS")

# # ↓ Only change this line when switching datasets
# SAMPLE_FOLDER = "F3_mRNA_TRIS_scan-84556_shotsE_subtracted_50frame_blocks2"
# # F3_empty_TRIS_scan-84555_shotsE_subtracted_50frame_blocks2
# # F3_mRNA_TRIS_scan-84556_shotsE_subtracted_50frame_blocks2
# # F3C_emtpy_TRIS_scan-84557_shotsE_subtracted_50frame_blocks2
# # F3C_mRNA_TRIS_scan-84558_shotsE_subtracted_50frame_blocks2

# ── Paths for offline samples ──────────────────────────────────────────────────────────────────
BASE_FOLDER   = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS")

# ↓ Only change this line when switching datasets
SAMPLE_FOLDER = "Tina OFFLINE SAXS"
# F3_empty_TRIS_scan-84555_shotsE_subtracted_50frame_blocks2
# F3_mRNA_TRIS_scan-84556_shotsE_subtracted_50frame_blocks2
# F3C_emtpy_TRIS_scan-84557_shotsE_subtracted_50frame_blocks2
# F3C_mRNA_TRIS_scan-84558_shotsE_subtracted_50frame_blocks2

INPUT_FOLDER  = BASE_FOLDER / SAMPLE_FOLDER

# Guinier results table (.xlsx, .xls, or .csv)
GUINIER_TABLE = INPUT_FOLDER / "guinier_results.csv"   # change if needed

# All outputs (*.out, logs, summary CSV) go here (created automatically)
OUTPUT_FOLDER = INPUT_FOLDER / f"p(r)_datgnom_{date.today():%Y-%m-%d}"

# ── File extensions to process ──────────────────────────────────────────────
# Case-insensitive; all files with these extensions in INPUT_FOLDER are candidates.
FILE_EXTENSIONS = [".dat", ".txt"]

# ── datgnom / autorg executables ────────────────────────────────────────────
DATGNOM_EXE = r"C:\Program Files\ATSAS-3.1.3\bin\datgnom.exe"
AUTORG_EXE  = r"C:\Program Files\ATSAS-3.1.3\bin\autorg.exe"

# ── autorg fallback ──────────────────────────────────────────────────────────
# When True: if a frame has no Guinier Rg, try autorg to get one before datgnom.
AUTORG_FALLBACK = True

# When True: also attempt p(r) on frames that failed the next_step filter
# (i.e. Guinier itself failed / was skipped). autorg will be used to get Rg.
# When False: only frames that passed the Guinier next_step filter are processed.
AUTORG_FALLBACK_ALL = False

# ── Guinier table columns ───────────────────────────────────────────────────
FILE_COL       = "file"
RG_COL         = "Rg"
NEXTSTEP_COL   = "next_step"   # set to None if not present
STATUS_COL     = "status"      # set to None if not present
DIAGNOSIS_COL  = "diagnosis"   # set to None if not present
SKIP_COL       = None          # e.g. "skip" if you add this later

# ── Selection rules ─────────────────────────────────────────────────────────
INCLUDE_INSPECT_THEN_PR = True
ALLOWED_NEXT_STEPS = {"p(r)"}
if INCLUDE_INSPECT_THEN_PR:
    ALLOWED_NEXT_STEPS.add("inspect_then_pr")

REQUIRE_VALID_RG = True

# Optional manual overrides (filenames with or without extension)
MANUAL_INCLUDE = []   # e.g. ["subtracted_008_sample_1070_1119_buffer_1_719.dat"]
MANUAL_EXCLUDE = []   # e.g. ["subtracted_004_sample_870_919_buffer_1_719.dat"]

# ── datgnom run settings ────────────────────────────────────────────────────
DEFAULT_SKIP = 0
DATGNOM_TIMEOUT = 300   # seconds
SAVE_STDOUT_STDERR = True

# ── Output names ────────────────────────────────────────────────────────────
MATCHED_TABLE_CSV = OUTPUT_FOLDER / "matched_files_with_rg.csv"
SUMMARY_CSV       = OUTPUT_FOLDER / "datgnom_summary.csv"

# =============================================================================
# Helpers
# =============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize a filename or table entry for robust matching.
    Matching is mostly done on lowercase stem.
    """
    if pd.isna(name):
        return ""

    s = str(name).strip().replace("\\", "/")
    s = s.split("/")[-1].lower()

    # remove known extension if present
    for ext in [".dat", ".txt", ".out", ".csv", ".xlsx", ".xls"]:
        if s.endswith(ext):
            s = s[:-len(ext)]
            break

    # normalize spaces / separators a bit
    s = re.sub(r"\s+", "_", s)
    return s


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Guinier table not found:\n{path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported Guinier table format: {suffix}")


def get_candidate_files(folder: Path, extensions):
    exts = {e.lower() for e in extensions}
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in exts:
            files.append(f)
    return sorted(files)


def build_file_lookups(files):
    """
    Build multiple lookup maps for robust matching.
    """
    exact = {}   # exact lowercase name -> file
    stem = {}    # normalized stem -> list of files

    for f in files:
        exact[f.name.lower()] = f
        key = normalize_name(f.name)
        stem.setdefault(key, []).append(f)

    return exact, stem


def match_table_file_to_actual_file(table_file, exact_lookup, stem_lookup):
    """
    Returns: (matched_file_or_None, match_type, note)
    """
    if pd.isna(table_file) or str(table_file).strip() == "":
        return None, "missing", "empty filename in Guinier table"

    raw = str(table_file).strip()
    exact_key = raw.lower()

    # 1) exact filename match
    if exact_key in exact_lookup:
        return exact_lookup[exact_key], "exact", ""

    # 2) normalized stem match
    stem_key = normalize_name(raw)
    candidates = stem_lookup.get(stem_key, [])

    if len(candidates) == 1:
        return candidates[0], "stem", ""
    elif len(candidates) > 1:
        return None, "ambiguous", "multiple files matched normalized name"
    else:
        return None, "not_found", "no matching SAXS file found in INPUT_FOLDER"


def parse_boolish_manual_list(items):
    """
    Normalize manual include/exclude filename lists to normalized stem keys.
    """
    return {normalize_name(x) for x in items if str(x).strip()}


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def run_autorg(infile: Path, autorg_exe: str, timeout: int = 60) -> dict:
    """
    Run autorg on infile and return a dict with Rg and Rg_err, or NaN if it fails.
    autorg CSV output columns:
      File, Rg, Rg StDev, I(0), I(0) StDev, First point, Last point, Quality, Aggregated
    """
    result = {"Rg": np.nan, "Rg_err": np.nan}
    cmd = [autorg_exe, str(infile), "-f", "csv"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            return result
        lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return result
        parts = lines[1].split(",")
        result["Rg"]     = safe_float(parts[1]) if len(parts) > 1 else np.nan
        result["Rg_err"] = safe_float(parts[2]) if len(parts) > 2 else np.nan
        return result
    except Exception:
        return result



def parse_datgnom_outfile(out_path: Path):
    """
    Parse results from the GNOM .out file (ATSAS 3.x).
    datgnom writes all results here; stdout is empty.
    """
    result = {
        "Dmax": np.nan,
        "Total": np.nan,
        "Rg_guinier_reported": np.nan,
        "Rg_pr": np.nan,
        "Rg_pr_err": np.nan,
        "I0_pr": np.nan,
        "I0_pr_err": np.nan,
    }

    if not out_path.exists():
        return result

    try:
        text = out_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return result

    # Maximum characteristic size = Dmax
    m = re.search(r"Maximum characteristic size:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", text)
    if m:
        result["Dmax"] = safe_float(m.group(1))

    # Total Estimate (quality score)
    m = re.search(r"Total Estimate:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", text)
    if m:
        result["Total"] = safe_float(m.group(1))

    # Reciprocal space Rg (= Guinier Rg as reported by GNOM)
    m = re.search(r"Reciprocal space Rg:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", text)
    if m:
        result["Rg_guinier_reported"] = safe_float(m.group(1))

    # Real space Rg and its error  (format: "value +- error")
    m = re.search(r"Real space Rg:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\+-\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", text)
    if m:
        result["Rg_pr"]     = safe_float(m.group(1))
        result["Rg_pr_err"] = safe_float(m.group(2))

    # Real space I(0) and its error
    m = re.search(r"Real space I\(0\):\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\+-\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", text)
    if m:
        result["I0_pr"]     = safe_float(m.group(1))
        result["I0_pr_err"] = safe_float(m.group(2))

    return result


def ensure_exe_exists(exe_path: str):
    """
    Check whether datgnom executable exists or is available in PATH.
    """
    p = Path(exe_path)
    if p.exists():
        return True
    if shutil.which(exe_path) is not None:
        return True
    return False


# =============================================================================
# Main workflow
# =============================================================================

def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"INPUT_FOLDER not found:\n{INPUT_FOLDER}")

    if not ensure_exe_exists(DATGNOM_EXE):
        raise FileNotFoundError(
            f"DATGNOM executable not found:\n{DATGNOM_EXE}\n"
            f"Check the path or add ATSAS bin to PATH."
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # read Guinier table
    df = read_table(GUINIER_TABLE)

    for col in [FILE_COL, RG_COL]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in Guinier table.")

    # candidate SAXS files
    data_files = get_candidate_files(INPUT_FOLDER, FILE_EXTENSIONS)
    if len(data_files) == 0:
        raise ValueError(f"No candidate SAXS files found in:\n{INPUT_FOLDER}")

    exact_lookup, stem_lookup = build_file_lookups(data_files)

    manual_include_keys = parse_boolish_manual_list(MANUAL_INCLUDE)
    manual_exclude_keys = parse_boolish_manual_list(MANUAL_EXCLUDE)

    matched_rows = []

    # -------------------------------------------------------------------------
    # Match table rows to actual files and decide which rows to process
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():
        table_file = str(row[FILE_COL]).strip() if not pd.isna(row[FILE_COL]) else ""
        table_norm = normalize_name(table_file)

        next_step = str(row[NEXTSTEP_COL]).strip() if (NEXTSTEP_COL and NEXTSTEP_COL in df.columns and not pd.isna(row[NEXTSTEP_COL])) else ""
        status = str(row[STATUS_COL]).strip() if (STATUS_COL and STATUS_COL in df.columns and not pd.isna(row[STATUS_COL])) else ""
        diagnosis = str(row[DIAGNOSIS_COL]).strip() if (DIAGNOSIS_COL and DIAGNOSIS_COL in df.columns and not pd.isna(row[DIAGNOSIS_COL])) else ""

        rg = pd.to_numeric(row[RG_COL], errors="coerce")
        skip_used = DEFAULT_SKIP
        if SKIP_COL and SKIP_COL in df.columns:
            try:
                skip_used = int(row[SKIP_COL])
            except Exception:
                skip_used = DEFAULT_SKIP

        matched_file, match_type, note = match_table_file_to_actual_file(table_file, exact_lookup, stem_lookup)

        # selection logic
        selected = False
        selection_reason = ""

        if table_norm in manual_exclude_keys:
            selected = False
            selection_reason = "manual_exclude"
        elif table_norm in manual_include_keys:
            selected = True
            selection_reason = "manual_include"
        else:
            if NEXTSTEP_COL and NEXTSTEP_COL in df.columns:
                if next_step in ALLOWED_NEXT_STEPS:
                    selected = True
                    selection_reason = f"next_step={next_step}"
                elif AUTORG_FALLBACK_ALL:
                    selected = True
                    selection_reason = f"autorg_fallback_all; next_step={next_step}"
                else:
                    selected = False
                    selection_reason = f"next_step={next_step} not selected"
            else:
                # fallback if next_step column is absent
                selected = True
                selection_reason = "no next_step column; selected by default"

        # Only gate on invalid Rg when autorg fallback is disabled —
        # otherwise autorg will supply the Rg at runtime.
        if REQUIRE_VALID_RG and not np.isfinite(rg) and not AUTORG_FALLBACK:
            selected = False
            if selection_reason:
                selection_reason += "; invalid Rg (autorg fallback disabled)"
            else:
                selection_reason = "invalid Rg (autorg fallback disabled)"

        matched_rows.append({
            "table_file": table_file,
            "matched_file": matched_file.name if matched_file else "",
            "matched_path": str(matched_file) if matched_file else "",
            "match_type": match_type,
            "match_note": note,
            "selected": selected,
            "selection_reason": selection_reason,
            "guinier_Rg": rg,
            "skip_used": skip_used,
            "guinier_status": status,
            "guinier_diagnosis": diagnosis,
            "guinier_next_step": next_step,
        })

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(MATCHED_TABLE_CSV, index=False)

    print(f"Matched file table saved to:\n{MATCHED_TABLE_CSV}\n")

    # -------------------------------------------------------------------------
    # Run datgnom on selected matched rows
    # -------------------------------------------------------------------------
    summary_rows = []

    selected_df = matched_df[matched_df["selected"] == True].copy()

    if len(selected_df) == 0:
        print("No files selected for datgnom.")
        matched_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Empty/selection-only summary saved to:\n{SUMMARY_CSV}")
        return

    for i, row in selected_df.iterrows():
        table_file = row["table_file"]
        matched_path = row["matched_path"]
        match_type = row["match_type"]
        guinier_rg = safe_float(row["guinier_Rg"])
        skip_used = int(row["skip_used"]) if pd.notna(row["skip_used"]) else DEFAULT_SKIP

        # defaults for summary
        datgnom_status = "failed"
        datgnom_returncode = np.nan
        datgnom_stdout_path = ""
        datgnom_stderr_path = ""
        out_file_path = ""
        dmax = np.nan
        total = np.nan
        rg_pr = np.nan
        rg_pr_err = np.nan
        rg_guinier_reported = np.nan
        i0_pr = np.nan
        i0_pr_err = np.nan
        autorg_rg = np.nan
        autorg_rg_err = np.nan
        pr_recommendation = "failed_pr"
        rg_source = "none"
        notes = ""

        # preflight checks
        if match_type in {"ambiguous", "not_found", "missing"} or not matched_path:
            notes = f"Skipping: file match problem ({match_type})"
            summary_rows.append({
                **row.to_dict(),
                "datgnom_status": datgnom_status,
                "datgnom_returncode": datgnom_returncode,
                "Dmax": dmax,
                "Total": total,
                "Rg_pr": rg_pr,
                "Rg_pr_err": rg_pr_err,
                "Rg_guinier_reported": rg_guinier_reported,
                "I0_pr": i0_pr,
                "I0_pr_err": i0_pr_err,
                "output_subfolder": "",
                "output_file": out_file_path,
                "stdout_log": datgnom_stdout_path,
                "stderr_log": datgnom_stderr_path,
                "pr_recommendation": pr_recommendation,
                "notes": notes,
            })
            print(f"[SKIP] {table_file} | {notes}")
            continue

        infile = Path(matched_path)

        # ── Always run autorg for comparison ──────────────────────────────
        autorg_result = run_autorg(infile, AUTORG_EXE)
        autorg_rg     = autorg_result["Rg"]
        autorg_rg_err = autorg_result["Rg_err"]

        # ── Rg source: Guinier table, or autorg fallback ──────────────────
        rg_used = guinier_rg
        if np.isfinite(rg_used):
            rg_source = "guinier"
        elif AUTORG_FALLBACK:
            rg_used = autorg_rg
            if np.isfinite(rg_used):
                rg_source = "autorg"
                print(f"[AUTORG] {infile.name} | no Guinier Rg, using autorg Rg={rg_used:.3f}")
            else:
                rg_source = "none"
        else:
            rg_source = "none"

        if not np.isfinite(rg_used):
            notes = "Skipping: no valid Rg (Guinier failed, autorg also failed or disabled)"
            summary_rows.append({
                **row.to_dict(),
                "datgnom_status": datgnom_status,
                "datgnom_returncode": datgnom_returncode,
                "rg_source": rg_source,
                "rg_used": np.nan,
                "autorg_Rg": autorg_rg,
                "autorg_Rg_err": autorg_rg_err,
                "Dmax": dmax,
                "Total": total,
                "Rg_pr": rg_pr,
                "Rg_pr_err": rg_pr_err,
                "Rg_guinier_reported": rg_guinier_reported,
                "I0_pr": i0_pr,
                "I0_pr_err": i0_pr_err,
                "output_subfolder": "",
                "output_file": out_file_path,
                "stdout_log": datgnom_stdout_path,
                "stderr_log": datgnom_stderr_path,
                "pr_recommendation": pr_recommendation,
                "notes": notes,
            })
            print(f"[SKIP] {table_file} | {notes}")
            continue
        stem = infile.stem
        subfolder = OUTPUT_FOLDER / stem
        subfolder.mkdir(parents=True, exist_ok=True)

        out_file = subfolder / f"{stem}.out"
        stdout_log = subfolder / "datgnom_stdout.txt"
        stderr_log = subfolder / "datgnom_stderr.txt"

        # datgnom cannot handle output paths with spaces, so write to a
        # space-free temp file and move it into the final destination.
        tmp_dir = Path(tempfile.gettempdir())
        tmp_out = tmp_dir / f"datgnom_{stem}.out"

        first_point = skip_used + 1  # datgnom uses 1-based --first
        cmd = [
            str(DATGNOM_EXE),
            str(infile),
            "--rg", f"{rg_used:.6f}",
            "--first", str(first_point),
            "-o", str(tmp_out),
        ]

        print(f"[RUN] {infile.name} | Rg={rg_used:.3f} (source={rg_source}) | skip={skip_used}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DATGNOM_TIMEOUT,
                check=False
            )

            datgnom_returncode = proc.returncode

            if SAVE_STDOUT_STDERR:
                stdout_log.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
                stderr_log.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
                datgnom_stdout_path = str(stdout_log)
                datgnom_stderr_path = str(stderr_log)

            # Move temp .out file to final destination
            if tmp_out.exists():
                shutil.move(str(tmp_out), str(out_file))

            parsed_outfile = parse_datgnom_outfile(out_file)

            dmax = parsed_outfile["Dmax"]
            total = parsed_outfile["Total"]
            rg_pr = parsed_outfile["Rg_pr"]
            rg_pr_err = parsed_outfile["Rg_pr_err"]
            rg_guinier_reported = parsed_outfile["Rg_guinier_reported"]
            i0_pr = parsed_outfile["I0_pr"]
            i0_pr_err = parsed_outfile["I0_pr_err"]
            out_file_path = str(out_file) if out_file.exists() else ""

            if proc.returncode == 0 and np.isfinite(dmax) and np.isfinite(rg_pr):
                datgnom_status = "ok"
                pr_recommendation = "good_pr"

                # mild warning if output looks incomplete
                if not out_file.exists():
                    pr_recommendation = "inspect_pr_manually"
                    notes = "datgnom returned success but .out file not found"
                elif not np.isfinite(total):
                    pr_recommendation = "inspect_pr_manually"
                    notes = "Dmax and Rg parsed, but Total quality not parsed"
            else:
                datgnom_status = "failed"
                pr_recommendation = "failed_pr"
                notes = (proc.stderr or proc.stdout or "datgnom failed").strip()

            summary_rows.append({
                **row.to_dict(),
                "datgnom_status": datgnom_status,
                "datgnom_returncode": datgnom_returncode,
                "rg_source": rg_source,
                "rg_used": rg_used,
                "autorg_Rg": autorg_rg,
                "autorg_Rg_err": autorg_rg_err,
                "Dmax": dmax,
                "Total": total,
                "Rg_pr": rg_pr,
                "Rg_pr_err": rg_pr_err,
                "Rg_guinier_reported": rg_guinier_reported,
                "I0_pr": i0_pr,
                "I0_pr_err": i0_pr_err,
                "output_subfolder": str(subfolder),
                "output_file": out_file_path,
                "stdout_log": datgnom_stdout_path,
                "stderr_log": datgnom_stderr_path,
                "pr_recommendation": pr_recommendation,
                "notes": notes,
            })

            if datgnom_status == "ok":
                print(f"     -> OK | Dmax={dmax:.3f} | Rg_pr={rg_pr:.3f} | Total={total if np.isfinite(total) else np.nan}")
            else:
                print(f"     -> FAILED | {notes[:150]}")

        except subprocess.TimeoutExpired:
            notes = f"datgnom timed out after {DATGNOM_TIMEOUT} s"
            summary_rows.append({
                **row.to_dict(),
                "datgnom_status": "failed",
                "datgnom_returncode": np.nan,
                "rg_source": rg_source,
                "rg_used": rg_used,
                "autorg_Rg": autorg_rg,
                "autorg_Rg_err": autorg_rg_err,
                "Dmax": np.nan,
                "Total": np.nan,
                "Rg_pr": np.nan,
                "Rg_pr_err": np.nan,
                "Rg_guinier_reported": np.nan,
                "I0_pr": np.nan,
                "I0_pr_err": np.nan,
                "output_subfolder": str(subfolder),
                "output_file": "",
                "stdout_log": str(stdout_log) if stdout_log.exists() else "",
                "stderr_log": str(stderr_log) if stderr_log.exists() else "",
                "pr_recommendation": "failed_pr",
                "notes": notes,
            })
            print(f"     -> FAILED | {notes}")

        except Exception as e:
            notes = f"unexpected error: {e}"
            summary_rows.append({
                **row.to_dict(),
                "datgnom_status": "failed",
                "datgnom_returncode": np.nan,
                "rg_source": rg_source,
                "rg_used": rg_used,
                "autorg_Rg": autorg_rg,
                "autorg_Rg_err": autorg_rg_err,
                "Dmax": np.nan,
                "Total": np.nan,
                "Rg_pr": np.nan,
                "Rg_pr_err": np.nan,
                "Rg_guinier_reported": np.nan,
                "I0_pr": np.nan,
                "I0_pr_err": np.nan,
                "output_subfolder": str(subfolder),
                "output_file": "",
                "stdout_log": "",
                "stderr_log": "",
                "pr_recommendation": "failed_pr",
                "notes": notes,
            })
            print(f"     -> FAILED | {notes}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nDone.")
    print(f"Summary saved to:\n{SUMMARY_CSV}")

    # compact terminal summary
    cols_to_show = [
        "table_file", "matched_file", "guinier_Rg", "rg_source", "rg_used",
        "datgnom_status", "Dmax", "Rg_pr", "pr_recommendation"
    ]
    cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
    print("\nSummary:")
    print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()