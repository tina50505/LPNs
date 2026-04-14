from pathlib import Path
import re
import colorsys
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CHANGE ONLY THESE
# =========================
FOLDER = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS\Tina OFFLINE SAXS")

EXTENSION  = ".dat"    # e.g. ".dat" or ".txt"
USE_LOGLOG = True

# Folder where plots are saved. Set to None to save next to the data folder.
SAVE_FOLDER = FOLDER

# Pattern to strip from the filename stem to get a clean label.
# E.g. "_scan-12345_eiger_sub" → stripped. Set to None to keep full stem.
SCAN_STRIP_PATTERN = r"_scan[-_].*$"
# =========================


# ── helpers ──────────────────────────────────────────────────────────────────

def select_files_interactively(folder: Path, extension: str) -> list[Path]:
    """Detect all matching files, print a numbered list, and let the user pick."""
    all_paths = sorted(folder.glob(f"*{extension}"))
    if not all_paths:
        print(f"No {extension} files found in:\n  {folder}")
        raise SystemExit(1)

    print(f"\nFound {len(all_paths)} file(s) in:\n  {folder}\n")
    for i, p in enumerate(all_paths, 1):
        print(f"  [{i:>3}]  {p.stem}")

    print("\nEnter numbers to plot (e.g. 1,3,5-8), or press Enter for ALL:")
    raw = input("> ").strip()

    if not raw:
        return all_paths

    selected = []
    for token in re.split(r"[,;\s]+", raw):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            for idx in range(lo, hi + 1):
                if 1 <= idx <= len(all_paths):
                    selected.append(all_paths[idx - 1])
                else:
                    print(f"[!] Index {idx} out of range, skipped.")
        elif token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(all_paths):
                selected.append(all_paths[idx - 1])
            else:
                print(f"[!] Index {idx} out of range, skipped.")
        else:
            print(f"[!] Unrecognised token '{token}', skipped.")

    # preserve order, deduplicate
    seen = set()
    unique = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def clean_stem(stem: str) -> str:
    """Strip scan suffix and return the clean sample label."""
    if SCAN_STRIP_PATTERN:
        stem = re.sub(SCAN_STRIP_PATTERN, "", stem, flags=re.IGNORECASE)
    return stem.replace("_", " ").strip()


def read_atsas_file(filepath: Path):
    """
    Read ATSAS-style whitespace-separated file.
    Returns (q, I, sigma) where sigma is None if no 3rd column found.
    """
    q_vals, i_vals, s_vals = [], [], []
    has_sigma = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                q_vals.append(float(parts[0]))
                i_vals.append(float(parts[1]))
                if len(parts) >= 3:
                    s_vals.append(float(parts[2]))
                    has_sigma = True
                else:
                    s_vals.append(np.nan)
            except ValueError:
                continue

    q = np.array(q_vals)
    I = np.array(i_vals)
    sigma = np.array(s_vals) if has_sigma else None
    return q, I, sigma


# ── auto style ────────────────────────────────────────────────────────────────

_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]

# One base hue per last token (buffer/condition). Add more hues as needed.
_BASE_HUES = [0.60, 0.02, 0.35, 0.10, 0.75, 0.50]   # blue, red, green, orange, purple, teal


def build_style_map(labels: list[str]) -> dict[str, tuple[str, str]]:
    """
    Assign (color, marker) from label structure:
      marker → first token          (sample group,   e.g. F3 / F3C)
      hue    → all-but-last tokens  (sample identity, e.g. F3 empty / F3 mRNA)
      shade  → last token           (buffer/condition, e.g. PBS / TRIS)
                                     — shade index is global so PBS is always
                                       the same shade across all hue groups.
    Fully general: works for any number of groups / conditions.
    """
    unique = list(dict.fromkeys(labels))

    def sample_key(lbl):
        parts = lbl.split()
        return " ".join(parts[:-1]) if len(parts) > 1 else lbl

    first_tokens   = list(dict.fromkeys(lbl.split()[0]  for lbl in unique))
    sample_keys    = list(dict.fromkeys(sample_key(lbl) for lbl in unique))
    last_tokens    = list(dict.fromkeys(lbl.split()[-1] for lbl in unique))

    marker_map = {w:  _MARKERS[i   % len(_MARKERS)]   for i, w in enumerate(first_tokens)}
    hue_map    = {sk: _BASE_HUES[i % len(_BASE_HUES)] for i, sk in enumerate(sample_keys)}
    shade_map  = {w:  i for i, w in enumerate(last_tokens)}   # consistent across all groups

    n_shades = max(len(last_tokens), 1)
    color_map = {}
    for lbl in unique:
        sk  = sample_key(lbl)
        hue = hue_map[sk]
        j   = shade_map[lbl.split()[-1]]
        value = 0.50 + 0.40 * j / max(n_shades - 1, 1)   # 0.50 (dark) → 0.90 (light)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, value)
        color_map[lbl] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    return {lbl: (color_map[lbl], marker_map[lbl.split()[0]]) for lbl in unique}


# ── main ──────────────────────────────────────────────────────────────────────

paths = select_files_interactively(FOLDER, EXTENSION)
missing = [p for p in paths if not p.exists()]
paths   = [p for p in paths if p.exists()]

if missing:
    for p in missing:
        print(f"[!] File not found: {p}")

if not paths:
    print("No files to plot.")
    raise SystemExit(1)

print("\nPlot error bars? [y/N]: ", end="")
PLOT_ERRORS = input().strip().lower() in ("y", "yes")

labels = [clean_stem(p.stem) for p in paths]
style_map = build_style_map(labels)

plt.figure(figsize=(8, 6))

for filepath, label in zip(paths, labels):
    q, I, sigma = read_atsas_file(filepath)
    color, marker = style_map[label]

    if USE_LOGLOG:
        mask = (q > 0) & (I > 0)
        if PLOT_ERRORS and sigma is not None:
            plt.errorbar(
                q[mask], I[mask],
                yerr=sigma[mask] if sigma is not None else None,
                fmt=marker, color=color, markersize=4,
                linewidth=1.2, elinewidth=0.6, capsize=2,
                label=label,
            )
            plt.xscale("log")
            plt.yscale("log")
        else:
            plt.loglog(q[mask], I[mask],
                       marker=marker, markersize=4,
                       linewidth=1.2, color=color, label=label)
    else:
        if PLOT_ERRORS and sigma is not None:
            plt.errorbar(
                q, I, yerr=sigma,
                fmt=marker, color=color, markersize=4,
                linewidth=1.2, elinewidth=0.6, capsize=2,
                label=label,
            )
        else:
            plt.plot(q, I, marker=marker, markersize=4,
                     linewidth=1.2, color=color, label=label)

plt.xlabel(r"q ($\mathrm{\AA}^{-1}$)")
plt.ylabel(r"I(q) ($\mathrm{cm}^{-1}$)")
plt.legend()
plt.tight_layout()

save_dir = Path(SAVE_FOLDER) if SAVE_FOLDER else FOLDER
print(f"\nSave plot as (filename without extension, or Enter to skip):")
print(f"  folder: {save_dir}")
save_name = input("> ").strip()
if save_name:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{save_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

plt.show()
