from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CHANGE ONLY THESE
# =========================
FOLDER = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\20260306 CoSAXS\Tina OFFLINE SAXS")

FILES_TO_PLOT = [
    "F3_empty_PBS_scan-84638_eiger_sub",
    "F3_empty_TRIS_scan-84619_eiger_sub",
    "F3_mRNA_PBS_scan-84641_eiger_sub",
    "F3_mRNA_TRIS_scan-84626_eiger_sub",
    "F3C_empty_PBS_scan-84644_eiger_sub",
    "F3C_empty_TRIS_scan-84622_eiger_sub",
    "F3C_mRNA_PBS_scan-84647_eiger_sub",
    "F3C_mRNA_TRIS_scan-84629_eiger_sub",
]

EXTENSION = ".dat"      # e.g. ".dat" or ".txt"
USE_LOGLOG = True
# =========================


def make_label_from_filename(filepath):
    name = filepath.stem
    name = name.split("_scan")[0]
    parts = name.split("_")
    return " ".join(parts)


def read_atsas_file(filepath):
    q_vals = []
    i_vals = []

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
                q_vals.append(q)
                i_vals.append(I)
            except ValueError:
                continue

    return np.array(q_vals), np.array(i_vals)


def get_style(filepath):
    name = filepath.stem.split("_scan")[0]
    parts = name.split("_")

    # Marker: F3 vs F3C
    if parts[0] == "F3":
        marker = "o"   # circle
    elif parts[0] == "F3C":
        marker = "s"   # square
    else:
        marker = "o"

    # Buffer: PBS vs TRIS
    buffer_type = parts[-1]

    # Sample key without buffer, to group matching PBS/TRIS pairs
    sample_key = "_".join(parts[:-1])

    # Base colors for each pair
    color_map = {
        "F3_empty": {"PBS": "#9ecae1", "TRIS": "#3182bd"},
        "F3_mRNA": {"PBS": "#fdae6b", "TRIS": "#e6550d"},
        "F3C_empty": {"PBS": "#a1d99b", "TRIS": "#31a354"},
        "F3C_mRNA": {"PBS": "#d4b9da", "TRIS": "#756bb1"},
    }

    color = color_map.get(sample_key, {"PBS": "lightgray", "TRIS": "gray"}).get(buffer_type, "black")

    return marker, color


plt.figure(figsize=(8, 6))

for name in FILES_TO_PLOT:
    filepath = FOLDER / f"{name}{EXTENSION}"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        continue

    q, I = read_atsas_file(filepath)
    label = make_label_from_filename(filepath)
    marker, color = get_style(filepath)

    if USE_LOGLOG:
        mask = (q > 0) & (I > 0)
        plt.loglog(
            q[mask], I[mask],
            marker=marker,
            markersize=4,
            linewidth=1.2,
            color=color,
            label=label
        )
    else:
        plt.plot(
            q, I,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            color=color,
            label=label
        )

plt.xlabel(r"q ($\mathrm{\AA}^{-1}$)")
plt.ylabel(r"I(q) ($\mathrm{cm}^{-1}$)")
# plt.title("SAXS curves")
plt.legend()
plt.tight_layout()
plt.show()
