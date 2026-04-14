"""
DLS Plotter - correlogram and/or intensity size distribution
------------------------------------------------------------
1. Set DATA_FOLDER to the folder containing your .txt files
2. Set SAVE_FOLDER to where you want figures saved (or None to just display)
3. Run: python dls_plot.py

File naming convention expected (case-insensitive):
    *Correlogram*    -> correlogram file
    *Size Dist*      -> size distribution file
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =============================================================================
#  SET YOUR PATHS HERE
# =============================================================================

DATA_FOLDER = r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\DLS batch samples data"   # folder with the .txt files

SAVE_FOLDER = DATA_FOLDER                    # where to save figures, e.g.:
                                              # r"C:\path\to\save\figures"
                                              # set to None to only display

SAVE_FORMAT = "png"   # "png", "svg", or "pdf"
SAVE_DPI    = 300

# =============================================================================


# -- file detection -----------------------------------------------------------

def find_dls_files(folder):
    txt = glob.glob(os.path.join(folder, "*.txt"))
    corr  = [f for f in txt if "correlogram" in os.path.basename(f).lower()]
    sdist = [f for f in txt if "size dist"   in os.path.basename(f).lower()]
    return corr, sdist


# -- parsing ------------------------------------------------------------------

def parse_file(filepath):
    """Return (x_array, {record_label: y_array}) or (None, None) on failure."""
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if not lines:
        return None, None

    header = lines[0].rstrip("\n").split("\t")
    record_labels = [h.strip() for h in header[1:] if h.strip()]

    rows = []
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if not parts or not parts[0].strip():
            continue
        try:
            rows.append([float(p) if p.strip() else np.nan for p in parts])
        except ValueError:
            continue

    if not rows:
        return None, None

    arr = np.array(rows)
    x = arr[:, 0]
    records = {
        label: arr[:, i + 1]
        for i, label in enumerate(record_labels)
        if i + 1 < arr.shape[1]
    }
    return x, records


def parse_record_label(label):
    """
    'Record 4: F3 empty 1'  ->  ('F3 empty', '1')
    Falls back gracefully if format differs.
    """
    if ":" in label:
        after = label.split(":", 1)[1].strip()
        parts = after.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0].strip(), parts[1]
        return after, ""
    return label, ""


def collect_samples(files):
    """Return sorted unique sample names across all given files."""
    names = set()
    for f in files:
        _, records = parse_file(f)
        if records:
            for lbl in records:
                name, _ = parse_record_label(lbl)
                names.add(name)
    return sorted(names)


# -- plotting -----------------------------------------------------------------

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _add_sample_to_ax(ax, x, records, sample_name, color_offset=0):
    added = 0
    for lbl, y in records.items():
        name, repeat = parse_record_label(lbl)
        if name != sample_name:
            continue
        ax.plot(x, y,
                label=f"{name} {repeat}".strip(),
                color=COLORS[(color_offset + added) % len(COLORS)],
                linewidth=1.4)
        added += 1
    return added


def plot_correlogram(files, sample_name, ax):
    for f in files:
        x, records = parse_file(f)
        if records:
            _add_sample_to_ax(ax, x, records, sample_name)
    ax.set_xscale("log")
    ax.set_xlabel("Lag time (us)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title(f"{sample_name} - Correlogram")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")


def plot_sizedist(files, sample_name, ax):
    for f in files:
        x, records = parse_file(f)
        if records:
            _add_sample_to_ax(ax, x, records, sample_name)
    ax.set_xscale("log")
    ax.set_xlabel("Dh (nm)")
    ax.set_ylabel("Intensity %")
    ax.set_title(f"{sample_name} - Size distribution intensity")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


# -- interactive CLI ----------------------------------------------------------

def ask_choice(prompt, options):
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("  0. All")
    raw = input(prompt).strip()
    if raw == "0" or raw == "":
        return list(options)
    chosen = []
    for token in raw.replace(",", " ").split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(options):
                chosen.append(options[idx])
        except ValueError:
            pass
    return chosen if chosen else list(options)


# -- main ---------------------------------------------------------------------

def main():
    folder = DATA_FOLDER
    if not os.path.isdir(folder):
        sys.exit(
            f"Folder not found:\n  {folder}\n"
            "Please update DATA_FOLDER at the top of the script."
        )

    if SAVE_FOLDER is not None:
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    corr_files, size_files = find_dls_files(folder)
    has_corr = bool(corr_files)
    has_size = bool(size_files)

    if not has_corr and not has_size:
        sys.exit("No correlogram or size-distribution files found in that folder.")

    # choose plot type
    if has_corr and has_size:
        print("\nWhat would you like to plot?")
        print("  1. Correlogram only")
        print("  2. Size distribution only")
        print("  3. Both")
        pt = input("Select [1/2/3, default 3]: ").strip() or "3"
        do_corr = pt in ("1", "3")
        do_size = pt in ("2", "3")
    else:
        do_corr = has_corr
        do_size = has_size
        kind = "correlogram" if has_corr else "size distribution"
        print(f"\nOnly {kind} files found - plotting those.")

    # choose samples
    relevant = (corr_files if do_corr else []) + (size_files if do_size else [])
    samples = collect_samples(relevant)

    if not samples:
        sys.exit("Could not extract any sample names from the files.")

    print("\nAvailable samples (comma-separated numbers, or 0 for all):")
    selected = ask_choice("Select samples: ", samples)

    if not selected:
        sys.exit("No samples selected.")

    # generate figures
    plt.rcParams.update({"font.size": 11})

    for sample in selected:
        ncols = int(do_corr) + int(do_size)
        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.8),
                                 constrained_layout=True)
        if ncols == 1:
            axes = [axes]

        ax_idx = 0
        if do_corr:
            plot_correlogram(corr_files, sample, axes[ax_idx])
            ax_idx += 1
        if do_size:
            plot_sizedist(size_files, sample, axes[ax_idx])

        fig.suptitle(sample, fontsize=13, fontweight="bold")

        if SAVE_FOLDER is not None:
            safe = sample.replace(" ", "_").replace("/", "-")
            tag  = ("correlogram_sizedist" if (do_corr and do_size)
                    else "correlogram" if do_corr else "sizedist")
            fname = os.path.join(SAVE_FOLDER, f"{safe}_{tag}.{SAVE_FORMAT}")
            fig.savefig(fname, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"Saved: {fname}")

    plt.show()


if __name__ == "__main__":
    main()