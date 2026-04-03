"""
AF4 Chromatogram Plotter
------------------------
Reads an AF4 Excel export, detects samples from the header row,
then lets you choose which samples and channels (UV / Rh / both) to plot.

Usage:
    python af4_plot.py                        # prompts for file path
    python af4_plot.py data.xlsx              # pass file directly
"""

import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.signal import find_peaks


# ── 1. Header parsing ────────────────────────────────────────────────────────

def parse_samples(xlsx_path: str) -> dict:
    """
    Read the first header row and return a dict:
        { sample_name: { 'uv': (time_col, uv_col),
                         'rh': (time_col, rh_col) } }

    Column naming convention expected:
        "1126 - F3 empty 200uL Vcf175 Vx250g Vf000 (UV)"
        "1126 - F3 empty 200uL Vcf175 Vx250g Vf000 (Rh(Q))"
        "time (min)"  /  "time (min).1"  /  "time (min).2"  ...
    """
    df_head = pd.read_excel(xlsx_path, nrows=0)
    cols = list(df_head.columns)

    # Pair each data column with its preceding time column
    # The sheet repeats:  time | UV | time | Rh | time | UV | time | Rh ...
    # pandas renames duplicate "time (min)" → "time (min).1", ".2", etc.
    time_pattern = re.compile(r'^time \(min\)', re.IGNORECASE)
    uv_pattern   = re.compile(r'\(UV\)\s*$',    re.IGNORECASE)
    rh_pattern   = re.compile(r'\(Rh',          re.IGNORECASE)

    # Walk columns; keep a running pointer to the most recent time column
    last_time = None
    col_map = {}   # col_name -> associated time_col

    for col in cols:
        if time_pattern.match(col):
            last_time = col
        elif (uv_pattern.search(col) or rh_pattern.search(col)) and last_time:
            col_map[col] = last_time

    # Extract a human-readable sample name from a column header.
    # E.g. "1126 - F3 empty 200uL Vcf175 Vx250g Vf000 (UV)"
    #   → strip leading run-number "1126 - ", strip trailing " (UV)" / " (Rh(Q))"
    #   → strip volume/param tokens like "200uL Vcf175 Vx250g Vf000"
    def extract_name(col: str) -> str:
        s = re.sub(r'^\d+\s*-\s*', '', col)           # remove "1126 - "
        s = re.sub(r'\s*\(UV\)\s*$', '', s, flags=re.I)
        s = re.sub(r'\s*\(Rh.*?\)\s*$', '', s, flags=re.I)
        # remove parameter tokens like "Vcf175", "Vx250g", "Vf000", "200uL"
        tokens = s.split()
        tokens = [t for t in tokens if not re.fullmatch(r'[A-Za-z]{2,}\d+[A-Za-z]*', t)
                                    and not re.fullmatch(r'\d+[A-Za-z]+', t)]
        return ' '.join(tokens)

    def display_name(name: str) -> str:
        """Convert sample name to clean display format (remove extra spaces)."""
        return ' '.join(name.split())

    # Group UV and Rh columns by their sample name
    samples = {}
    for col, time_col in col_map.items():
        name = extract_name(col)
        samples.setdefault(name, {})
        if uv_pattern.search(col):
            samples[name]['uv'] = (time_col, col)
        elif rh_pattern.search(col):
            samples[name]['rh'] = (time_col, col)

    return samples


# ── 2. User interaction ───────────────────────────────────────────────────────

def pick_samples(samples: dict) -> list:
    names = sorted(samples.keys())
    print("\nSamples found in file:")
    for i, n in enumerate(names, 1):
        channels = ' + '.join(k.upper() for k in samples[n])
        print(f"  [{i}] {n}  ({channels})")

    print("\nEnter sample numbers to plot (comma-separated), or press Enter for all:")
    raw = input("  > ").strip()
    if not raw:
        return names
    chosen = []
    for tok in raw.split(','):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= len(names):
            chosen.append(names[int(tok) - 1])
        else:
            print(f"  Skipping unrecognised token: '{tok}'")
    return chosen if chosen else names


def pick_channel(samples: dict, chosen_samples: list) -> str:
    # Determine which channels are actually available across chosen samples
    has_uv = any('uv' in samples[s] for s in chosen_samples)
    has_rh = any('rh' in samples[s] for s in chosen_samples)

    options = []
    if has_uv:
        options.append('uv')
    if has_rh:
        options.append('rh')
    if has_uv and has_rh:
        options.append('both')

    if len(options) == 1:
        return options[0]

    print("\nWhich channel(s) to plot?")
    for i, o in enumerate(options, 1):
        label = {'uv': 'UV 260 nm', 'rh': 'Rh (hydrodynamic radius)', 'both': 'Both'}[o]
        print(f"  [{i}] {label}")

    while True:
        raw = input("  > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        # also accept text
        if raw.lower() in options:
            return raw.lower()
        print("  Please enter a number from the list.")


# ── 3. Plotting ───────────────────────────────────────────────────────────────

def load_series(xlsx_path: str, samples: dict, chosen: list) -> dict:
    """Load only the columns we need."""
    needed_cols = set()
    for name in chosen:
        for ch in ('uv', 'rh'):
            if ch in samples[name]:
                t_col, d_col = samples[name][ch]
                needed_cols.update([t_col, d_col])

    df = pd.read_excel(xlsx_path)
    return df


def plot(df, samples: dict, chosen: list, channel: str, xlsx_path: str):
    channels_to_plot = ['uv', 'rh'] if channel == 'both' else [channel]
    n_panels = len(channels_to_plot)

    # Color cycle
    colors = cm.tab10(np.linspace(0, 0.9, len(chosen)))

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(10, 4 * n_panels),
                             sharex=False,
                             squeeze=False)

    y_labels = {
        'uv': 'UV absorbance (AU at 260 nm)',
        'rh': 'Hydrodynamic radius Rh (nm)',
    }

    def display_name(name: str) -> str:
        """Convert sample name to clean display format."""
        return ' '.join(name.split())

    for panel_i, ch in enumerate(channels_to_plot):
        ax = axes[panel_i][0]
        for color, name in zip(colors, chosen):
            if ch not in samples[name]:
                continue
            t_col, d_col = samples[name][ch]
            sub = df[[t_col, d_col]].dropna()
            sub.columns = ['time', 'value']
            if ch == 'uv':
                sub['value'] = sub['value'] * 1000  # AU → mAU
            
            label = display_name(name)
            
            # For UV, detect peak and add it to label
            if ch == 'uv':
                # Find peaks in the data
                peaks, _ = find_peaks(sub['value'].values, height=0)
                if len(peaks) > 0:
                    # Get the highest peak
                    peak_idx = peaks[np.argmax(sub['value'].values[peaks])]
                    peak_time = sub['time'].iloc[peak_idx]
                    peak_value = sub['value'].iloc[peak_idx]
                    label = f"{label} (peak: {peak_time:.2f} min)"
                    
                    # Plot dotted vertical line at peak
                    ax.axvline(peak_time, color=color, linestyle='--', alpha=0.6, linewidth=1.0)
            
            ax.plot(sub['time'], sub['value'],
                    label=label, color=color, linewidth=1.4)

        ax.set_xlabel('Elution time (min)')
        lbl = y_labels[ch]
        if ch == 'uv':
            lbl = 'UV absorbance (260 nm) [a.u.]'
        ax.set_ylabel(lbl)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    title_samples = ', '.join(chosen)
    fig.suptitle(f'AF4 chromatogram - {title_samples}', fontsize=11, y=1.01)
    plt.tight_layout()

    out = xlsx_path.replace('.xlsx', '_af4_plot.png').replace('.xls', '_af4_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out}")
    plt.show()


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main():
    xlsx_path = r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\AF4 UV and Rh on fractions\Kopija datoteke Chromatograms UV260 and Rh.xlsx"

    print(f"\nReading: {xlsx_path}")
    samples = parse_samples(xlsx_path)

    if not samples:
        print("No samples detected. Check that the file matches the expected format.")
        sys.exit(1)

    print(f"\n✓ Found {len(samples)} samples:")
    chosen   = pick_samples(samples)
    channel  = pick_channel(samples, chosen)

    print(f"\nLoading data for: {', '.join(chosen)}  |  channel: {channel.upper()}")
    df = load_series(xlsx_path, samples, chosen)
    plot(df, samples, chosen, channel, xlsx_path)


if __name__ == '__main__':
    main()