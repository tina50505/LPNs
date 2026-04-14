from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# =========================================================
# CHANGE THESE SETTINGS
# =========================================================
FOLDER = Path(r"C:\Users\tjurk\Documents\DTU\Engineering Physics\GRADIVO\Thesis\Analysis\DATA\CoSAXS data MARCH\20260306 CoSAXS\process\Tina ONLINE AF4 SAXS\F3C_emtpy_TRIS_scan-84557_shotsE")

# Allowed file extensions
EXTENSIONS = [".dat", ".txt", ""]

# Number of frames per exported sample curve
WINDOW_SIZE = 50


# Step between windows
# 50 = non-overlapping blocks: 1-50, 51-100, ...
# 1  = sliding windows: 1-50, 2-51, 3-52, ...
STEP = 50

# If False, leftover sample frames smaller than WINDOW_SIZE are ignored
# If True, last smaller block is also exported
EXPORT_REMAINDER = True

# Optional q-range for making the intensity-vs-frame trace
# Set both to None to use all q values
QMIN = None
QMAX = None

# ---------------------------------------------------------
# Selection mode: "slider", "manual", or "auto"
#
#   "slider" — drag to select regions interactively
#   "manual" — type frame numbers in the terminal
#   "auto"   — automatically detect flat buffer baseline and peak
#
# After "auto" and "manual" a confirmation plot is shown.
# Press Enter to accept, r to redo with the slider.
# --------------------------------------------------------
SELECTION_MODE = "auto"

# Auto-detection parameters (only used when SELECTION_MODE = "auto")
AUTO_BASELINE_FRACTION = 0.15   # fraction of initial frames used to estimate the baseline 0.15 = first 15% of frames are used for baseline estimation; increase to use more frames, decrease to use fewer
AUTO_PEAK_FRACTION     = 0.005  # threshold = baseline + AUTO_PEAK_FRACTION * (peak_max - baseline)
                                # 0.005 → threshold sits 0.5% of the way from baseline to peak top
                                # increase to start the sample region later, decrease to start earlier
AUTO_SMOOTH_WINDOW     = 30     # frames to smooth over before thresholding (suppresses noise
                                # spikes so a low AUTO_PEAK_FRACTION doesn't false-trigger)
                                # replaces each frame's value with the average of the 30 frames around it before applying the threshold

# Buffer spike rejection (applies to ALL selection modes)
BUFFER_SPIKE_SIGMA = 5.0        # buffer frames above median + N*std of the buffer are excluded
                                # lower = stricter rejection; set to None to disable
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


def read_curve(filepath):
    q_vals, i_vals, err_vals = [], [], []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                q_vals.append(float(parts[0]))
                i_vals.append(float(parts[1]))
                err_vals.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(q_vals), np.array(i_vals), np.array(err_vals)


def average_intensity_for_trace(filepath, qmin=None, qmax=None):
    q, I, err = read_curve(filepath)
    mask = np.isfinite(q) & np.isfinite(I) & (I > 0)
    if qmin is not None:
        mask &= (q >= qmin)
    if qmax is not None:
        mask &= (q <= qmax)
    n = np.sum(mask)
    if n == 0:
        return np.nan, np.nan
    I_mean   = np.mean(I[mask])
    err_mean = np.sqrt(np.sum(err[mask]**2)) / n   # propagated error of the mean over q
    return I_mean, err_mean


def average_curves(file_list): #used for both buffer and each sample block
    q_ref = None
    all_I, all_err = [], []
    for fp in file_list:
        q, I, err = read_curve(fp)
        if q_ref is None:
            q_ref = q
        else:
            if not np.allclose(q, q_ref):
                raise ValueError(f"q grid mismatch in file: {fp.name}")
        all_I.append(I)
        all_err.append(err)
    all_I  = np.array(all_I)
    all_err = np.array(all_err)
    I_avg   = np.mean(all_I, axis=0) # I̅(q) = (1/N) Σ Iₖ(q)
    err_avg = np.sqrt(np.sum(all_err**2, axis=0)) / len(file_list) # σ_avg = √(Σσₖ²) / N
    return q_ref, I_avg, err_avg


def save_curve(filepath, q, I, err, header_lines=None):
    with open(filepath, "w", encoding="utf-8") as f:
        if header_lines is not None:
            for line in header_lines:
                f.write(f"# {line}\n")
        f.write("# q I errors\n")
        for qv, iv, ev in zip(q, I, err):
            f.write(f"{qv:.16e} {iv:.16e} {ev:.16e}\n")


# ---------------------------------------------------------
# Selection functions
# ---------------------------------------------------------

def select_via_slider(frame_numbers, trace):
    """
    Interactive span-selector.
    First drag = buffer (red), second drag = sample (green).
    r = reset, Enter = confirm.
    Returns (buffer_range, sample_range) or (None, None).
    """
    buffer_range    = [None, None]
    sample_range    = [None, None]
    selection_stage = ["buffer"]
    buffer_patch    = [None]
    sample_patch    = [None]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(top=0.88)
    ax.plot(frame_numbers, trace, color="steelblue", lw=1.1)
    ax.set_xlabel("frame nr.")
    ax.set_ylabel("average intensity")
    ax.set_title("Select buffer first, then select full sample region")

    info_text  = fig.text(0.02, 0.95,
        "Drag once for BUFFER, then drag once for FULL SAMPLE REGION. Press Enter when done.",
        fontsize=11)
    range_text = fig.text(0.02, 0.91,
        "Buffer: none    Sample region: none", fontsize=11)

    def update_text():
        btxt = "none" if buffer_range[0] is None else f"{buffer_range[0]}\u2013{buffer_range[1]}"
        stxt = "none" if sample_range[0] is None else f"{sample_range[0]}\u2013{sample_range[1]}"
        range_text.set_text(f"Buffer: {btxt}    Sample region: {stxt}")
        fig.canvas.draw_idle()

    def onselect(xmin, xmax):
        x1 = max(frame_numbers[0],  int(round(min(xmin, xmax))))
        x2 = min(frame_numbers[-1], int(round(max(xmin, xmax))))

        if selection_stage[0] == "buffer":
            buffer_range[0], buffer_range[1] = x1, x2
            if buffer_patch[0] is not None:
                buffer_patch[0].remove()
            buffer_patch[0] = ax.axvspan(x1, x2, color="red", alpha=0.25)
            selection_stage[0] = "sample"
            info_text.set_text("Now drag to select the FULL SAMPLE REGION. Press Enter when done.")

        elif selection_stage[0] == "sample":
            sample_range[0], sample_range[1] = x1, x2
            if sample_patch[0] is not None:
                sample_patch[0].remove()
            sample_patch[0] = ax.axvspan(x1, x2, color="green", alpha=0.25)
            info_text.set_text("Selections done. Press Enter to confirm, r to reset.")

        update_text()
        print(f"  Buffer: {buffer_range}   Sample: {sample_range}")

    def on_key(event):
        if event.key == "r":
            buffer_range[0] = buffer_range[1] = None
            sample_range[0] = sample_range[1] = None
            selection_stage[0] = "buffer"
            for patch in [buffer_patch[0], sample_patch[0]]:
                if patch is not None:
                    patch.remove()
            buffer_patch[0] = sample_patch[0] = None
            info_text.set_text("Reset. Drag once for BUFFER, then for FULL SAMPLE REGION.")
            update_text()
            print("  Selections reset.")

        elif event.key == "enter":
            if buffer_range[0] is None or sample_range[0] is None:
                print("  Please select both regions first.")
                return
            plt.close(fig)

    SpanSelector(ax, onselect, "horizontal", useblit=True,
                 props=dict(alpha=0.2, facecolor="gray"),
                 interactive=True, drag_from_anywhere=True)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if buffer_range[0] is None or sample_range[0] is None:
        return None, None
    return tuple(buffer_range), tuple(sample_range)


def select_via_manual(frame_numbers):
    """
    Terminal input for manual range entry.
    Returns (buffer_range, sample_range).
    """
    n_min, n_max = int(frame_numbers[0]), int(frame_numbers[-1])
    print(f"\n--- Manual selection  (valid frames: {n_min}\u2013{n_max}) ---")

    while True:
        try:
            b1 = int(input(f"  Buffer start [{n_min}\u2013{n_max}]: "))
            b2 = int(input(f"  Buffer end   [{n_min}\u2013{n_max}]: "))
            s1 = int(input(f"  Sample start [{n_min}\u2013{n_max}]: "))
            s2 = int(input(f"  Sample end   [{n_min}\u2013{n_max}]: "))
        except ValueError:
            print("  Enter integer frame numbers. Try again.\n")
            continue

        if not (n_min <= b1 <= b2 <= n_max):
            print(f"  Buffer range must satisfy {n_min} \u2264 start \u2264 end \u2264 {n_max}. Try again.\n")
            continue
        if not (n_min <= s1 <= s2 <= n_max):
            print(f"  Sample range must satisfy {n_min} \u2264 start \u2264 end \u2264 {n_max}. Try again.\n")
            continue

        return (b1, b2), (s1, s2)


def select_via_auto(trace, frame_numbers,
                    baseline_fraction=0.15,
                    peak_fraction=0.02, smooth_window=30):
    """
    Automatically detect buffer and sample regions.

    Buffer  = flat baseline frames at the start of the run
              (frames before the signal rises above threshold).
    Sample  = frames inside the peak.

    The trace is smoothed before thresholding so that small noise spikes
    don't falsely trigger a very low peak_fraction threshold.

    The threshold is the *higher* of two estimates:
      - baseline_mean + peak_sigma * baseline_std   (noise-based, minimum guard)
      - baseline_mean + peak_fraction * (peak_max - baseline_mean)  (height-based)

    Returns (buffer_range, sample_range) or (None, None) if detection fails.
    """
    n = len(trace)

    # smooth the trace to suppress noise spikes before thresholding
    if smooth_window > 1:
        kernel   = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(trace, kernel, mode="same")
    else:
        smoothed = trace.copy()

    # --- estimate baseline from the first `baseline_fraction` of frames ---
    n_baseline    = max(5, int(n * baseline_fraction))
    baseline_med  = np.nanmedian(smoothed[:n_baseline])
    baseline_std  = np.nanstd(smoothed[:n_baseline])

    peak_max = np.nanmax(smoothed)

    # Threshold is purely fraction-based: the smoothing already suppresses noise
    # spikes, so the sigma guard is not needed and would override the user's choice.
    threshold = baseline_med + peak_fraction * (peak_max - baseline_med)

    above           = smoothed > threshold
    indices_above   = np.where(above)[0]

    if len(indices_above) == 0:
        print(f"  Auto-detection: no peak found above threshold "
              f"({threshold:.4g}).  Try lowering AUTO_PEAK_FRACTION.")
        return None, None

    # sample = span from first to last frame above threshold
    sample_start = int(frame_numbers[indices_above[0]])
    sample_end   = int(frame_numbers[indices_above[-1]])

    # buffer = flat frames before the peak starts
    pre_peak_idx    = indices_above[0]
    baseline_before = np.where(~above[:pre_peak_idx])[0]

    if len(baseline_before) < 3:
        print("  Auto-detection: fewer than 3 buffer frames before peak.  "
              "Try lowering AUTO_BASELINE_FRACTION.")
        return None, None

    buffer_start = int(frame_numbers[0])
    buffer_end   = int(frame_numbers[baseline_before[-1]])

    print(f"  Baseline estimate : {baseline_med:.4g}  \u00b1  {baseline_std:.4g}")
    print(f"  Peak maximum      : {peak_max:.4g}")
    print(f"  Threshold         : {threshold:.4g}  "
          f"(= baseline + {peak_fraction*100:.1f}% of peak height)")
    print(f"  Auto buffer       : frames {buffer_start}\u2013{buffer_end}")
    print(f"  Auto sample       : frames {sample_start}\u2013{sample_end}")

    return (buffer_start, buffer_end), (sample_start, sample_end)


def show_confirmation(frame_numbers, trace, buffer_range, sample_range,
                      spike_frames=None, spike_vals=None):
    """
    Read-only confirmation plot showing the selected regions.
    spike_frames / spike_vals: frame numbers and trace values of buffer frames
    that were excluded due to spike rejection (shown as orange crosses).
    Press Enter to accept, r to redo with the interactive slider.
    Returns True (accepted) or False (redo with slider).
    """
    result = [False]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(top=0.88)

    ax.plot(frame_numbers, trace, color="steelblue", lw=1.1)
    ax.set_xlabel("frame nr.")
    ax.set_ylabel("average intensity")
    ax.set_title("Confirm selection")

    ax.axvspan(buffer_range[0], buffer_range[1], color="red",   alpha=0.30,
               label=f"Buffer : {buffer_range[0]}\u2013{buffer_range[1]}")
    ax.axvspan(sample_range[0], sample_range[1], color="green", alpha=0.30,
               label=f"Sample : {sample_range[0]}\u2013{sample_range[1]}")

    if spike_frames is not None and len(spike_frames) > 0:
        ax.plot(spike_frames, spike_vals, "x", color="darkorange",
                markersize=8, markeredgewidth=2,
                label=f"Excluded buffer spikes ({len(spike_frames)} frames)")

    ax.legend(loc="upper right", fontsize=10)

    n_spikes = len(spike_frames) if spike_frames is not None else 0
    spike_note = f"    ({n_spikes} spike frames excluded from buffer)" if n_spikes else ""

    fig.text(0.02, 0.95,
             "Press  Enter  to accept   |   r  to redo with the interactive slider",
             fontsize=11)
    fig.text(0.02, 0.91,
             f"Buffer: {buffer_range[0]}\u2013{buffer_range[1]}    "
             f"Sample: {sample_range[0]}\u2013{sample_range[1]}{spike_note}",
             fontsize=11)

    def on_key(event):
        if event.key == "enter":
            result[0] = True
            plt.close(fig)
        elif event.key == "r":
            result[0] = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return result[0]


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
files = get_all_files(FOLDER, EXTENSIONS)

if len(files) == 0:
    raise ValueError("No files found in the folder.")

print(f"Found {len(files)} files.")
for i, f in enumerate(files, start=1):
    print(f"{i:4d}  {f.name}")

frame_numbers = np.arange(1, len(files) + 1)
_trace_data   = [average_intensity_for_trace(f, QMIN, QMAX) for f in files]
trace         = np.array([t[0] for t in _trace_data])
trace_err     = np.array([t[1] for t in _trace_data])

# ---------------------------------------------------------
# Select buffer and sample regions
# ---------------------------------------------------------
mode = SELECTION_MODE
buffer_range = sample_range = None

while True:
    if mode == "auto":
        print("\n--- Auto-detection ---")
        buffer_range, sample_range = select_via_auto(
            trace, frame_numbers,
            AUTO_BASELINE_FRACTION,
            AUTO_PEAK_FRACTION, AUTO_SMOOTH_WINDOW
        )
        if buffer_range is None:
            print("Falling back to interactive slider.")
            mode = "slider"
            continue

    elif mode == "manual":
        buffer_range, sample_range = select_via_manual(frame_numbers)

    elif mode == "slider":
        buffer_range, sample_range = select_via_slider(frame_numbers, trace)
        if buffer_range is None:
            raise SystemExit("No valid selection made. Nothing exported.")

    # --- spike rejection in buffer region ---
    buf_slice       = slice(buffer_range[0] - 1, buffer_range[1])
    buf_trace_vals  = trace[buf_slice]
    buf_frame_nums  = frame_numbers[buf_slice]

    if BUFFER_SPIKE_SIGMA is not None:
        buf_med       = np.nanmedian(buf_trace_vals)
        buf_std       = np.nanstd(buf_trace_vals)
        spike_mask    = buf_trace_vals > buf_med + BUFFER_SPIKE_SIGMA * buf_std
        spike_frames  = buf_frame_nums[spike_mask]
        spike_vals    = buf_trace_vals[spike_mask]
        if len(spike_frames):
            print(f"  Excluding {len(spike_frames)} spike frame(s) from buffer: "
                  f"{spike_frames.tolist()}")
    else:
        spike_mask   = np.zeros(len(buf_trace_vals), dtype=bool)
        spike_frames = np.array([])
        spike_vals   = np.array([])

    # Show confirmation for every mode.
    # Pressing r from here always sends the user to the slider.
    accepted = show_confirmation(frame_numbers, trace, buffer_range, sample_range,
                                 spike_frames, spike_vals)
    if accepted:
        break

    print("Redoing with interactive slider...")
    mode = "slider"

print(f"\nFinal selection:")
print(f"  BUFFER : {buffer_range[0]} \u2013 {buffer_range[1]}")
print(f"  SAMPLE : {sample_range[0]} \u2013 {sample_range[1]}")

# ---------------------------------------------------------
# Average buffer (spike frames excluded)
# ---------------------------------------------------------
all_buffer_files = files[buffer_range[0] - 1 : buffer_range[1]]
buffer_files     = [f for f, is_spike in zip(all_buffer_files, spike_mask) if not is_spike]
sample_files_all = files[sample_range[0] - 1 : sample_range[1]]

if len(buffer_files) == 0:
    raise ValueError("Buffer selection is empty after spike rejection.")
if len(sample_files_all) == 0:
    raise ValueError("Sample selection is empty.")

print("\nAveraging buffer...")
q_buf, I_buf, err_buf = average_curves(buffer_files) #buffer averaged once here, then subtracted from each sample block average in the loop below
                                                     # σ_buffer — fixed, reused every block

out_folder = FOLDER.parent / f"{FOLDER.name}_subtracted_50frame_blocks2"
out_folder.mkdir(exist_ok=True)
print(f"Saving output to: {out_folder}")

# ---------------------------------------------------------
# Export buffer-subtracted blocks
# ---------------------------------------------------------
curve_no    = 1
start_local = 0

while start_local < len(sample_files_all):
    end_local  = start_local + WINDOW_SIZE
    block_files = sample_files_all[start_local:end_local]

    if len(block_files) < WINDOW_SIZE and not EXPORT_REMAINDER:
        print(f"Skipping last incomplete block of {len(block_files)} frames.")
        break

    q_sam, I_sam, err_sam = average_curves(block_files) # σ_sample for this block

    if not np.allclose(q_sam, q_buf):
        raise ValueError("Sample and buffer q grids do not match.")

    I_sub   = I_sam - I_buf  # I_sub = I_sample - I_buffer
    err_sub = np.sqrt(err_sam**2 + err_buf**2) # σ_sub = √(σ_sample² + σ_buffer²)

    global_start = sample_range[0] + start_local
    global_end   = sample_range[0] + start_local + len(block_files) - 1

    out_name = (
        f"subtracted_{curve_no:03d}"
        f"_sample_{global_start}_{global_end}"
        f"_buffer_{buffer_range[0]}_{buffer_range[1]}.dat"
    )
    out_path = out_folder / out_name

    header_lines = [
        f"Buffer frames: {buffer_range[0]}-{buffer_range[1]}",
        f"Sample frames: {global_start}-{global_end}",
        f"Window size used: {len(block_files)}"
    ]

    save_curve(out_path, q_sam, I_sub, err_sub, header_lines)
    print(f"Saved: {out_name}")

    curve_no    += 1
    start_local += STEP

print("\nDone.")

# ---------------------------------------------------------
# Summary plot with error bars
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(top=0.88)

ax.fill_between(frame_numbers, trace - trace_err, trace + trace_err,
                color="steelblue", alpha=0.35, zorder=1, label="\u00b11\u03c3 uncertainty")
ax.plot(frame_numbers, trace, color="steelblue", lw=1.1, zorder=2)

ax.axvspan(buffer_range[0], buffer_range[1], color="red",   alpha=0.25,
           label=f"Buffer : {buffer_range[0]}\u2013{buffer_range[1]}")
ax.axvspan(sample_range[0], sample_range[1], color="green", alpha=0.25,
           label=f"Sample : {sample_range[0]}\u2013{sample_range[1]}")

if len(spike_frames) > 0:
    ax.plot(spike_frames, spike_vals, "x", color="darkorange",
            markersize=8, markeredgewidth=2,
            label=f"Excluded buffer spikes ({len(spike_frames)} frames)")

ax.set_xlabel("frame nr.")
ax.set_ylabel("average intensity")
ax.set_title(f"{FOLDER.name}  —  chromatogram with error bars")
ax.legend(loc="upper right", fontsize=10)

fig.text(0.02, 0.95,
         f"Buffer: {buffer_range[0]}\u2013{buffer_range[1]}    "
         f"Sample: {sample_range[0]}\u2013{sample_range[1]}    "
         f"Window: {WINDOW_SIZE} frames    Step: {STEP}",
         fontsize=10)

plot_path = out_folder / "chromatogram_summary.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Summary plot saved: {plot_path}")
plt.show()
