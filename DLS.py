import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

def plot_dls_binned(data_str, title="DLS intensity distribution"):
    arr = np.loadtxt(StringIO(data_str), skiprows=1)
    d = arr[:, 0]   # bin centers (nm)
    y = arr[:, 1]   # intensity (%)

    # --- bin edges from centers (geometric midpoints; good for log-spaced DLS grids) ---
    edges = np.empty(len(d) + 1)
    edges[1:-1] = np.sqrt(d[:-1] * d[1:])
    edges[0]    = d[0] / np.sqrt(d[1] / d[0])
    edges[-1]   = d[-1] * np.sqrt(d[-1] / d[-2])

    left  = edges[:-1]
    width = edges[1:] - edges[:-1]

    plt.figure(figsize=(8.0, 4.6))

    # 1) full grid faintly (keeps full range, no zoom)
    plt.bar(left, y, width=width, align="edge",
            color="lightgray", edgecolor="none", alpha=0.35)

    # 2) overlay non-zero bins
    mask = y > 0
    plt.bar(left[mask], y[mask], width=width[mask], align="edge",
            color="#4C78A8", edgecolor="#1F2D3A", linewidth=1.1, alpha=0.95)

    # optional: annotate main peak (largest bin)
    if np.any(mask):
        i = np.argmax(y)
        plt.annotate(f"Peak ~ {d[i]:.1f} nm",
                     xy=(d[i], y[i]),
                     xytext=(d[i]*1.3, y[i]*0.85),
                     arrowprops=dict(arrowstyle="->", lw=1.0),
                     fontsize=10)

    plt.xscale("log")
    plt.xlim(edges[0], edges[-1])              # full diameter range
    plt.ylim(0, max(1.0, y.max() * 1.12))      # headroom
    plt.xlabel("Diameter (nm)")
    plt.ylabel("Intensity (%)")
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", alpha=0.35)
    plt.grid(True, which="minor", linestyle=":",  alpha=0.20)
    plt.tight_layout()
    plt.show()

    print(f"Sum of intensity = {y.sum():.3f} %")

# ---- Formulation 0 (your data) ----
data_form0 = """Diameter(nm)\t% Intensity
0.0380193\t0
0.0474358\t0
0.0591846\t0
0.0738432\t0
0.0921325\t0
0.114952\t0
0.143423\t0
0.178945\t0
0.223266\t0
0.278564\t0
0.347557\t0
0.433639\t0
0.541042\t0
0.675046\t0
0.842239\t0
1.05084\t0
1.31111\t0
1.63585\t0
2.04101\t0
2.54652\t0
3.17723\t0
3.96416\t0
4.94599\t0
6.171\t0
7.69942\t0
9.60639\t0
11.9857\t0
14.9543\t0
18.6581\t0
23.2793\t0
29.045\t0
36.2388\t0
45.2143\t0
56.4129\t0
70.3851\t0
87.8179\t1.10632
109.568\t79.6935
136.706\t19.2001
170.565\t0
"""

plot_dls_binned(data_form0, title="DLS – Formulation 0")
##########################################################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from io import StringIO

# def plot_dls_binned(data_str, title="DLS intensity distribution"):
#     arr = np.loadtxt(StringIO(data_str), skiprows=1)
#     d = arr[:, 0]   # bin centers (nm)
#     y = arr[:, 1]   # intensity (%)

#     # --- bin edges from centers (geometric midpoints; good for log-spaced DLS grids) ---
#     edges = np.empty(len(d) + 1)
#     edges[1:-1] = np.sqrt(d[:-1] * d[1:])
#     edges[0]    = d[0] / np.sqrt(d[1] / d[0])
#     edges[-1]   = d[-1] * np.sqrt(d[-1] / d[-2])

#     left  = edges[:-1]
#     width = edges[1:] - edges[:-1]

#     plt.figure(figsize=(8.0, 4.6))

#     # 1) full grid faintly (keeps full range, no zoom)
#     plt.bar(left, y, width=width, align="edge",
#             color="lightgray", edgecolor="none", alpha=0.35)

#     # 2) overlay non-zero bins
#     mask = y > 0
#     plt.bar(left[mask], y[mask], width=width[mask], align="edge",
#             color="#4C78A8", edgecolor="#1F2D3A", linewidth=1.1, alpha=0.95)

#     # optional: annotate main peak (largest bin)
#     if np.any(mask):
#         i = np.argmax(y)
#         plt.annotate(f"Peak ~ {d[i]:.1f} nm",
#                      xy=(d[i], y[i]),
#                      xytext=(d[i]*1.3, y[i]*0.85),
#                      arrowprops=dict(arrowstyle="->", lw=1.0),
#                      fontsize=10)

#     plt.xscale("log")
#     plt.xlim(edges[0], edges[-1])              # full diameter range
#     plt.ylim(0, max(1.0, y.max() * 1.12))      # headroom
#     plt.xlabel("Diameter (nm)")
#     plt.ylabel("Intensity (%)")
#     plt.title(title)
#     plt.grid(True, which="major", linestyle="--", alpha=0.35)
#     plt.grid(True, which="minor", linestyle=":",  alpha=0.20)
#     plt.tight_layout()
#     plt.show()

#     print(f"Sum of intensity = {y.sum():.3f} %")

# # ---- Formulation  (your data) ----
# data_form3_loaded = """Diameter(nm)\t% Intensity
# data_str = """Diameter(nm)\t% Intensity
# 0.0380193\t0
# 0.0474358\t0
# 0.0591846\t0
# 0.0738432\t0
# 0.0921325\t0
# 0.114952\t0
# 0.143423\t0
# 0.178945\t0
# 0.223266\t0
# 0.278564\t0
# 0.347557\t0
# 0.433639\t0
# 0.541042\t0
# 0.675046\t0
# 0.842239\t0
# 1.05084\t0
# 1.31111\t0
# 1.63585\t0
# 2.04101\t0
# 2.54652\t0
# 3.17723\t0
# 3.96416\t0
# 4.94599\t0
# 6.171\t0
# 7.69942\t0
# 9.60639\t0
# 11.9857\t0
# 14.9543\t0
# 18.6581\t0
# 23.2793\t0
# 29.045\t0
# 36.2388\t0
# 45.2143\t0
# 56.4129\t0
# 70.3851\t0
# 87.8179\t1.10632
# 109.568\t79.6935
# 136.706\t19.2001
# 170.565\t0
# """

# # ---- Formulation 3 loaded (your data) ----
# data_form3_loaded = """Diameter(nm)\t% Intensity
# 0.0380193\t0
# 0.0474358\t0
# 0.0591846\t0
# 0.0738432\t0
# 0.0921325\t0
# 0.114952\t0
# 0.143423\t0
# 0.178945\t0
# 0.223266\t0
# 0.278564\t0
# 0.347557\t0
# 0.433639\t0
# 0.541042\t0
# 0.675046\t0
# 0.842239\t0
# 1.05084\t0
# 1.31111\t0
# 1.63585\t0
# 2.04101\t0
# 2.54652\t0
# 3.17723\t0
# 3.96416\t0
# 4.94599\t0
# 6.171\t0
# 7.69942\t0
# 9.60639\t0
# 11.9857\t0
# 14.9543\t0
# 18.6581\t0
# 23.2793\t0
# 29.045\t0
# 36.2388\t0
# 45.2143\t0
# 56.4129\t0
# 70.3851\t23.3118
# 87.8179\t37.0041
# 109.568\t28.9438
# 136.706\t10.5271
# 170.565\t0
# 212.81\t0
# 265.518\t0
# 331.281\t0
# 413.332\t0
# 515.704\t0
# 643.433\t0
# 802.796\t0
# 1001.63\t0
# 1249.71\t0
# 1559.24\t0
# 1945.42\t0
# 2427.26\t0
# 3028.44\t0
# 3778.52\t0
# 4714.37\t0
# 5882.01\t0
# 7338.85\t0
# 9156.51\t0
# 11424.4\t0
# 14253.9\t0
# 17784.3\t0.213191
# 22189.1\t0
# """

# plot_dls_binned(data_form3_loaded, title="DLS – Formulation 3 loaded")