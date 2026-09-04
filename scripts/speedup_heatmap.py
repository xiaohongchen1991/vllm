# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

batch_sizes = [1, 2, 4, 8, 16, 24, 32]
models = ["1.7B", "4B", "8B", "14B", "27B", "32B"]

fp8_dynamic = np.array(
    [
        [1.074, 1.060, 1.044, 1.051, 1.024, 1.037, 1.024],
        [1.062, 1.036, 1.025, 1.016, 1.016, 1.066, 1.045],
        [1.037, 1.030, 1.014, 0.992, 1.019, 1.031, 1.024],
        [1.025, 1.014, 1.009, 1.006, 0.995, 1.042, 1.032],
        [1.004, 1.007, 1.008, 1.002, 0.999, 1.023, 1.018],
        [1.036, 1.009, 1.010, 1.025, 1.019, 1.052, 1.030],
    ]
)

w8a8 = np.array(
    [
        [1.108, 1.095, 1.087, 1.080, 1.061, 1.051, 1.023],
        [1.145, 1.119, 1.109, 1.088, 1.046, 1.047, 1.036],
        [1.084, 1.050, 1.050, 1.040, 1.035, 1.005, 1.004],
        [1.074, 1.061, 1.056, 1.050, 1.034, 1.004, 1.015],
        [1.070, 1.054, 1.047, 1.046, 1.032, 1.017, 1.022],
        [1.071, 1.059, 1.050, 1.039, 1.025, 0.999, 1.004],
    ]
)

block_fp8 = np.array(
    [
        [1.107, 1.113, 1.103, 1.091, 1.042, 1.153, 1.023],
        [1.108, 1.134, 1.128, 1.099, 1.068, 1.142, 1.146],
        [1.023, 1.008, 1.013, 1.072, 1.048, 1.070, 1.051],
        [1.015, 1.023, 0.939, 1.052, 1.032, 1.067, 1.034],
        [0.980, 1.000, 0.948, 1.024, 1.022, 1.042, 1.040],
        [0.997, 1.016, 0.997, 1.048, 1.017, 1.039, 1.012],
    ]
)

datasets = [
    ("FP8 Dynamic", fp8_dynamic),
    ("W8A8 INT8", w8a8),
    ("Block FP8", block_fp8),
]

# Use fixed limits across all plots so colors have the same meaning.
#
# I recommend clipping the visualization range rather than letting the
# 1.364x outlier dominate the entire color scale.
vmin = 0.90
vcenter = 1.00
vmax = 1.15

norm = TwoSlopeNorm(
    vmin=vmin,
    vcenter=vcenter,
    vmax=vmax,
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.8),
    sharey=True,
    constrained_layout=True,
)

for ax, (title, data) in zip(axes, datasets):
    im = ax.imshow(
        data,
        cmap="RdYlGn",
        norm=norm,
        aspect="auto",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=11)

    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    # Draw subtle cell boundaries.
    ax.set_xticks(np.arange(-0.5, len(batch_sizes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate every cell with its actual value.
    for i in range(len(models)):
        for j in range(len(batch_sizes)):
            value = data[i, j]

            # Choose text color based roughly on distance from center
            # for readability on darker cells.
            if value <= 0.94 or value >= 1.11:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                f"{value:.3f}×",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

axes[0].set_ylabel("Model", fontsize=11)

cbar = fig.colorbar(
    im,
    ax=axes,
    shrink=0.9,
    pad=0.02,
)

cbar.set_label("Throughput Speedup", fontsize=11)
cbar.set_ticks([0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
cbar.set_ticklabels(
    [
        "0.90×",
        "0.95×",
        "1.00×",
        "1.05×",
        "1.10×",
        "1.15×",
    ]
)

fig.suptitle(
    "Helion Linear Backend End-to-End Throughput Speedup",
    fontsize=16,
    fontweight="bold",
)

plt.savefig(
    "helion_throughput_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "helion_throughput_heatmap.pdf",
    bbox_inches="tight",
)

plt.show()
