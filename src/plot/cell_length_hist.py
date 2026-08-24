import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CELL_LENGTH_KEY = "CELL::PROPS::AXIS_MAJOR_LENGTH"

def plot_cell_length(cells_df, cell_length_key=DEFAULT_CELL_LENGTH_KEY):
    min_len_um = 0
    max_len_um = 3
    num_bins = 50

    h_cell_lengths, bin_edges = np.histogram(
        cells_df[cell_length_key], bins=np.linspace(min_len_um, max_len_um, num_bins)
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3))

    ax.bar(
        bin_edges[:-1],
        h_cell_lengths,
        width=(bin_edges[1] - bin_edges[0]) / 2,
        align="edge",
        facecolor="xkcd:orange",
        edgecolor="none",
    )

    ax.set_xlabel("Cell length (μm)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.set_xlim(0, 5)

    # PERCENTILE_STEP = 25

    # percentiles = np.linspace(0, 100, int(100 / PERCENTILE_STEP) + 1)
    # cell_length_percentiles = np.percentile(cells_df[cell_length_key], percentiles)

    return fig
