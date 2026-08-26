from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_NUM_FOCI_KEY = "CELL_LAYER::PROPS::NUM_FOCI"


def plot_foci_counts(cell_layers_df: pd.DataFrame, num_foci_key=DEFAULT_NUM_FOCI_KEY):
    fig_foci_count, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)

    # Proportion of cells for each number of foci
    foci_counts: Any = (
        cell_layers_df[num_foci_key].value_counts(normalize=True).sort_index()
    )

    ax.bar(
        foci_counts.index,
        foci_counts.values,
        width=0.8,
    )

    ax.set_ylabel("Proportion of cells")
    ax.set_xlabel("Number of foci")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))

    return fig_foci_count
