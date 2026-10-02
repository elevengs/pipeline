# Quantification Pipeline

This repository contains Python code to process and quantify characteristics
of bacterial cells from phase contrast and fluorescence microscopy images.

### How to use

1. `git clone` the repository or download the source code.

2. Set up a data folder containing your images and labels.

   Place all of your ND2 or TIFF files, with whatever organization you want, in some folder.

   Place labels next to the source files with the following pattern: for an ND2 file named `data/A/B/Example.nd2`, there should be a file called `data/A/B/Example_labels.png` (or `.tif`/`.tiff`).

   Make sure that for each source image, there is a `meta.json` in one of its parent folders. For example, you might place `meta.json` at `data/A/B/meta.json` or `data/A/meta.json`; the former would be used for all source images in `data/A/B`, and the latter for all source images in `data/A`.

3. Install dependencies however you want.

   **Nix:**

   ```bash
   nix develop
   ```

   **UV (RECOMMENDED):**

   ```bash
   uv venv .venv
   uv sync
   source .venv/bin/activate
   ```

   **Anaconda/miniconda/conda:**

   ```bash
   conda create -n pipeline python=3.12
   conda activate pipeline
   conda install -c conda-forge numpy scipy pandas matplotlib scikit-image tifffile imageio shapely seaborn tqdm pathspec h5py dask
   python -m pip install "nd2[legacy]" mpl-tools matplotlib-scalebar
   python -m pip install ruff ty
   ```

4. Open `pipeline/pipeline.ipynb` in your favorite Jupyter notebook editor.

5. In the first cell:

   - In the line that starts with `DATA = …`, replace `"../../data"` with the relative path to the data folder you created.
   - Adjust the `INCLUDE` pattern to pull in whatever source files you want.
   - Change the output directory to wherever you want all of the output to go.
   - I recommend leaving the rest of the settings alone, but you can mess with it if you understand what it will do.

6. Run the first cell to configure the notebook.

7. In the second cell:

   - Adjust the focus detection parameters however you would like.
   - These are well-documented and you can view the details with:

     ```bash
     python -m src.quant.main --help
     ```

8. Run the third cell to produce the final output.

### What is created?

#### `batch_inventory`

For each image, say for example `data/A/B/Example.nd2`:

1. `out/cells/A/B/Example.csv`
   1. Data about each cell in the image, including but not limited to
      1. Dimensions
      2. ID
      3. FOV path
      4. Group
2. `out/cell_layers/A/B/Example.csv`
   1. Data about each layer of each cell in the image, including but not limited to
      1. How many foci were detected on that layer
      2. What type of layer it is
      3. Group
      4. ID
      5. FOV path
3. `out/foci/A/B/Example.csv`
   1. Data about each focus in the image, including but not limited to
      1. What cell it belongs to (by ID)
      2. What layer it belongs to (by ID)
      3. Dimensions
      4. Group
      5. ID
      6. FOV path
      7. Coordinates

#### `finalize`

For each group, e.g. `group_a`:

1. Aggregated CSVs that combine all of the CSVs for the images in the group
   1. `out/final/group_a/cells.csv`
   2. `out/final/group_a/cell_layers.csv`
   3. `out/final/group_a/foci.csv`
2. `out/final/group_a/cell_length.{png,svg}`
   1. Histogram of cell lengths in the group
3. `out/final/group_a/cell_length.{png,svg}`
4. For each fluorescence channel, e.g. `channel_a`, a folder with the following:
   1. `long_axis_position_vs_cell_length.{png,svg}`
      1. A plot of foci by their position on the long axis of the cell they were detected in and the length of that cell. Colors are kernel density estimations of probability density.
   2. `long_axis_position_vs_foci_intensity.{png,svg}`
      1. Same thing as above, except colors are focus intensity in raw arbitrary units.
   3. `long_axis_position_vs_foci_number.{png,svg}`
      1. Same thing as above, except colors are number of foci in the same cell.
   4. `num_foci.{png,svg}`
      1. Bar chart of the number of foci in the cells in the group.
   5. `rep.{png,svg}`
      1. Representative cell shapes and foci locations for four different cell length bins.
      2. The outline is created as an average of all of the cells in the length bin.
      3. Foci are placed by scaling and rotating the cell they were detected in so that the poles match those of the representative cell, and then plotting those coordinates, as well as the three mirror points across the x and y axes.