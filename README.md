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

   **UV:**

   ```bash
   uv venv .venv
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
