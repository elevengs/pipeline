import numpy as np
import pandas as pd
from pathlib import Path
import shapely as sl
from skimage.measure import find_contours
from skimage.measure import regionprops_table
from src.spideymaps.spideymaps_v2 import Spideymap, smooth_skin

from src.spideymaps.vis import create_rep_grid_colicoords
from src.util.misc import replace_root_with
import pickle
import concurrent.futures
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.spideymaps.spideymaps_v2 import SpideyAtlas
from src.spideymaps.plot_spideymaps import plot_spideymaps
from src.util.save import save_fig
from src.spideymaps.stats import Stats
from src.defs import (
    Channel,
    CellLayer,
    Focus,
    load,
    cell_layers_from_csv,
    cell_layers_to_dataframe,
    cells_from_csv,
    cells_to_dataframe,
    foci_from_csv,
    foci_to_dataframe,
)

grid_params = dict(
    radius=6,
    n_shells=5,
    n_cols=5,
    n_phi=(1, 3, 5, 5, 5),
    level=0,
)

def cells_path(source, args) -> Path:
    return replace_root_with(source, args.root, args.cells_dir).with_suffix(".csv")

def cell_layers_path(source, args) -> Path:
    return replace_root_with(source, args.root, args.cell_layers_dir).with_suffix(".csv")

def foci_path(source, args) -> Path:
    return replace_root_with(source, args.root, args.foci_dir).with_suffix(".csv")

def channel_output_dir(out: Path, channel: Channel) -> Path:
    return channel_output_dir_for_name(out, channel.name)

def channel_output_dir_for_name(out: Path, channel_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", channel_name).strip("._")
    if not safe_name:
        safe_name = "unnamed"
    return out.joinpath(f"channel_{safe_name}")

def microns_per_pixel(fov) -> float:
    x_scale = fov.image_dimensions.x_microns_per_pixel
    y_scale = fov.image_dimensions.y_microns_per_pixel
    if not np.isclose(x_scale, y_scale, rtol=1e-3):
        raise ValueError(
            "Spideymap requires approximately square pixels, but "
            f"{fov.tif_path} has scales {x_scale} and {y_scale} microns per pixel"
        )
    return float(np.sqrt(x_scale * y_scale))

def handle_cell_layer(cell_layer: CellLayer, foci: list[Focus], bboxes: pd.DataFrame):
    cell = cell_layer.cell
    foci = list(foci)
    stats = Stats()

    num_foci = cell_layer.props.num_foci
    stats.totals_foci[num_foci] = 1
    stats.total_cells = 1

    if num_foci < 1 or num_foci > 3:
        return None, stats
    if cell.label not in bboxes.index:
        return None, stats
    if len(foci) != num_foci:
        raise ValueError(
            f'cell-layer "{cell_layer.id}" reports '
            f"{num_foci} foci but {len(foci)} were loaded"
        )

    # Now that we've passed the two main conditions,
    # 1) at least one but less than four foci and 
    # 2) availability of a bounding box
    # try to create a map
    stats.total_tried_map_creation = 1

    foci_df = foci_to_dataframe({focus.id: focus for focus in foci})
    cell_mask = cell.map.mask

    try:
        contour = find_contours(cell_mask, level=0.5)
        out = contour[0][:, ::-1]
    except Exception:
        stats.total_failed_contour_creation = 1
        return None, stats

    out = smooth_skin(out, sigma=0.5)
    out = sl.LinearRing(out)

    spidey_map = Spideymap(
        bimage=cell_mask, coords=foci_df,
        xcol="FOCUS::PROPS::X", ycol="FOCUS::PROPS::Y",
    )

    try:
        spidey_map.make_grid(out=out, **grid_params)
        spidey_map.count()
        spidey_map.coords["cell_length"] = spidey_map.mid.length
        spidey_map.microns_per_pixel = microns_per_pixel(cell.fov)
        stats.total_passed_map_creation = 1
    except Exception as exc:
        stats.total_failed_grid_creation = 1
        return None, stats

    return spidey_map, stats
        

def handle_source(source, args):
    fov = load(source, args.root)

    cells = cells_from_csv(fov, cells_path(source, args))
    cell_layers = cell_layers_from_csv(cells, cell_layers_path(source, args))
    foci = foci_from_csv(cell_layers, foci_path(source, args))

    bboxes = regionprops_table(label_image=fov.labels, properties=("label", "bbox"))
    bboxes = pd.DataFrame(bboxes).set_index("label", drop=False)

    maps_by_channel = {}
    stats_by_channel = {}

    for cell_layer in cell_layers.values():
        layer = cell_layer.layer
        if not layer.is_fluor:
            continue
        channel = layer.channel

        cell_foci = [
            focus for focus in foci.values()
            if focus.cell_layer.id == cell_layer.id
        ]
        map, new_stats = handle_cell_layer(
            cell_layer,
            cell_foci,
            bboxes
        )

        if map is not None:
            maps_by_channel.setdefault(channel, []).append(map)

        stats_by_channel[channel] = (
            stats_by_channel.get(channel, Stats()) + new_stats
        )

    return maps_by_channel, stats_by_channel, cells, cell_layers, foci

def handle_source_foci(source, args):
    fov = load(source, args.root, load_tif = False)

    cells = cells_from_csv(fov, cells_path(source, args))
    cell_layers = cell_layers_from_csv(cells, cell_layers_path(source, args))
    foci = foci_from_csv(cell_layers, foci_path(source, args))

    return cells, cell_layers, foci


def plot_foci_counts(totals_foci, grand_total_foci):
    fig_foci_count, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)

    num_foci, ct = list(zip(*(totals_foci.items())))

    num_foci = list(num_foci)
    ct = list(map(lambda x: float(x) / float(grand_total_foci), list(ct)))

    ax.bar(num_foci, ct)
    ax.set_ylabel("Proportion of cells")
    ax.set_xlabel("Number of foci")
    ax.set_yticks(np.arange(0, 1, 0.1))

    return fig_foci_count, ax


def cell_length_histogram(atlas):
    min_len_um = 0
    max_len_um = 3
    num_bins = 50

    cell_lengths_um = np.array([
        map.mid.length * map.microns_per_pixel for map in atlas.maps.values()
    ])

    h_cell_lengths, bin_edges = np.histogram(
        cell_lengths_um, bins=np.linspace(min_len_um, max_len_um, num_bins)
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3))

    ax.bar(
        bin_edges[:-1],
        h_cell_lengths,
        width=(bin_edges[1] - bin_edges[0] / 2),
        align="edge",
        facecolor="xkcd:orange",
        edgecolor="none",
    )

    ax.set_xlabel("Cell length (μm)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.set_xlim(0, 11)

    PERCENTILE_STEP = 5

    percentiles = np.linspace(0, 100, int(100 / PERCENTILE_STEP) + 1)
    cell_length_percentiles = np.percentile(cell_lengths_um, percentiles)

    for percentile, cell_length in zip(percentiles, cell_length_percentiles):
        print(f"{percentile:.0f}th percentile: {cell_length:.2f} μm")

    return fig, ax, cell_lengths_um


def get_rep(length_ranges, cell_lengths_um, pixel_size_um):
    rep_cell_lengths = np.zeros(len(length_ranges))
    rep_cells = []

    for i, (min_length, max_length) in enumerate(length_ranges):
        rep_cell_lengths[i] = cell_lengths_um[
            (cell_lengths_um > min_length) & (cell_lengths_um <= max_length)
        ].mean()

        cc_params = dict(
            xl=-rep_cell_lengths[i] / 2 / pixel_size_um + grid_params["radius"],
            xr=rep_cell_lengths[i] / 2 / pixel_size_um - grid_params["radius"],
            a0=0,
            a1=0,
            a2=0,
            r=grid_params["radius"],
        )
        try:
            rep_cell = create_rep_grid_colicoords(cc_params, grid_params)
            rep_cells.append(rep_cell)
        except Exception as e:
            print("Failed to create representative grid for this cell length")
            print(e)

    return rep_cell_lengths, rep_cells

def make_plot_labels(atlas, length_ranges, pixel_size_um):

    plot_labels = []

    for i, (min_length, max_length) in enumerate(length_ranges):
        rng_tag = f"{min_length:.1f}-{max_length:.1f}um"

        filt_col_label = f"length_range:{rng_tag}"
        count_col_label = f"counts:{rng_tag}"
        area_col_label = f"areas:{rng_tag}"
        density_col_label = f"density:{rng_tag}"  # only symmetric density is calculated
        density_norm_col_label = f"density_norm:{rng_tag}"

        symcount_col_label = f"{count_col_label}_symsum"
        symarea_col_label = f"{area_col_label}_symsum"

        # isolate data for cells within length range
        atlas.coords[filt_col_label] = (atlas.coords["cell_length_um"] > min_length) & (
            atlas.coords["cell_length_um"] <= max_length
        )

        atlas.sum_coords(
            sumcol_name=count_col_label,
            filt_col=filt_col_label,
        )

        atlas.sum_maps(
            map_data_col="area",
            atlas_data_col=area_col_label,
            min_length=min_length / pixel_size_um,
            max_length=max_length / pixel_size_um,
        )

        atlas.add_symmetric_elements(col_name=count_col_label)
        atlas.add_symmetric_elements(col_name=area_col_label)

        atlas.data[density_col_label] = (
            atlas.data[symcount_col_label] / atlas.data[symarea_col_label]
        )

        mean_density = (
            atlas.data[symcount_col_label].sum() / atlas.data[symarea_col_label].sum()
        )

        atlas.data[density_norm_col_label] = (
            atlas.data[density_col_label] / mean_density
        )

        plot_labels.append(density_norm_col_label)

    return plot_labels

def add_polarity(atlas_coords):
    atlas_coords["l_abs_centered"] = (
        atlas_coords["l_abs"] - atlas_coords["cell_length"] / 2
    )

    # Set the cell's pole to align with the mean location of its foci
    atlas_coords["polarity"] = atlas_coords.groupby("CELL::ID")[
        "l_abs_centered"
    ].transform(lambda x: 1 if x.mean() > 0 else -1)

    atlas_coords["l_abs_centered_pol"] = (
        atlas_coords["l_abs_centered"] * atlas_coords["polarity"]
    )

    atlas_coords["l_abs_centered_pol_um"] = (
        atlas_coords["l_abs_centered_pol"] * atlas_coords["MICRONS_PER_PIXEL"]
    )

def make_channel_atlas(
    channel: Channel,
    maps,
    stats: Stats,
    out: Path,
) -> pd.DataFrame | None:
    out.mkdir(parents=True, exist_ok=True)

    fig_foci_counts, _ = plot_foci_counts(stats.totals_foci, stats.total_cells)
    fig_foci_counts.suptitle(channel.name)
    save_fig(fig_foci_counts, out.joinpath("fig_foci_counts"))
    plt.close(fig_foci_counts)

    if not maps:
        return None

    atlas = SpideyAtlas(maps)

    pixel_sizes_um = np.array([map.microns_per_pixel for map in maps])
    if not np.allclose(pixel_sizes_um, pixel_sizes_um[0], rtol=1e-3):
        raise ValueError(
            f'all FOVs for channel "{channel.name}" must have the same pixel size'
        )
    pixel_size_um = float(pixel_sizes_um.mean())

    fig_hist, _, cell_lengths_um = cell_length_histogram(atlas)
    fig_hist.suptitle(channel.name)
    save_fig(fig_hist, out.joinpath("fig_hist"))
    plt.close(fig_hist)

    # Put cells into roughly four equally populated length ranges.
    quartiles = np.percentile(cell_lengths_um, [0.0, 25.0, 50.0, 75.0, 100.0])
    length_ranges = tuple(zip(quartiles[:-1], quartiles[1:]))

    rep_cell_lengths, rep_cells = get_rep(
        length_ranges, cell_lengths_um, pixel_size_um
    )

    atlas.get_colicoords()

    pixel_size_by_map = {
        map_name: map.microns_per_pixel for map_name, map in atlas.maps.items()
    }
    atlas.coords["MICRONS_PER_PIXEL"] = atlas.coords["map_name"].map(
        pixel_size_by_map
    )
    atlas.coords["cell_length_um"] = (
        atlas.coords["cell_length"] * atlas.coords["MICRONS_PER_PIXEL"]
    )

    plot_labels = make_plot_labels(atlas, length_ranges, pixel_size_um)

    fig_maps, _ = plot_spideymaps(
        atlas, rep_cells, rep_cell_lengths, plot_labels, pixel_size_um
    )
    fig_maps.suptitle(channel.name)
    save_fig(fig_maps, out.joinpath("fig_maps"))
    plt.close(fig_maps)

    add_polarity(atlas.coords)

    atlas.coords.to_csv(out.joinpath("atlas_coords.csv"), index=False)
    atlas.data.to_csv(
        out.joinpath("atlas_data.csv"),
        index=True,
        index_label=("i_r", "i_l", "i_phi"),
    )

    rep_cells_dict = {i: rep_cell for i, rep_cell in enumerate(rep_cells)}
    with open(out.joinpath("rep_cells_dict.pkl"), "wb") as file:
        pickle.dump(rep_cells_dict, file)

    return atlas.coords

def make_atlas_coords(sources, args):
    maps_by_channel = {}
    stats_by_channel = {}
    cells = {}
    cell_layers = {}
    foci = {}
    
    use_main_thread = args.max_workers == 0

    def update_data(source, handle_source_results):
        try:
            (
                source_maps_by_channel,
                source_stats_by_channel,
                source_cells,
                source_cell_layers,
                source_foci,
            ) = handle_source_results

            for channel, source_maps in source_maps_by_channel.items():
                maps_by_channel.setdefault(channel, []).extend(source_maps)

            for channel, source_stats in source_stats_by_channel.items():
                stats_by_channel[channel] = (
                    stats_by_channel.get(channel, Stats()) + source_stats
                )

            cells.update(source_cells)
            cell_layers.update(source_cell_layers)
            foci.update(source_foci)
                
            print({
                channel: source_stats.to_dict()
                for channel, source_stats in source_stats_by_channel.items()
            })
        except Exception as exc:
            print('%r generated an exception: %s' % (source, exc))
        else:
            pass
        
    if use_main_thread:
        for source in sources:
            update_data(source, handle_source(source, args))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_source = {executor.submit(handle_source, source, args): source for source in sources}
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                update_data(source, future.result())
                

    args.out.mkdir(parents=True, exist_ok=True)
    cells_to_dataframe(cells).to_csv(args.out.joinpath("cells.csv"), index=False)
    cell_layers_to_dataframe(cell_layers).to_csv(
        args.out.joinpath("cell_layers.csv"), index=False
    )
    foci_to_dataframe(foci).to_csv(args.out.joinpath("foci.csv"), index=False)

    coords_by_channel = {}
    channels = sorted(
        stats_by_channel,
        key=lambda channel: channel.name,
    )
    for channel in channels:
        coords = make_channel_atlas(
            channel,
            maps_by_channel.get(channel, []),
            stats_by_channel[channel],
            channel_output_dir(args.out, channel),
        )
        if coords is not None:
            coords_by_channel[channel] = coords

    combined_coords = (
        pd.concat(coords_by_channel.values(), ignore_index=True)
        if coords_by_channel
        else pd.DataFrame()
    )
    combined_coords.to_csv(args.out.joinpath("atlas_coords.csv"), index=False)

    return coords_by_channel

def make_foci_data(sources, args):
    cells = {}
    cell_layers = {}
    foci = {}

    use_main_thread = args.max_workers == 0

    def update_data(source, handle_source_results):
        try:
            source_cells, source_cell_layers, source_foci = handle_source_results

            cells.update(source_cells)
            cell_layers.update(source_cell_layers)
            foci.update(source_foci)
        except Exception as exc:
            print('%r generated an exception: %s' % (source, exc))
        else:
            pass

    if use_main_thread:
        for source in tqdm(
            sources,
            desc = "Reading cells and foci from CSVs",
            unit = "FOV"
        ):
            update_data(source, handle_source_foci(source, args))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_source = {executor.submit(handle_source_foci, source, args): source for source in sources}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_source),
                desc = "Reading cells and foci from CSVs",
                unit = "FOV"
            ):
                source = future_to_source[future]
                update_data(source, future.result())

    args.out.mkdir(parents=True, exist_ok=True)

    cells_df = cells_to_dataframe(cells)
    cell_layers_df = cell_layers_to_dataframe(cell_layers)
    
    cells_df.to_csv(args.out.joinpath("cells.csv"), index=False)
    cell_layers_df.to_csv(
        args.out.joinpath("cell_layers.csv"), index=False
    )
    foci_df = foci_to_dataframe(foci)
    foci_df.to_csv(args.out.joinpath("foci.csv"), index=False)

    foci_by_channel = {}
    if not foci_df.empty:
        for channel_name, channel_foci_df in foci_df.groupby("LAYER::CHANNEL_NAME"):
            foci_by_channel[channel_name] = channel_foci_df.reset_index(drop=True)

    return cells_df, cell_layers_df, foci_by_channel
