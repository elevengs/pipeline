from ..map import grid_params
import numpy as np
import shapely as sl
from src.spideymaps.spideymaps_v2 import Spideymap, extend_spine


def create_rep_grid_colicoords(cc_params, grid_params):
    map = Spideymap()

    out = calc_outline(**cc_params)
    out = sl.LinearRing(out)

    mid = calc_midline(
        np.linspace(cc_params["xl"], cc_params["xr"], 100),
        a0=cc_params["a0"],
        a1=cc_params["a1"],
        a2=cc_params["a2"],
    )

    mid = sl.LineString(mid)
    mid = extend_spine(mid, out)
    map.make_grid(out=out, mid=mid, **grid_params)

    return map


def calc_midline(x_arr, a0, a1, a2):
    y = a0 + a1 * x_arr + a2 * x_arr**2
    mid = np.array([x_arr, y]).T

    return mid


def calc_outline(xl, xr, a0, a1, a2, r):
    numpoints = 500
    t = np.linspace(xl, xr, num=numpoints)

    x_top = t + r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_top = a0 + a1 * t + a2 * (t**2) - r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    x_bot = t + -r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_bot = a0 + a1 * t + a2 * (t**2) + r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    psi = np.arctan(-p_dx(xl, a1, a2))

    th_l = np.linspace(-0.5 * np.pi + psi, 0.5 * np.pi + psi, num=200)
    cl_dx = r * np.cos(th_l)
    cl_dy = r * np.sin(th_l)

    cl_x = xl - cl_dx
    cl_y = calc_midline(xl, a0, a1, a2)[1] + cl_dy

    psi = np.arctan(-p_dx(xr, a1, a2))

    th_r = np.linspace(0.5 * np.pi - psi, -0.5 * np.pi - psi, num=200)
    cr_dx = r * np.cos(th_r)
    cr_dy = r * np.sin(th_r)

    cr_x = cr_dx + xr
    cr_y = cr_dy + calc_midline(xr, a0, a1, a2)[1]

    x_all = np.concatenate((cl_x[::-1], x_top, cr_x[::-1], x_bot[::-1]))
    y_all = np.concatenate((cl_y[::-1], y_top, cr_y[::-1], y_bot[::-1]))

    out = np.array([x_all, y_all]).T

    return out


def p_dx(x_arr, a1, a2):
    return a1 + 2 * a2 * x_arr


def get_rep(length_ranges, cell_lengths_um, pixel_size_um):
    cell_lengths_um = np.asarray(cell_lengths_um)
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
