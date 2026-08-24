from src.spideymaps.spideymaps_v2 import Spideymap, extend_spine
import numpy as np
from pathlib import Path
from skimage.measure import label
import imageio.v3 as iio
from skimage import (segmentation)
import shapely as sl

def get_stem(source) -> str:
    return str(source)[:-4]

def get_im(source: str) -> np.ndarray:
    return iio.imread(source)

def get_mask(source: str) -> np.ndarray:
    source = Path(source)
    return iio.imread(source.with_name(f"{source.stem}_labels.png"))
    
def get_labels(mask: np.ndarray) -> np.ndarray:
    return label(segmentation.clear_border(mask))


def create_rep_grid_colicoords(cc_params, grid_params):

    map = Spideymap()


    out = calc_outline(**cc_params)
    out = sl.LinearRing(out)

    mid = calc_midline(
        np.linspace(cc_params['xl'], cc_params['xr'], 100), 
        a0 = cc_params['a0'], 
        a1 = cc_params['a1'], 
        a2 = cc_params['a2']
        )
    
    mid = sl.LineString(mid)
    mid = extend_spine(mid, out)
    map.make_grid(out=out, mid=mid, **grid_params)

    return map

def calc_midline(x_arr, a0, a1, a2):
    """From colicoords.
    Calculate p(x).

    The function p(x) describes the midline of the cell.

    Parameters
    ----------
    x_arr : :class:`~numpy.ndarray`
        Input x values.
    a0, a1, a2
        Coefficients for 2nd order polynomial.

    Returns
    -------
    p : :class:`~numpy.ndarray`
        Evaluated polynomial p(x)
    """
    y = a0 + a1 * x_arr + a2 * x_arr ** 2
    mid = np.array([x_arr, y]).T
    
    return mid


def calc_outline(xl, xr, a0, a1, a2, r):
    """
    From colicoords.
    Plot the outline of the cell based on the current coordinate system.

    The outline consists of two semicircles and two offset lines to the central parabola.[1]_[2]_

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`, optional
        Matplotlib axes to use for plotting.
    **kwargs
        Additional kwargs passed to ax.plot().

    Returns
    -------
    line : :class:`~matplotlib.lines.Line2D`
        Matplotlib line artist object.


    .. [1] T. W. Sederberg. "Computer Aided Geometric Design". Computer Aided Geometric Design Course Notes.
        January 10, 2012
    .. [2] Rida T.Faroukia, Thomas W. Sederberg, Analysis of the offset to a parabola, Computer Aided Geometric Design
        vol 12, issue 6, 1995

    """

    # Parametric plotting of offset line
    # http://cagd.cs.byu.edu/~557/text/ch8.pdf
    #
    # Analysis of the offset to a parabola
    # https://doi-org.proxy-ub.rug.nl/10.1016/0167-8396(94)00038-T

    numpoints = 500
    t = np.linspace(xl, xr, num=numpoints)
    # a0, a1, a2 = self.cell_obj.coords.coeff

    x_top = t + r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_top = a0 + a1*t + a2*(t**2) - r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    x_bot = t + - r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_bot = a0 + a1*t + a2*(t**2) + r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    # Left semicirlce
    psi = np.arctan(-p_dx(xl, a1, a2))

    th_l = np.linspace(-0.5*np.pi+psi, 0.5*np.pi + psi, num=200)
    cl_dx = r * np.cos(th_l)
    cl_dy = r * np.sin(th_l)

    cl_x = xl - cl_dx
    cl_y = calc_midline(xl, a0, a1, a2)[1] + cl_dy

    #Right semicircle
    psi = np.arctan(-p_dx(xr, a1, a2))

    th_r = np.linspace(0.5*np.pi - psi, -0.5*np.pi - psi, num=200)
    cr_dx = r * np.cos(th_r)
    cr_dy = r * np.sin(th_r)

    cr_x = cr_dx + xr
    cr_y = cr_dy + calc_midline(xr, a0, a1, a2)[1]

    x_all = np.concatenate((cl_x[::-1], x_top, cr_x[::-1], x_bot[::-1]))
    y_all = np.concatenate((cl_y[::-1], y_top, cr_y[::-1], y_bot[::-1]))

    out = np.array([x_all, y_all]).T

    return out

def p_dx(x_arr, a1, a2):
    """
    Calculate the derivative p'(x) evaluated at x.

    Parameters
    ----------
    x_arr :class:`~numpy.ndarray`:
        Input x values.

    Returns
    -------
    p_dx : :class:`~numpy.ndarray`:
        Evaluated function p'(x).
    """
    return a1 + 2 * a2 * x_arr
