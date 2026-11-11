import numpy as np
from pandas import DataFrame


def transform_points(
    points_df: DataFrame,
    transforms: dict[str, np.ndarray],
    coordinate_0_key: str,
    coordinate_1_key: str,
    *,
    cell_id_key: str = "CELL::ID",
    symmetrize: str = "xy",
) -> np.ndarray:
    """Transform centroid-relative vectors in micrometers to a representative cell's coordinate grid.
    """
    rep_points = []
    for _, point in points_df.iterrows():
        cell_id = str(point[cell_id_key])
        if cell_id not in transforms:
            continue
        vector = np.asarray(
            [point[coordinate_0_key], point[coordinate_1_key]], dtype=float
        )
        if not np.all(np.isfinite(vector)):
            continue
        rep_points.append(transforms[cell_id] @ vector)

    points = np.asarray(rep_points, dtype=float).reshape(-1, 2)
    if "x" in symmetrize:
        points = np.concatenate([points, points * [-1, 1]])
    if "y" in symmetrize:
        points = np.concatenate([points, points * [1, -1]])
    return points
