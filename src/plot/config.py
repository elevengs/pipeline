from argparse import Namespace, ArgumentParser
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Parameters controlling plot output."""

    lengthwise_scatter_intensity_colorbar_min: float | None
    lengthwise_scatter_intensity_colorbar_max: float | None
    lengthwise_scatter_kde_colorbar_min: float | None
    lengthwise_scatter_kde_colorbar_max: float | None
    lengthwise_scatter_num_foci_colorbar_min: int
    lengthwise_scatter_num_foci_colorbar_max: int | None

    lengthwise_scatter_intensity_y_min: float
    lengthwise_scatter_intensity_y_max: float
    lengthwise_scatter_kde_y_min: float
    lengthwise_scatter_kde_y_max: float
    lengthwise_scatter_num_foci_y_min: float
    lengthwise_scatter_num_foci_y_max: float

    rep_foci_colorbar_min: float | None
    rep_foci_colorbar_max: float | None
    rep_foci_symmetrize: str
    rep_foci_n_strata: int
    rep_foci_scalebar_size_um: float
    rep_foci_average_method: str

    rep_intensity_colorbar_min: float | None
    rep_intensity_colorbar_max: float | None
    rep_intensity_symmetrize: str
    rep_intensity_n_strata: int
    rep_intensity_scalebar_size_um: float
    rep_intensity_average_method: str

    def __post_init__(self):
        for prefix in (
            "lengthwise_scatter_intensity",
            "lengthwise_scatter_kde",
            "lengthwise_scatter_num_foci",
            "rep_foci",
            "rep_intensity",
        ):
            lower = getattr(self, f"{prefix}_colorbar_min")
            upper = getattr(self, f"{prefix}_colorbar_max")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{prefix}_colorbar_min must not exceed its maximum")
            if prefix.startswith("lengthwise_scatter"):
                lower = getattr(self, f"{prefix}_y_min")
                upper = getattr(self, f"{prefix}_y_max")
                if lower >= upper:
                    raise ValueError(f"{prefix}_y_min must be less than its maximum")
        for prefix in ("rep_foci", "rep_intensity"):
            symmetrize = getattr(self, f"{prefix}_symmetrize")
            if symmetrize not in ("", "x", "y", "xy"):
                raise ValueError(f"unsupported {prefix}_symmetrize {symmetrize}")
            average_method = getattr(self, f"{prefix}_average_method")
            if average_method not in ("mean", "tukey_biweight_location"):
                raise ValueError(f"unsupported {prefix}_average_method {average_method}")
            n_strata = getattr(self, f"{prefix}_n_strata")
            if n_strata < 1:
                raise ValueError(f"{prefix}_n_strata must be at least 1")
            scalebar_size = getattr(self, f"{prefix}_scalebar_size_um")
            if scalebar_size <= 0:
                raise ValueError(f"{prefix}_scalebar_size_um must be positive")

    def with_additional_config(
        self, additional_config: Namespace | None
    ) -> "PlotConfig":
        if additional_config is None:
            return self

        overrides = vars(additional_config)
        valid_fields = {field.name for field in fields(self)}
        unknown_fields = set(overrides) - valid_fields
        if unknown_fields:
            raise ValueError(
                "unsupported plot config option(s): "
                + ", ".join(sorted(unknown_fields))
            )
        return replace(self, **overrides)

    @classmethod
    def from_args(cls, args: Namespace) -> "PlotConfig":
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})


DEFAULT_PLOT_CONFIG = PlotConfig(
    lengthwise_scatter_intensity_colorbar_min=None,
    lengthwise_scatter_intensity_colorbar_max=None,
    lengthwise_scatter_kde_colorbar_min=0.0,
    lengthwise_scatter_kde_colorbar_max=4.0,
    lengthwise_scatter_num_foci_colorbar_min=1,
    lengthwise_scatter_num_foci_colorbar_max=None,

    lengthwise_scatter_intensity_y_min=0.8,
    lengthwise_scatter_intensity_y_max=2.1,
    lengthwise_scatter_kde_y_min=0.8,
    lengthwise_scatter_kde_y_max=2.1,
    lengthwise_scatter_num_foci_y_min=0.8,
    lengthwise_scatter_num_foci_y_max=2.1,

    rep_foci_colorbar_min=0.0,
    rep_foci_colorbar_max=4.0,
    rep_foci_symmetrize="xy",
    rep_foci_n_strata=4,
    rep_foci_scalebar_size_um=1.0,
    rep_foci_average_method="tukey_biweight_location",

    rep_intensity_colorbar_min=None,
    rep_intensity_colorbar_max=None,
    rep_intensity_symmetrize="xy",
    rep_intensity_n_strata=4,
    rep_intensity_scalebar_size_um=1.0,
    rep_intensity_average_method="tukey_biweight_location",
)


def add_plot_config_args(parser: ArgumentParser):
    """Register plot options with defaults matching the existing plots.
    """
    for plot in ("intensity", "kde", "num_foci"):
        prefix = f"lengthwise_scatter_{plot}"
        for bound in ("min", "max"):
            name = f"{prefix}_colorbar_{bound}"
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=int if plot == "num_foci" else float,
                default=getattr(DEFAULT_PLOT_CONFIG, name),
                help=(
                    f"{bound.capitalize()}imum colorbar value for the {plot} "
                    "lengthwise scatter plot - unset uses existing normalization; "
                    f"default: {getattr(DEFAULT_PLOT_CONFIG, name)}."
                ),
            )
            name = f"{prefix}_y_{bound}"
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=float,
                default=getattr(DEFAULT_PLOT_CONFIG, name),
                help=(
                    f"{bound.capitalize()}imum cell length in micrometers for the "
                    f"{plot} lengthwise scatter plot; "
                    f"default: {getattr(DEFAULT_PLOT_CONFIG, name)}."
                ),
            )

    for plot in ("foci", "intensity"):
        prefix = f"rep_{plot}"
        for bound in ("min", "max"):
            name = f"{prefix}_colorbar_{bound}"
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=float,
                default=getattr(DEFAULT_PLOT_CONFIG, name),
                help=(
                    f"{bound.capitalize()}imum colorbar value for representative "
                    f"{plot} plots - unset uses existing normalization; "
                    f"default: {getattr(DEFAULT_PLOT_CONFIG, name)}."
                ),
            )
        parser.add_argument(
            f"--rep-{plot}-symmetrize",
            choices=("", "x", "y", "xy"),
            default=getattr(DEFAULT_PLOT_CONFIG, f"{prefix}_symmetrize"),
            help=(
                "Axes to mirror across - use an empty string for no mirroring; "
                f"default: {getattr(DEFAULT_PLOT_CONFIG, f'{prefix}_symmetrize')}."
            ),
        )
        parser.add_argument(
            f"--rep-{plot}-n-strata",
            type=int,
            default=getattr(DEFAULT_PLOT_CONFIG, f"{prefix}_n_strata"),
            help=(
                "Number of cell-length strata for representative plots; "
                f"default: {getattr(DEFAULT_PLOT_CONFIG, f'{prefix}_n_strata')}."
            ),
        )
        parser.add_argument(
            f"--rep-{plot}-scalebar-size-um",
            type=float,
            default=getattr(DEFAULT_PLOT_CONFIG, f"{prefix}_scalebar_size_um"),
            help=(
                "Representative plot scalebar length in micrometers; "
                f"default: {getattr(DEFAULT_PLOT_CONFIG, f'{prefix}_scalebar_size_um')}."
            ),
        )
        parser.add_argument(
            f"--rep-{plot}-average-method",
            choices=("mean", "tukey_biweight_location"),
            default=getattr(DEFAULT_PLOT_CONFIG, f"{prefix}_average_method"),
            help=(
                "Method for averaging representative cell geometry; "
                f"default: {getattr(DEFAULT_PLOT_CONFIG, f'{prefix}_average_method')}."
            ),
        )
