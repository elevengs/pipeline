from argparse import Namespace, ArgumentParser
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class FocusDetectionConfig:
    """Parameters controlling fluorescence preprocessing and focus detection."""

    detection_threshold_method: str
    absolute_detection_threshold: float
    IQR_multiple: float
    min_sigma: float
    max_sigma: float
    num_sigma: float
    overlap: float
    expand_by: float
    bg_sub_radius: float
    denoise_radius: float
    denoise_amount: float
    gaussian_sigma: float
    plot_crop_margin: int

    @classmethod
    def from_args(cls, args: Namespace) -> "FocusDetectionConfig":
        threshold_method = args.detection_threshold_method
        if threshold_method not in ("absolute", "IQR_multiple"):
            raise ValueError(
                f"unsupported detection threshold method {threshold_method}"
            )

        return cls(
            detection_threshold_method=threshold_method,
            absolute_detection_threshold=args.absolute_detection_threshold,
            IQR_multiple=args.IQR_multiple,
            min_sigma=args.detect_min_sigma,
            max_sigma=args.detect_max_sigma,
            num_sigma=args.detect_num_sigma,
            overlap=args.overlap,
            expand_by=args.expand_by,
            bg_sub_radius=args.bg_sub_radius,
            denoise_radius=args.denoise_radius,
            denoise_amount=args.denoise_amount,
            gaussian_sigma=args.gaussian_sigma,
            plot_crop_margin=args.plot_crop_margin
        )


DEFAULT_DETECTION_CONFIG = FocusDetectionConfig(
    detection_threshold_method="IQR_multiple",
    absolute_detection_threshold=1.0,
    IQR_multiple=2.0,
    min_sigma=2.0,
    max_sigma=3.8,
    num_sigma = 20,
    overlap=0.75,
    expand_by=2.0,
    bg_sub_radius=1.5,
    denoise_radius=2.0,
    denoise_amount=6.0,
    gaussian_sigma=1.2,
    plot_crop_margin=3
)

def add_detection_config_args(parser: ArgumentParser):
    parser.add_argument(
        "--detection-threshold-method",
        type=str,
        default=DEFAULT_DETECTION_CONFIG.detection_threshold_method,
        help=f"What method to use to determine the intensity threshold for focus detection. 'absolute' will use the value provided for --absolute-detection-threshold, and 'IQR_multiple' will use the median plus a multiple of the interquartile range of the pixel intensities inside cell boundaries; default: {DEFAULT_DETECTION_CONFIG.detection_threshold_method}",
    )
    parser.add_argument(
        "--absolute-detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.absolute_detection_threshold,
        help = f"What absolute intensity value a putative focus must surpass in order to be detected; default: {str(DEFAULT_DETECTION_CONFIG.absolute_detection_threshold)}."
    )
    parser.add_argument(
        "--IQR-multiple",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.IQR_multiple,
        help=f"Multiple of the IQR above the median fluorescence intensity inside the foreground (all cells) which a focus must surpass to qualify for detection. Decrease this to detect dimmer foci; default: {str(DEFAULT_DETECTION_CONFIG.IQR_multiple)}.",
    )
    parser.add_argument(
        "--detect-min-sigma",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.min_sigma,
        help=f"Minimum Gaussian blur kernel standard deviation used for Laplacian of Gaussian foci detection. Decrease this to detect smaller foci; default: {str(DEFAULT_DETECTION_CONFIG.min_sigma)}",
    )
    parser.add_argument(
        "--detect-max-sigma",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.max_sigma,
        help=f"Maximum Gaussian blur kernel standard deviation used for Laplacian of Gaussian foci detection. Increase this to detect larger foci; default: {str(DEFAULT_DETECTION_CONFIG.max_sigma)}.",
    )

    parser.add_argument(
        "--detect-num-sigma",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.num_sigma,
        help=f"Number of Gaussian blur kernel standard deviations used for Laplacian of Gaussian foci detection. Increase this to differentiate foci more precisely as to their size; default: {str(DEFAULT_DETECTION_CONFIG.num_sigma)}.",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.overlap,
        help=f"Maximum percent overlap between two foci before the smaller is discarded; default: {str(DEFAULT_DETECTION_CONFIG.overlap)}.",
    )
    parser.add_argument(
        "--expand-by",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.expand_by,
        help=f"Distance (in pixels) to expand each cell label in every direction by for the purpose of assigning foci to it. Increase this to detect foci further outside the edge of a cell label; default: {str(DEFAULT_DETECTION_CONFIG.expand_by)}",
    )
    parser.add_argument(
        "--plot-crop-margin",
        type=int,
        help=f"Margin (in pixels) to show around the cell contour for the stepwise plots; default: {str(DEFAULT_DETECTION_CONFIG.plot_crop_margin)}.",
        default = DEFAULT_DETECTION_CONFIG.plot_crop_margin
    )
    parser.add_argument(
        "--bg-sub-radius",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.bg_sub_radius,
        help=f"Width (in pixels) of the kernel for background subtraction. Increase this to take into account further out pixels in assigning the background intensity for a given pixel; default: {str(DEFAULT_DETECTION_CONFIG.bg_sub_radius)}",
    )
    parser.add_argument(
        "--denoise-radius",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.denoise_radius,
        help=f"Width (in pixels) of the kernel for denoising. Increase this to smooth features on a larger scale in the denoising stage; default: {str(DEFAULT_DETECTION_CONFIG.denoise_radius)}.",
    )
    parser.add_argument(
        "--denoise-amount",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.denoise_amount,
        help=f"Magnitude of the denoise operation. Increase this to smooth noise features to a greater extent; default: {str(DEFAULT_DETECTION_CONFIG.denoise_amount)}.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=DEFAULT_DETECTION_CONFIG.gaussian_sigma,
        help=f"Standard deviation of the Gaussian blur kernel. Increase this to blur together features that are farther apart; default: {str(DEFAULT_DETECTION_CONFIG.gaussian_sigma)}.",
    )