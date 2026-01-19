import cv2
import numpy as np
from typing import Tuple


def apply_clahe_luminance(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    colorspace: str = "LAB",
) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel of a BGR image.

    This function is intended for human visual enhancement, especially for
    low-light or night-time images. It is NOT guaranteed to improve the
    performance of machine learning models.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format (uint8).
    clip_limit : float, optional
        Threshold for contrast limiting. Higher values give more contrast.
    tile_grid_size : Tuple[int, int], optional
        Size of grid for histogram equalization.
    colorspace : str, optional
        Color space to use for luminance separation. Supported: "LAB", "YCrCb".

    Returns
    -------
    np.ndarray
        BGR image with CLAHE applied to the luminance channel.
    """

    if image is None:
        raise ValueError("Input image is None")

    if image.dtype != np.uint8:
        raise ValueError("Input image must be uint8")

    if colorspace.upper() not in {"LAB", "YCRCB"}:
        raise ValueError("colorspace must be 'LAB' or 'YCrCb'")

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    if colorspace.upper() == "LAB":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))

        output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    else:  # YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_clahe = clahe.apply(y)
        ycrcb_clahe = cv2.merge((y_clahe, cr, cb))

        output = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    return output