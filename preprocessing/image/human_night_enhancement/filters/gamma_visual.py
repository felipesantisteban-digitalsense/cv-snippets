import cv2
import numpy as np
from typing import Tuple


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = 1.5,
    colorspace: str = "LAB",
) -> np.ndarray:
    """
    Apply Gamma Correction to the luminance channel of a BGR image.

    Human eyes perceive brightness non-linearly. Gamma correction adjusts the 
    luminance to match this perception, stretching details in the shadows 
    without over-saturating highlights.

    Formula: Output = Input ^ (1 / gamma)

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format (uint8).
    gamma : float, optional
        Gamma value. 
        - gamma > 1.0: Brightens shadows (ideal for night vision).
        - gamma < 1.0: Darkens the image.
        Typical values for night enhancement: 1.2 to 2.2.
    colorspace : str, optional
        Color space to use for luminance separation. Supported: "LAB", "YCrCb".

    Returns
    -------
    np.ndarray
        BGR image with gamma correction applied to the luminance channel.
    """

    if image is None:
        raise ValueError("Input image is None")

    if image.dtype != np.uint8:
        raise ValueError("Input image must be uint8")

    if colorspace.upper() not in {"LAB", "YCRCB"}:
        raise ValueError("colorspace must be 'LAB' or 'YCrCb'")

    # 1. Build a Look-Up Table (LUT) for performance
    # This avoids calculating the power function for every single pixel.
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in np.arange(0, 256)
    ]).astype("uint8")

    # 2. Convert to the desired colorspace and split channels
    if colorspace.upper() == "LAB":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(converted)
        
        # Apply LUT to the L (luminance) channel
        l_gamma = cv2.LUT(l, table)
        
        output_converted = cv2.merge((l_gamma, a, b))
        output = cv2.cvtColor(output_converted, cv2.COLOR_LAB2BGR)
        
    else:  # YCrCb
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(converted)
        
        # Apply LUT to the Y (luminance) channel
        y_gamma = cv2.LUT(y, table)
        
        output_converted = cv2.merge((y_gamma, cr, cb))
        output = cv2.cvtColor(output_converted, cv2.COLOR_YCrCb2BGR)

    return output