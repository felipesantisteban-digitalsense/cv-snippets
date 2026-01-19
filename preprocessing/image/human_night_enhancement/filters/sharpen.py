import cv2
import numpy as np
from typing import Tuple


def apply_unsharp_mask_luminance(
    image: np.ndarray,
    strength: float = 1.5,
    sigma: float = 1.0,
    colorspace: str = "LAB",
) -> np.ndarray:
    """
    Apply an Unsharp Mask filter to the luminance channel of a BGR image.

    This technique enhances the edges and fine details by subtracting a 
    blurred version of the image from the original, effectively increasing 
    the acutance of the image for better human recognition.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format (uint8).
    strength : float, optional
        Weight of the sharpening effect. Typical values range from 0.5 to 2.0.
    sigma : float, optional
        Standard deviation of the Gaussian blur. Controls the "radius" of the 
        details to be sharpened. Smaller values sharpen fine details.
    colorspace : str, optional
        Color space to use for luminance separation. Supported: "LAB", "YCrCb".

    Returns
    -------
    np.ndarray
        BGR image with sharpened luminance channel.
    """

    if image is None:
        raise ValueError("Input image is None")

    if image.dtype != np.uint8:
        raise ValueError("Input image must be uint8")

    if colorspace.upper() not in {"LAB", "YCRCB"}:
        raise ValueError("colorspace must be 'LAB' or 'YCrCb'")

    # 1. Convert to the desired colorspace and split channels
    if colorspace.upper() == "LAB":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(converted)
        luminance = l
    else:  # YCrCb
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(converted)
        luminance = y

    # 2. Apply Unsharp Masking logic:
    # Formula: Sharpened = Original + strength * (Original - Blurred)
    
    # We work with float32 for the calculation to avoid overflow/underflow
    luminance_f = luminance.astype(np.float32)
    
    # Create the "unsharp" mask by blurring the luminance
    blurred = cv2.GaussianBlur(luminance_f, (0, 0), sigma)
    
    # Calculate the sharpened luminance
    sharpened_f = cv2.addWeighted(luminance_f, 1.0 + strength, blurred, -strength, 0)

    # Clip values to [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened_f, 0, 255).astype(np.uint8)

    # 3. Merge channels and convert back to BGR
    if colorspace.upper() == "LAB":
        output_converted = cv2.merge((sharpened, a, b))
        output = cv2.cvtColor(output_converted, cv2.COLOR_LAB2BGR)
    else:
        output_converted = cv2.merge((sharpened, cr, cb))
        output = cv2.cvtColor(output_converted, cv2.COLOR_YCrCb2BGR)

    return output