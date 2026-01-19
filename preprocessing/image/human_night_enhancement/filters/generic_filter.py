import cv2
import numpy as np

def apply_tactical_recognition(image: np.ndarray, mode: str = "heatmap") -> np.ndarray:
    """
    Transforma la imagen para maximizar el reconocimiento de objetos 
    sobre fondos oscuros (cielo). No busca estética.
    """
    # 1. Convertir a escala de grises (Luminancia)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Normalizar para usar todo el rango 0-255
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if mode == "negative":
        # Invertir: Objeto oscuro sobre fondo claro
        return cv2.bitwise_not(gray_norm)
    
    elif mode == "heatmap":
        # Aplicar un mapa de color (INFERNO o JET son excelentes para esto)
        # El cielo será negro/púrpura y el objeto será naranja/amarillo
        return cv2.applyColorMap(gray_norm, cv2.COLORMAP_INFERNO)
        
    return gray_norm