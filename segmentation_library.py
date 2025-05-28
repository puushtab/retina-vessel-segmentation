from skimage.morphology import dilation, erosion, disk, closing
from skimage.segmentation import watershed
from skimage.filters import frangi, sobel
from skimage.feature import peak_local_max
from skimage.exposure import rescale_intensity
from scipy import ndimage as ndi
import numpy as np


def morphological_gradient_segmentation(img, img_mask, seuil, selem_radius=1):
    """
    Applique un seuillage sur le gradient morphologique pour segmenter une image.

    Parameters:
    -----------
    img : ndarray
        Image en niveaux de gris (uint8).
    img_mask : ndarray
        Masque binaire des régions valides (True/False).
    seuil : int or float
        Seuil d’intensité appliqué au gradient.
    selem_radius : int
        Rayon de l’élément structurant utilisé pour la morphologie.

    Returns:
    --------
    img_out : ndarray
        Image binaire résultant de la segmentation.
    """
    selem = disk(selem_radius)
    dilated = dilation(img, selem)
    eroded = erosion(img, selem)
    grad = dilated - eroded
    img_out = (img_mask & (grad > seuil))
    return img_out


def external_gradient_segmentation(img, img_mask, seuil, selem_radius=1):
    """
    Gradient morphologique externe + seuillage dans le masque.

    Gradient externe = Dilatation - Image originale.

    Parameters:
    -----------
    img : ndarray
        Image grayscale.
    img_mask : ndarray
        Binary mask (True inside valid region).
    seuil : int
        Threshold for segmentation.
    selem_radius : int
        Structuring element radius.

    Returns:
    --------
    img_out : ndarray
        Binary segmented image.
    """
    selem = disk(selem_radius)
    grad_ext = dilation(img, selem) - img
    img_out = (img_mask & (grad_ext > seuil))
    return img_out


def gradient_based_segmentation(img, img_mask, seuil_grad, seuil_closing, selem_radius_grad=1, selem_radius_closing=1):
    """
    Applique une segmentation basée sur le gradient morphologique externe suivi
    d'une fermeture morphologique et d'un seuillage supplémentaire pour combler
    les discontinuités et affiner la segmentation.

    Parameters:
    -----------
    img : ndarray
        Image en niveaux de gris (uint8).
    img_mask : ndarray
        Masque binaire des régions valides (True/False).
    seuil_grad : int or float
        Seuil d’intensité appliqué au gradient externe.
    seuil_closing : int or float
        Seuil d’intensité appliqué après la fermeture morphologique.
    selem_radius_grad : int
        Rayon de l’élément structurant pour le calcul du gradient.
    selem_radius_closing : int
        Rayon de l’élément structurant pour la fermeture morphologique.

    Returns:
    --------
    img_out : ndarray
        Image binaire résultant de la segmentation.
    """
    # Créer les éléments structurants pour le gradient et la fermeture
    selem_grad = disk(selem_radius_grad)
    selem_closing = disk(selem_radius_closing)
    
    # Calculer le gradient morphologique externe (dilatation - image originale)
    grad_ext = dilation(img, selem_grad) - img
    
    # Appliquer le premier seuillage dans le masque
    img_out = img_mask & (grad_ext > seuil_grad)
    
    # Appliquer une fermeture morphologique pour combler les petits trous
    img_out = closing(img_out, selem_closing)
    
    # Appliquer un second seuillage sur le gradient pour affiner la segmentation
    img_out = img_mask & (img_out & (grad_ext > seuil_closing))
    
    return img_out

# Updated segmentation function with Frangi filter
def watershed_segmentation_with_markers(img, img_mask, seuil, min_distance=15, scale_min=1, scale_max=10):
    """
    Segment retinal vessels using watershed with markers derived from Frangi vessel enhancement.

    Parameters:
    -----------
    img : ndarray
        Grayscale input image (uint8).
    img_mask : ndarray
        Binary mask of valid regions.
    seuil : int
        Threshold for Frangi vesselness map (0-255).
    min_distance : int
        Minimum distance between marker peaks.
    scale_min : int
        Minimum scale for Frangi filter (vessel width).
    scale_max : int
        Maximum scale for Frangi filter (vessel width).

    Returns:
    --------
    img_out : ndarray
        Binary segmented image.
    """
    # Compute vesselness map with Frangi filter
    vesselness = frangi(img, scale_range=(scale_min, scale_max), scale_step=2, beta=0.9, gamma=15)
    # Rescale to 0-255 for consistency with previous threshold range
    vesselness_scaled = rescale_intensity(vesselness, out_range=(0, 255)).astype(np.uint8)

    # Threshold the vesselness map
    vessel_thresh = vesselness_scaled > seuil

    # Generate vessel markers using distance transform and local maxima
    distance = ndi.distance_transform_edt(vessel_thresh)
    coords = peak_local_max(distance, labels=vessel_thresh, min_distance=min_distance)

    # Generate background markers
    eroded_mask = erosion(img_mask, disk(3))
    background_markers = img_mask & ~eroded_mask

    # Initialize markers array
    markers = np.zeros_like(img, dtype=np.int32)
    markers[background_markers] = 1  # Background label
    for i, coord in enumerate(coords, start=2):
        markers[coord[0], coord[1]] = i  # Vessel markers starting from 2

    # Compute gradient of original image for watershed
    grad = sobel(img)
    grad_scaled = rescale_intensity(grad, out_range=(0, 255)).astype(np.uint8)

    # Apply watershed segmentation
    labels = watershed(grad_scaled, markers=markers, mask=img_mask)

    # Binary output: vessels are regions with labels > 1
    img_out = labels > 1

    return img_out
