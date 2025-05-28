from skimage.morphology import dilation, erosion, disk

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
