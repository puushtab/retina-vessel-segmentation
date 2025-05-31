from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile

from skimage.segmentation import watershed
from skimage.filters import frangi, sobel
from skimage.feature import peak_local_max
from skimage.exposure import rescale_intensity
from scipy import ndimage as ndi
import numpy as np
import cv2

def adaptive_treshold_segmentation(img, img_mask=None, adaptive_method='mean', block_size=11, C=2, blur=False):
    """
    Applique un seuillage adaptatif local sur une image (canal vert ou niveaux de gris).
    
    Parameters:
    - img: Image d'entrée (couleur ou niveaux de gris).
    - img_mask: Masque binaire optionnel (même taille que l'image).
    - adaptive_method: 'mean' pour moyenne locale, 'median' pour médiane locale.
    - block_size: Taille du voisinage pour le calcul local (doit être impair).
    - C: Constante soustraite du seuil local.
    - blur: Si True, applique un floutage Gaussien avant le seuillage.
    
    Returns:
    - Image binaire (uint8, 0 ou 255) après seuillage adaptatif.
    """
    # Vérifier les dimensions de l'image
    if len(img.shape) == 3:
        # Extraire le canal vert (index 1 dans BGR)
        img_proc = img[:, :, 1]
    else:
        # Image déjà en niveaux de gris
        img_proc = img

    # Appliquer un floutage Gaussien si demandé
    if blur:
        img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)

    # Vérifier que block_size est impair
    if block_size % 2 == 0:
        raise ValueError("block_size doit être impair.")

    # Seuillage adaptatif
    if adaptive_method.lower() == 'mean':
        # Utiliser la moyenne locale (OpenCV)
        thresh = cv2.adaptiveThreshold(
            img_proc,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
    elif adaptive_method.lower() == 'median':
        # Calcul manuel pour la médiane locale
        # Appliquer un filtre médian local pour estimer le seuil
        local_median = cv2.medianBlur(img_proc, block_size)
        thresh = (img_proc < local_median - C).astype(np.uint8) * 255
    else:
        raise ValueError("adaptive_method doit être 'mean' ou 'median'.")

    # Appliquer le masque si fourni
    if img_mask is not None:
        if img_mask.shape != img_proc.shape:
            raise ValueError("Le masque doit avoir la même forme que l'image.")
        # Convertir le masque en binaire (0 ou 255)
        img_mask = (img_mask > 0).astype(np.uint8) * 255
        thresh = cv2.bitwise_and(thresh, img_mask)

    return thresh


def adaptive_treshold_segmentation_with_opening(
    img,
    img_mask=None,
    adaptive_method='mean',
    block_size=11,
    C=2,
    blur=False,
    opening_radius=1
):
    """
    Applique un seuillage adaptatif local suivi d'une ouverture morphologique.

    Parameters:
    - img: Image d'entrée (niveaux de gris ou couleur).
    - img_mask: Masque binaire optionnel (même taille que l'image).
    - adaptive_method: 'mean' ou 'median' pour le calcul local du seuil.
    - block_size: Taille de la fenêtre locale (impair).
    - C: Constante soustraite du seuil local.
    - blur: Si True, applique un floutage Gaussien avant seuillage.
    - opening_radius: Rayon de l'élément structurant pour l'ouverture.

    Returns:
    - Image binaire (uint8, 0 ou 255) après seuillage adaptatif et ouverture.
    """
    if len(img.shape) == 3:
        img_proc = img[:, :, 1]  # canal vert
    else:
        img_proc = img

    if blur:
        img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)

    if block_size % 2 == 0:
        raise ValueError("block_size doit être impair.")

    if adaptive_method.lower() == 'mean':
        thresh = cv2.adaptiveThreshold(
            img_proc,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
    elif adaptive_method.lower() == 'median':
        local_median = cv2.medianBlur(img_proc, block_size)
        thresh = (img_proc < local_median - C).astype(np.uint8) * 255
    else:
        raise ValueError("adaptive_method doit être 'mean' ou 'median'.")

    if img_mask is not None:
        if img_mask.shape != img_proc.shape:
            raise ValueError("Le masque doit avoir la même forme que l'image.")
        img_mask = (img_mask > 0).astype(np.uint8) * 255
        thresh = cv2.bitwise_and(thresh, img_mask)

    # Appliquer une ouverture morphologique pour supprimer les artefacts fins
    binary = (thresh > 0).astype(np.uint8)
    selem = disk(opening_radius)
    opened = opening(binary, selem)
    
    return (opened * 255).astype(np.uint8)


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
        Seuil d'intensité appliqué au gradient.
    selem_radius : int
        Rayon de l'élément structurant utilisé pour la morphologie.

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
        Seuil d'intensité appliqué au gradient externe.
    seuil_closing : int or float
        Seuil d'intensité appliqué après la fermeture morphologique.
    selem_radius_grad : int
        Rayon de l'élément structurant pour le calcul du gradient.
    selem_radius_closing : int
        Rayon de l'élément structurant pour la fermeture morphologique.

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

# New function for multi-scale skeletonization
def multi_scale_skeletonization(img, img_mask=None, scales=[1, 2, 3]):
    """
    Applique une squelettisation multi-échelle sur une image binaire.
    
    Parameters:
    - img: Image binaire (uint8, 0 ou 255).
    - img_mask: Masque binaire optionnel (même taille que l'image).
    - scales: Liste des rayons d'éléments structurants à utiliser.
    
    Returns:
    - Image binaire (uint8, 0 ou 255) après squelettisation multi-échelle.
    """
    if img_mask is not None:
        if img_mask.shape != img.shape:
            raise ValueError("Le masque doit avoir la même forme que l'image.")
        img = cv2.bitwise_and(img, img_mask)
    
    # Convertir l'image en binaire (0 ou 1)
    binary = (img > 0).astype(np.uint8)
    
    # Initialiser le résultat
    result = np.zeros_like(binary)
    
    # Appliquer la squelettisation à chaque échelle
    for scale in scales:
        selem = disk(scale)
        # Utiliser l'opération d'érosion pour simuler la squelettisation
        skeleton = erosion(binary, selem)
        result = np.logical_or(result, skeleton)
    
    # Convertir le résultat en uint8 (0 ou 255)
    return (result * 255).astype(np.uint8)

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def skeleton_multi_scale(binary_img, s=1):
    """
    Calcule des squelettes multi-échelles à partir d'une image binaire.
    
    Args:
        binary_img: Image binaire (booléenne) dont on veut extraire les squelettes.
        scales: Liste de valeurs d'échelle (distance au bord minimale).

    Returns:
        Dictionnaire {échelle: squelette}
    """
    distance_map = distance_transform_edt(binary_img)
    mask = distance_map > s
    skel = skeletonize(mask)
    return skel

def skeleton_segmentation(img, img_mask=None, s=1):
    """
    Applique une squelettisation sur une image binaire.
    """
    img_out = adaptive_treshold_segmentation(img, img_mask,  adaptive_method='median', block_size=17, C=5, blur=True)

    return skeleton_multi_scale(img_out, s)

def skeleton_multi_scale2(binary_img, scales=[1, 3, 5, 10]):
    """
    Calcule des squelettes multi-échelles à partir d'une image binaire.
    
    Args:
        binary_img: Image binaire (booléenne) dont on veut extraire les squelettes.
        scales: Liste de valeurs d'échelle (distance au bord minimale).

    Returns:
        Dictionnaire {échelle: squelette}
    """
    distance_map = distance_transform_edt(binary_img)
    skeletons = {}
    for s in scales:
        mask = distance_map > s
        skel = skeletonize(mask)
        skeletons[s] = skel
    return skeletons

def tophat(img, radius, black=True):
    """
    Applique une opération de tophat (black ou white) sur une image.
    
    Parameters:
    -----------
    img : ndarray
        Image en niveaux de gris (uint8).
    radius : int
        Rayon de l'élément structurant.
    black : bool
        Si True, applique un black tophat (fermeture - image).
        Si False, applique un white tophat (image - ouverture).
    
    Returns:
    --------
    ndarray
        Image résultant de l'opération de tophat.
    """
    # Créer l'élément structurant
    selem = disk(radius)
    
    if black:
        # Black tophat = fermeture - image originale
        closed = closing(img, selem)
        return closed - img
    else:
        # White tophat = image originale - ouverture
        opened = opening(img, selem)
        return img - opened

def tophat_segmentation(img, img_mask=None, radius=3, black=True, threshold=30):
    """
    Applique une segmentation basée sur l'opération de tophat pour détecter les vaisseaux.
    
    Parameters:
    -----------
    img : ndarray
        Image en niveaux de gris (uint8).
    img_mask : ndarray
        Masque binaire des régions valides (True/False).
    radius : int
        Rayon de l'élément structurant pour le tophat.
    black : bool
        Si True, utilise black tophat (pour vaisseaux sombres).
        Si False, utilise white tophat (pour vaisseaux clairs).
    threshold : int
        Seuil pour binariser le résultat du tophat.
    
    Returns:
    --------
    ndarray
        Image binaire résultant de la segmentation.
    """
    # Appliquer l'opération de tophat
    tophat_result = tophat(img, radius, black)
    
    # Binariser le résultat
    binary = tophat_result > threshold
    
    # Appliquer le masque si fourni
    if img_mask is not None:
        binary = binary & img_mask
    
    return binary.astype(np.uint8) * 255


def reconstruction_segmentation(img, img_mask=None, radius=3, black=True, threshold=30):
    """
    Applique une segmentation basée sur l'opération de reconstruction pour détecter les vaisseaux.
    
    Parameters:
    -----------
    img : ndarray
        Image en niveaux de gris (uint8).
    img_mask : ndarray
        Masque binaire des régions valides (True/False).
    radius : int
        Rayon de l'élément structurant pour la reconstruction.
    black : bool
        Si True, utilise la reconstruction par érosion (pour vaisseaux sombres).
        Si False, utilise la reconstruction par dilatation (pour vaisseaux clairs).
    threshold : int
        Seuil pour binariser le résultat de la reconstruction.
    
    Returns:
    --------
    ndarray
        Image binaire résultant de la segmentation.
    """
    # Créer l'élément structurant
    adapt_thresh = adaptive_treshold_segmentation(img, img_mask,  adaptive_method='median', block_size=17, C=5, blur=True)
    
    elt = disk(radius)
    reconstruction_mask = opening(adapt_thresh, elt)
    img_out = reconstruction(reconstruction_mask, adapt_thresh).astype(int)


    return img_out

def alternating_filter(img, radius=1, num_repetitions=2):
    """
    Applique un filtre alterné (opening -> closing -> opening -> closing -> ...) pour lisser le bruit.
    
    Parameters:
    -----------
    img : ndarray
        Image binaire (uint8, 0 ou 255).
    radius : int
        Rayon de l'élément structurant.
    num_repetitions : int
        Nombre de répétitions de la séquence (opening -> closing).
    
    Returns:
    --------
    ndarray
        Image binaire après application du filtre alterné.
    """
    result = img.copy()
    
    for _ in range(num_repetitions):
        selem = disk(radius)
        # Opening
        result = opening(result, selem)
        # Closing
        result = closing(result, selem) 
        radius += 1
    
    return result
    
    