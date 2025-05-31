import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

from skimage.filters import threshold_local
from scipy.ndimage import gaussian_filter

from segmentation_library import *


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


def seuillage_local(img, img_mask, block_size=35, offset=10, inverse=False):
    """
    Segmentation par seuillage adaptatif local après floutage.

    Args:
        img: image niveau de gris (uint8)
        img_mask: masque de validité (booléen)
        block_size: taille de la fenêtre pour le seuillage local (impair)
        offset: valeur soustraite au seuillage local pour rendre la binarisation plus stricte

    Returns:
        Image binaire (booléenne)
    """
    # Floutage léger pour réduire le bruit
    img_blur = gaussian_filter(img, sigma=1)

    # Seuillage adaptatif local
    adaptive_thresh = threshold_local(img_blur, block_size, offset=offset)

    # Binarisation + masque
    if inverse:
        img_out = (img_blur > adaptive_thresh) & img_mask
    else:
        img_out = (img_blur < adaptive_thresh) & img_mask
    return img_out

def seuillage_global(img, img_mask, seuil, inverse=False):
    # Seuillage global
    if inverse:
        img_out = (img_mask & (img > seuil))
    else:
        img_out = (img_mask & (img < seuil))
    return img_out

def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT)  # Reduce evaluation to skeleton pixels
    img_out_skel = skeletonize(img_out)  # Skeletonize segmented image
    TP = np.sum(img_out_skel & img_GT)  # True positives
    FP = np.sum(img_out_skel & ~img_GT)  # False positives
    FN = np.sum(GT_skel & ~img_out)  # False negatives

    ACCU = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    return ACCU, RECALL, img_out_skel, GT_skel

def tophat(img, elt, black=True):
    if black:
        return black_tophat(img, elt)
    return tophat(img, elt)

# Load the original grayscale image
img = np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg').convert('L')).astype(np.uint8)
print(img.shape)

# Create a circular mask
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows/2)**2)
img_mask[invalid_pixels] = False

# Segment the image (seuillage)
# img_out = seuillage_global(img, img_mask, 60)

# Tophat
adapt_tresh = adaptive_treshold_segmentation(img, img_mask, adaptive_method='median', block_size=17, C=3, blur=True)

img_closed = closing(adapt_tresh, disk(1))
img_open = opening(img_closed, disk(3))
rec1 = reconstruction(img_open, img_closed).astype(int)
img_closed = closing(rec1, disk(2))
img_open = opening(img_closed, disk(2))
rec2 = reconstruction(img_open, img_closed).astype(int)
img_out = rec2

# Load the ground truth image as binary
img_GT = np.asarray(Image.open('./images_IOSTAR/GT_01.png').convert('L')).astype(bool)
# Ensure img_GT is binary (0s and 1s)
img_GT = img_GT > 0  # Convert to boolean array

# Evaluate segmentation
ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU, ', Recall =', RECALL)
print('F1 score =', f1_score(ACCU, RECALL))

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out, cmap='gray')
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel, cmap='gray')
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT, cmap='gray')
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel, cmap='gray')
plt.title('Verite Terrain Squelette')
plt.tight_layout()
plt.show()