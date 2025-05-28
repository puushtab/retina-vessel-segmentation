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

from segmentation_library import morphological_gradient_segmentation, external_gradient_segmentation
from validation_tuning_library import evaluate, tune_hyperparameters


# Load the original grayscale image
img = np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg').convert('L')).astype(np.uint8)
print(img.shape)

# Create a circular mask
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows/2)**2)
img_mask[invalid_pixels] = False


# Load the ground truth image as binary
img_GT = np.asarray(Image.open('./images_IOSTAR/GT_01.png').convert('L')).astype(bool)
# Ensure img_GT is binary (0s and 1s)
img_GT = img_GT > 0  # Convert to boolean array


# Tunning des hyperparamètres
# Define parameter grid to explore
param_grid = {
    'seuil': list(range(10, 120, 10)),        # threshold
    'selem_radius': list(range(1, 6))         # structuring element radius
}

# Run hyperparameter tuning
best_parameter, best_score, history = tune_hyperparameters(
    segmentation_func=external_gradient_segmentation,
    img=img,
    img_mask=img_mask,
    img_GT=img_GT,
    param_grid=param_grid,
    verbose=False
)


print("Best parameters:", best_parameter, "with F1 score:", best_score)

# Segment the image
# img_out = morphological_gradient_segmentation(img, img_mask, 15)
img_out = external_gradient_segmentation(img, img_mask, **best_parameter)

# Evaluate segmentation
ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU, ', Recall =', RECALL)

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