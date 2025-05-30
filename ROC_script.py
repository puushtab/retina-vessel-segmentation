import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from segmentation_library import (
    adaptive_treshold_segmentation,
    adaptive_treshold_segmentation_with_opening,
    morphological_gradient_segmentation,
    external_gradient_segmentation,
    gradient_based_segmentation,
    watershed_segmentation_with_markers
)
from learning_library import infer_segmentation
from skimage.morphology import disk, dilation, erosion, skeletonize
import os
import ast

# Paths to the image and ground truth
img_path = './images_IOSTAR/star01_OSC.jpg'
gt_path = './images_IOSTAR/GT_01.png'

# Check if files exist
if not os.path.exists(img_path) or not os.path.exists(gt_path):
    raise FileNotFoundError("Image or ground truth file not found.")

# Load image and ground truth
img = np.asarray(Image.open(img_path).convert('L')).astype(np.uint8)
img_GT = np.asarray(Image.open(gt_path).convert('L')).astype(bool)
img_GT = img_GT > 0  # Ensure binary

# Create circular mask
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows/2)**2)
img_mask[invalid_pixels] = False

# Define models with their optimal hyperparameters
models = [
    {
        'name': 'Morphological Gradient',
        'func': morphological_gradient_segmentation,
        'params': {'seuil': 30, 'selem_radius': 3}
    },
    {
        'name': 'External Gradient',
        'func': external_gradient_segmentation,
        'params': {'seuil': 30, 'selem_radius': 4}
    },
    {
        'name': 'Gradient Based with Closing',
        'func': gradient_based_segmentation,
        'params': {'seuil_grad': 10, 'seuil_closing': 30, 'selem_radius_grad': 4, 'selem_radius_closing': 1}
    },
    {
        'name': 'Watershed with Markers',
        'func': watershed_segmentation_with_markers,
        'params': {'min_distance': 10, 'seuil': 20, 'scale_min': 2, 'scale_max': 8}
    },
    {
        'name': 'Adaptive Threshold',
        'func': adaptive_treshold_segmentation,
        'params': {'adaptive_method': 'median', 'block_size': 17, 'C': 5, 'blur': True}
    },
    {
        'name': 'Adaptive Threshold with Opening',
        'func': adaptive_treshold_segmentation_with_opening,
        'params': {'adaptive_method': 'median', 'block_size': 17, 'C': 5, 'blur': False, 'opening_radius': 1}
    },
    {
        'name': 'U-Net',
        'func': infer_segmentation,
        'params': {'model_path': 'unet_retina.pth', 'seuil': 0.7, 'device': 'cpu'}
    }
]

# Store ROC data for final plot
fpr_list = {}
tpr_list = {}
auc_list = {}

# Process each model with visualization
for model in models:
    print(f"[INFO] Processing {model['name']}")
    
    if model['name'] == 'U-Net':
        # U-Net returns probabilities; assume modification is done
        probs = model['func'](img, img_mask, **model['params'])
        if probs.max() > 1:  # If output is 0-255, normalize
            probs = probs / 255.0
        # Convert to binary mask for skeleton
        img_out = (probs > model['params']['seuil']).astype(np.uint8) * 255
    else:
        # For traditional methods, generate scores or probabilities
        if model['name'] in ['Morphological Gradient', 'External Gradient', 'Gradient Based with Closing']:
            if model['name'] == 'Morphological Gradient':
                selem = disk(model['params']['selem_radius'])
                grad = dilation(img, selem) - erosion(img, selem)
                probs = grad / np.max(grad)
            elif model['name'] == 'External Gradient':
                selem = disk(model['params']['selem_radius'])
                grad_ext = dilation(img, selem) - img
                probs = grad_ext / np.max(grad_ext)
            elif model['name'] == 'Gradient Based with Closing':
                selem_grad = disk(model['params']['selem_radius_grad'])
                grad_ext = dilation(img, selem_grad) - img
                probs = grad_ext / np.max(grad_ext)
            img_out = model['func'](img, img_mask, **model['params'])
        else:
            img_out = model['func'](img, img_mask, **model['params'])
            probs = img_out / 255.0 if img_out.max() > 0 else img_out.astype(float)

    # Compute skeleton of the segmented mask
    img_out_skel = skeletonize(img_out > 0).astype(np.uint8) * 255
    GT_skel = skeletonize(img_GT).astype(np.uint8) * 255

    # Flatten arrays within the mask for ROC computation
    y_true = img_GT[img_mask].flatten()
    y_scores = probs[img_mask].flatten()

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Store ROC data
    fpr_list[model['name']] = fpr
    tpr_list[model['name']] = tpr
    auc_list[model['name']] = roc_auc

    # Plot segmentation results for inspection
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(img_out_skel, cmap='gray')
    plt.title(f'{model["name"]} Mask Skeleton')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img_GT, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(GT_skel, cmap='gray')
    plt.title('Ground Truth Skeleton')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    print(f"[INFO] Close the plot to proceed to the next model ({model['name']} done).")
    plt.waitforbuttonpress()  # Wait until the plot is closed
    plt.close()

# Final ROC plot
plt.figure(figsize=(10, 8))
for name in models:
    plt.plot(fpr_list[name['name']], tpr_list[name['name']], 
             label=f"{name['name']} (AUC = {auc_list[name['name']]:.2f})")

# Add diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random')

# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Segmentation Models')
plt.legend(loc="lower right")
plt.grid(True)

# Save the plot
plt.savefig('roc_curves.png')
print("[INFO] ROC curves saved as 'roc_curves.png'")
plt.show()