import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from segmentation_library import adaptive_treshold_segmentation
from skimage.morphology import disk, closing, opening, reconstruction
import os

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

def apply_pipeline(img, img_mask):
    """Apply the exact pipeline from pipeline.py."""
    # Step 1: Adaptive Threshold
    adapt_tresh = adaptive_treshold_segmentation(
        img, 
        img_mask,
        adaptive_method='median',
        block_size=17,
        C=3,
        blur=True
    )
    
    # Step 2: First reconstruction
    img_closed = closing(adapt_tresh, disk(1))
    img_open = opening(img_closed, disk(3))
    rec1 = reconstruction(img_open, img_closed).astype(int)
    
    # Step 3: Second reconstruction
    img_closed = closing(rec1, disk(2))
    img_open = opening(img_closed, disk(2))
    rec2 = reconstruction(img_open, img_closed).astype(int)
    
    return rec2

# Apply pipeline and get result
result = apply_pipeline(img, img_mask)

# Normalize scores to [0,1] range
scores = result.astype(float) / 255.0

# Flatten arrays within the mask for ROC computation
y_true = img_GT[img_mask].flatten()
y_scores = scores[img_mask].flatten()

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Pipeline ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Pipeline Segmentation')
plt.legend(loc="lower right")
plt.grid(True)

# Save the plot
plt.savefig('pipeline_roc_curve.png')
print("[INFO] ROC curve saved as 'pipeline_roc_curve.png'")
plt.show()

# Print AUC score
print(f"Area Under Curve (AUC): {roc_auc:.3f}") 