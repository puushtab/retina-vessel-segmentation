import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage.io import imsave
import math
from skimage import data, filters
from matplotlib import pyplot as plt
import os

# Import your custom libraries
from segmentation_library import *
from validation_tuning_library import evaluate, tune_hyperparameters, save_results
from learning_library import train_model, infer_segmentation

# Load the original grayscale image
img_path = './images_IOSTAR/star01_OSC.jpg'
gt_path = './images_IOSTAR/GT_01.png'

if not os.path.exists(img_path):
    print(f"[ERROR] Image file not found: {img_path}")
    exit(1)
if not os.path.exists(gt_path):
    print(f"[ERROR] Ground truth file not found: {gt_path}")
    exit(1)

img = np.asarray(Image.open(img_path).convert('L')).astype(np.uint8)
print(f"[INFO] Loaded image shape: {img.shape}")

# Create a circular mask (FOV mask)
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows/2)**2)
img_mask[invalid_pixels] = False
print(f"[INFO] Created circular mask with {np.sum(img_mask)} valid pixels")

# Load the ground truth image as binary
img_GT = np.asarray(Image.open(gt_path).convert('L')).astype(bool)
img_GT = img_GT > 0  # Ensure binary
print(f"[INFO] Ground truth loaded. Vessel pixels: {np.sum(img_GT)}")

# Check if model exists, if not train it
model_path = 'unet_retina.pth'
if not os.path.exists(model_path):
    print(f"[INFO] Model file '{model_path}' not found. Training new model...")
    train_model(epochs=30, batch_size=2, save_path=model_path)
else:
    print(f"[INFO] Using existing model: {model_path}")

# Run inference with the trained U-Net
print("\n" + "="*50)
print("RUNNING INFERENCE")
print("="*50)

# Try different thresholds to find optimal one
best_f1 = 0
best_threshold = 0.5
best_img_out = None

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"\n[INFO] Testing threshold: {threshold}")
    img_out = infer_segmentation(img, img_mask, model_path=model_path, seuil=threshold)
    
    if np.sum(img_out) == 0:
        print(f"[WARNING] No vessels detected with threshold {threshold}")
        continue
    
    # Quick F1 evaluation
    img_out_binary = img_out > 0
    tp = np.sum(img_out_binary & img_GT)
    fp = np.sum(img_out_binary & ~img_GT)
    fn = np.sum(~img_out_binary & img_GT)
    
    if tp > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"[INFO] Threshold {threshold}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_img_out = img_out.copy()

if best_img_out is None:
    print("[ERROR] No valid segmentation found with any threshold!")
    # Use threshold 0.1 as fallback
    best_img_out = infer_segmentation(img, img_mask, model_path=model_path, seuil=0.1)
    best_threshold = 0.1

print(f"\n[INFO] Best threshold: {best_threshold} with F1 score: {best_f1:.3f}")

# Final evaluation with the best result
try:
    ACCU, RECALL, img_out_skel, GT_skel = evaluate(best_img_out, img_GT)
    print(f'[RESULTS] Accuracy = {ACCU:.4f}, Recall = {RECALL:.4f}')
except Exception as e:
    print(f"[ERROR] Evaluation failed: {e}")
    # Create dummy skeleton images if evaluation fails
    from skimage.morphology import skeletonize
    img_out_skel = skeletonize(best_img_out > 0).astype(np.uint8) * 255
    GT_skel = skeletonize(img_GT).astype(np.uint8) * 255
    ACCU, RECALL = 0, 0

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(232)
plt.imshow(best_img_out, cmap='gray')
# SAVE the best segmentation image
best_img_out_path = 'unet_best_segmentation.png'
imsave(best_img_out_path, img_as_ubyte(best_img_out), cmap='gray')

plt.title(f'U-Net Segmentation (threshold={best_threshold})')
plt.axis('off')

plt.subplot(233)
plt.imshow(img_out_skel, cmap='gray')
plt.title('Segmentation Skeleton')
plt.axis('off')

plt.subplot(234)
plt.imshow(img_mask, cmap='gray')
plt.title('FOV Mask')
plt.axis('off')

plt.subplot(235)
plt.imshow(img_GT, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(236)
plt.imshow(GT_skel, cmap='gray')
plt.title('Ground Truth Skeleton')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed results
try:
    # Create a dummy segmentation function for save_results
    def unet_segmentation(img, img_mask, **kwargs):
        return infer_segmentation(img, img_mask, model_path=model_path, **kwargs)
    
    save_results(
        segmentation_func=unet_segmentation,
        results_dir='media/results',
        img_path=img_path,
        img_out=best_img_out,
        img_out_skel=img_out_skel,
        precision=ACCU,
        recall=RECALL,
        f1_score=best_f1,
        best_params={'threshold': best_threshold}
    )
    print("[INFO] Results saved successfully")
except Exception as e:
    print(f"[WARNING] Could not save results: {e}")

print("\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print(f"Best threshold: {best_threshold}")
print(f"F1 Score: {best_f1:.4f}")
print(f"Accuracy: {ACCU:.4f}")
print(f"Recall: {RECALL:.4f}")
print(f"Detected vessel pixels: {np.sum(best_img_out > 0)}")
print(f"Ground truth vessel pixels: {np.sum(img_GT)}")