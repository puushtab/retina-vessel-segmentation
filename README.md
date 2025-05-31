# Retina Vessel Segmentation

This project implements various methods for segmenting blood vessels in retinal images using traditional image processing techniques and deep learning approaches.

## Project Structure

```
retina-vessel-segmentation/
├── images_IOSTAR/           # Dataset directory
├── media/                   # Output directory for results
├── segmentation_library.py  # Implementation of segmentation methods
├── validation_tuning_library.py  # Validation and hyperparameter tuning
├── learning_library.py      # Deep learning implementation (U-Net)
├── script_tp2.py           # Main script for single method evaluation
├── pipeline.py             # Implementation of the morphological pipeline
├── pipeline_roc.py         # ROC curve computation for the pipeline
├── ROC_script.py           # ROC curves for all methods
├── learning_script.py      # Deep learning training and inference
└── requirements.txt        # Project dependencies
```

## Features

### Traditional Image Processing Methods
- Adaptive Thresholding
- Morphological Operations (Opening, Closing, Reconstruction)
- Top-hat Transform
- Gradient-based Segmentation
- Watershed Segmentation
- Multi-scale Skeletonization

### Deep Learning
- U-Net implementation for vessel segmentation
- Pre-trained model available (`unet_retina.pth`)

### Evaluation Metrics
- Precision (Accuracy)
- Recall
- F1-score
- ROC curves and AUC scores

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd retina-vessel-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Method Evaluation
```bash
python script_tp2.py
```
This script allows you to evaluate a single segmentation method with hyperparameter tuning.

### Pipeline Evaluation
```bash
python pipeline.py
```
Runs the complete morphological pipeline for vessel segmentation.

### ROC Curves
```bash
python ROC_script.py
```
Generates ROC curves for all implemented methods.

### Deep Learning
```bash
python learning_script.py
```
Trains or uses the U-Net model for vessel segmentation.

## Pipeline Method

The main pipeline consists of:
1. Adaptive thresholding with median method
2. First reconstruction with morphological operations
3. Second reconstruction with different parameters

Parameters:
- Adaptive threshold: block_size=17, C=3
- First reconstruction: closing(disk(1)), opening(disk(3))
- Second reconstruction: closing(disk(2)), opening(disk(2))

## Results

Results are saved in:
- `media/results/` - Segmentation outputs
- `pipeline_roc_curve.png` - ROC curve for the pipeline
- `roc_curves.png` - ROC curves for all methods
- `results_summary.csv` - Performance metrics

## Dependencies

- numpy
- scikit-image
- OpenCV
- scikit-learn
- PyTorch
- matplotlib
- PIL

## License

[Your License Here]

## Authors
Amine Maazizi
Gabriel Dupuis
