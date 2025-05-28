import numpy as np
from skimage.morphology import skeletonize
import cv2 as  cv


def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT)  # Reduce evaluation to skeleton pixels
    img_out_skel = skeletonize(img_out)  # Skeletonize segmented image

    # cv.imwrite('media/GT_skel.png', (GT_skel * 255).astype(np.uint8))
    
    TP = np.sum(img_out_skel & img_GT)  # True positives
    FP = np.sum(img_out_skel & ~img_GT)  # False positives
    FN = np.sum(GT_skel & ~img_out)  # False negatives

    ACCU = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    return ACCU, RECALL, img_out_skel, GT_skel

from itertools import product

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def tune_hyperparameters(segmentation_func, param_grid, img, img_mask, img_GT, verbose=False):
    """
    Recherche les meilleurs hyperparamètres pour une fonction de segmentation.

    Paramètres :
    ------------
    segmentation_func : function
        Fonction de segmentation prenant (img, img_mask, **params).
    param_grid : dict
        Dictionnaire des plages de valeurs pour les paramètres : {nom_paramètre: liste_valeurs}.
    img : ndarray
        Image en niveaux de gris.
    img_mask : ndarray
        Masque binaire de la région d’intérêt.
    img_GT : ndarray
        Vérité terrain (image binaire).
    verbose : bool
        Si True, affiche les scores à chaque étape.

    Retourne :
    ----------
    best_params : dict
        Combinaison optimale des hyperparamètres.
    best_score : float
        Meilleur score F1 obtenu.
    history : list of dict
        Chaque élément contient : {'params': ..., 'accuracy': ..., 'recall': ..., 'f1': ...}
    """
    from itertools import product

    def f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -1
    best_params = None
    history = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        img_out = segmentation_func(img, img_mask, **params)
        accu, recall, *_ = evaluate(img_out, img_GT)
        f1 = f1_score(accu, recall)

        history.append({'params': params.copy(), 'accuracy': accu, 'recall': recall, 'f1': f1})

        if verbose:
            print(f"Paramètres testés : {params} → Accuracy={accu:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_params = params.copy()

    return best_params, best_score, history

import inspect
import os
import csv
from skimage.io import imsave

def save_results(
    segmentation_func,
    results_dir: str,
    img_path: str,
    img_out: np.ndarray,
    img_out_skel: np.ndarray,
    precision: float,
    recall: float,
    f1_score: float,
    best_params: dict
):
    """
    Save segmentation and skeleton images, and append evaluation results to a single CSV file.

    Parameters
    ----------
    segmentation_func : function
        Segmentation function used (name extracted automatically).
    results_dir : str
        Directory to save images.
    img_path : str
        Path to original image.
    img_out : np.ndarray
        Segmented binary image.
    img_out_skel : np.ndarray
        Skeletonized segmented image.
    precision : float
        Precision metric.
    recall : float
        Recall metric.
    f1_score : float
        F1 score metric.
    best_params : dict
        Best hyperparameters used.

    Returns
    -------
    None
    """

    method_name = segmentation_func.__name__

    os.makedirs(results_dir, exist_ok=True)

    # Build a descriptive base filename from method name and params
    base_filename = method_name + ''.join(f"_{k}{v}" for k, v in best_params.items())

    seg_path = os.path.join(results_dir, base_filename + '_segmentation.png')
    skel_path = os.path.join(results_dir, base_filename + '_skeleton.png')

    # Save images as uint8 (0-255)
    imsave(seg_path, (img_out * 255).astype('uint8'))
    imsave(skel_path, (img_out_skel * 255).astype('uint8'))

    csv_path = os.path.join(os.getcwd(), 'results_summary.csv')  # single CSV in current working dir

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['method', 'image_path', 'segmentation_path', 'skeleton_path',
                      'precision', 'recall', 'f1_score', 'parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'method': method_name,
            'image_path': img_path,
            'segmentation_path': seg_path,
            'skeleton_path': skel_path,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'parameters': str(best_params)
        })

    print(f"[INFO] Results saved to {csv_path} and images saved to {results_dir}")

##  Example usage
# ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
# print('Accuracy =', ACCU, ', Recall =', RECALL)