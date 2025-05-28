import numpy as np
from skimage.morphology import skeletonize


def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT)  # Reduce evaluation to skeleton pixels
    img_out_skel = skeletonize(img_out)  # Skeletonize segmented image
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


##  Example usage
# ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
# print('Accuracy =', ACCU, ', Recall =', RECALL)