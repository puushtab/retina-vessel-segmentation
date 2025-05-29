
- Note != excellent résultat
- Explication de la démarche
- Montrer qu'on comprend ce qu'on fait

# TODO: 
Tester différentes méthodes, étudier différents résultats
Implémenter en profondeur les méthodes complexes
Choisir les meilleures à éexpliquer
Evaluer efficacité
Faire courbe ROC + autres métriques

 1. Tophat/Tophat conjugué:
    1. Tester opencv
    2. Pousser résultat, évaluer par métrique ?
 2. 

# Notes rapport
- Artefacct supprimable par opérations de **reconstruction** par ouverture, pas juste ouverture, il faut conserver la taille de nos vaisseaux.
- Protocole expérimental (sans e)
- Pourquoi pas de pour le seuil ? Grossier ? 
- Recherches par grille pour toutes les fonctions ?
- Rajouter détail sur la méthode de seuillage adaptatif sur la forme d u résultats recherchés (long tunnel connexe)

# Idées ChatGPT
### 🟢 **Idées simples (triviales mais utiles pour un premier jet ou une baseline)**

1. **Seuillage adaptatif simple** :

   * Appliquer un seuillage sur la luminance ou un canal contrasté (ex: green channel), éventuellement après un floutage.
   * Variante : seuillage local (adaptive thresholding) basé sur la moyenne ou la médiane locale.

2. **Filtrage top-hat conjugué** :

   * Supprimer l’arrière-plan pour mieux faire ressortir les vaisseaux : `image - ouverture(image, structurant circulaire)`.
   * Améliore le contraste local des structures fines.

3. **Gradient morphologique** :

   * Calculer le gradient morphologique pour mettre en évidence les contours des vaisseaux.

4. **Seuillage + ouverture morphologique** :

   * Pour éliminer le bruit (points isolés) après seuillage.

---

### 🟡 **Idées intermédiaires (intégrant des outils vus en cours)**

5. **Multiscale filtering** :

   * Appliquer des ouvertures/fermetures à différentes tailles d’éléments structurants, puis combiner les résultats.
   * Tu peux aussi calculer une moyenne pondérée des segmentations.

6. **Analyse granulométrique** : 

   * Étudier l’effet des ouvertures pour extraire les structures à certaines tailles → identifier automatiquement l’échelle caractéristique des vaisseaux.

7. **Ouverture par reconstruction (Reconstruction par géodésie)** :

   * Appliquer une ouverture morphologique suivie d’une reconstruction géodésique pour ne garder que les composantes connexes significatives.

8. **Utilisation des transformées en distance (d4 ou quasi-euclidienne)** :

   * Pour calculer des érosions rapides ou extraire les axes médiaux (centre des vaisseaux).

9. **Seuillage de la transformée en distance + squelette morphologique** :

   * Utiliser les maxima locaux de la distance pour localiser les vaisseaux fins.

10. **Squelettisation par amincissement** :

* Pour convertir une première détection en structure filaire, conservant la connectivité du réseau vasculaire.

11. **Post-traitement par ouverture par reconstruction** :

* Pour éliminer les artefacts sans casser les vaisseaux fins.

---

### 🔴 **Idées avancées (plus poussées, bonnes pour bonus + analyse critique)**

12. **Squelette multi-échelle par fonction de choc géodésique** :

* Calculer la transformée en distance + zones d’influence + fonction de choc, puis seuiller.
* Permet d’obtenir un réseau très fin et connecté.
* Adapté à la diversité d’échelles des vaisseaux.

13. **Segmentation guidée par Ligne de Partage des Eaux (LPE)** :

* Appliquer une LPE sur le gradient morphologique ou l’image préfiltrée.
* Avec filtrage des minima peu significatifs (bruit spatial ou faible dynamique).

14. **Utilisation de filtres alternés séquentiels (FAS)** :

* Combinaison d’ouvertures/fermetures pour améliorer la robustesse aux artefacts.

15. **Érosion ultime + squelette par points d’ancrage** :
   
* Pour détecter les points centraux des vaisseaux et reconstruire un squelette fin mais stable.

---

### 🔁 **Stratégies hybrides (pipeline combiné)**

16. **Pipeline typique efficace** :

```text
Prétraitement → Gradient morphologique → LPE ou seuillage → Reconstruction géodésique → Squelette ou filtrage
```

17. **Masquage itératif** :

* Segmentation initiale grossière → masquage → affinement local via une méthode différente.

18. **Utilisation d’une zone d’intérêt (ROI)** :

* Si l’image est bruitée sur les bords, réduire l’analyse à une ROI basée sur la vérité terrain.

---

Si tu veux, je peux t’aider à coder une des approches ou t’écrire une partie de rapport critique pour expliquer les forces/faiblesses de tes choix. Tu veux commencer par quoi ?

# Idées prof
## Normalisation de la métrique
Comparaison du nombres de pixels entre la personne segmentée en premier plan et la personne segmentée en dernier plan donne importance différente si on compare que la différence des pixels entre la segmentation de référence et la segmentation calculée ! Par conséquent, il faut comparer les squelettes des deux images !
Squelette et transformée en distance définis que pour des images binaires.
Objectif: 
- Transformer image en binaire
- Faire squelette
OU:
- Segmenter l'image directement