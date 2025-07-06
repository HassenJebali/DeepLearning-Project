# Deep Learnig Projects
# 🧠 AI Deep Learning Practice Suite - Tâche 1 : Régression avec Keras

## 🎯 Objectif
Cette tâche vise à construire un modèle de régression utilisant un **MLP (Multilayer Perceptron)** avec **Keras (TensorFlow backend)** afin de prédire une variable continue, comme le prix des maisons.

---

## 📂 Tâche 1 : Régression supervisée

### 📌 Description
Utilisation du dataset **California Housing** pour entraîner un réseau de neurones simple à estimer une valeur continue à partir de données tabulaires.

---

## 🛠️ Outils & Technologies

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn

---

## 📊 Dataset

- **California Housing** : Dataset accessible via `sklearn.datasets.fetch_california_housing()`

---

## ⚙️ Étapes du projet

### 1. Chargement et exploration des données
- Import du dataset
- Séparation features / target

### 2. Prétraitement
- Division en jeu d'entraînement/test (80/20)
- Normalisation des données avec `StandardScaler`

### 3. Modélisation avec Keras
- Modèle MLP : 
  - Dense(64, activation='relu')
  - Dense(32, activation='relu')
  - Dense(1) (sortie)
- Optimiseur : `adam`
- Fonction de coût : `mse`
- Métriques : `mae`

### 4. Entraînement
- Validation split (20%)
- Epochs : 100
- Batch size : 32

### 5. Évaluation
- Métriques calculées :
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Visualisation de la courbe d'apprentissage

---

✅ **But commun de ces métriques** : Évaluer la capacité du modèle à **faire des prédictions proches des valeurs réelles**. Plus les scores sont bas, meilleur est le modèle.


---

## 📈 Résultats attendus

- Affichage des scores MAE, MSE, RMSE sur le jeu de test
- Graphe de la perte d'entraînement/validation au fil des epochs

---

## 📊 Métriques utilisées

### 🔹 MSE (Mean Squared Error)
- **Définition** : Moyenne des carrés des écarts entre les prédictions du modèle et les vraies valeurs.
- **Objectif** : Pénaliser fortement les erreurs importantes (car l’erreur est élevée au carré).
- **Utilité** : Utile pour détecter les modèles qui ont tendance à faire de grosses erreurs sur certains échantillons.

### 🔹 RMSE (Root Mean Squared Error)
- **Définition** : Racine carrée du MSE.
- **Objectif** : Fournir une mesure d’erreur dans la même unité que la variable prédite (par exemple, en milliers de dollars pour le prix des maisons).
- **Utilité** : Interprétable directement sur l’échelle de la cible. Plus facile à comparer à une valeur réelle moyenne.

### 🔹 MAE (Mean Absolute Error)
- **Définition** : Moyenne des valeurs absolues des écarts entre les prédictions et les vraies valeurs.
- **Objectif** : Mesurer l’erreur moyenne sans donner trop d’importance aux erreurs extrêmes.
- **Utilité** : Plus robuste que le MSE en présence d’outliers (valeurs aberrantes).

---

✅ **But commun de ces métriques** : Évaluer la capacité du modèle à **faire des prédictions proches des valeurs réelles**. Plus les scores sont bas, meilleur est le modèle.


