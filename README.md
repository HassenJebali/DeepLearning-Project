# Deep Learnig Projects
# 🧠 AI Deep Learning Practice - Tâche 1 : Régression avec Keras

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

---
---

## 📂 Tâche 2 : Classification supervisée

### 📌 Description
Utilisation du dataset **MNIST** pour entraîner un modèle de classification d’images à base de convolution. L’objectif est de reconnaître automatiquement les chiffres manuscrits à partir d’images en niveaux de gris (28x28 pixels).

---

## 🎯 Objectif
Cette tâche consiste à construire un **réseau de neurones convolutif (CNN)** avec **PyTorch** afin de classer des images de chiffres manuscrits du dataset **MNIST** en 10 classes (de 0 à 9).

---

## 🛠️ Outils & Technologies

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn

---

## 📊 Dataset

- **MNIST** : Dataset intégré à `torchvision.datasets.MNIST`
  - 60 000 images d’entraînement
  - 10 000 images de test
  - 10 classes (chiffres de 0 à 9)
  - Taille : 28x28 pixels, monochrome

---

## ⚙️ Étapes du projet

### 1. Chargement des données
- Import du dataset via `torchvision`
- Normalisation des images

### 2. Prétraitement
- Création des `DataLoader` pour l’entraînement et le test
- Batch size = 64 pour le train, 1000 pour le test

### 3. Modélisation avec PyTorch

Dans cette étape, on construit un **réseau de neurones convolutifs (CNN)** à l’aide de la bibliothèque `torch.nn`. Ce type d’architecture est particulièrement efficace pour les données image, car il capture les **caractéristiques spatiales** (bords, textures, formes...) à travers des couches de convolution.

#### 🧱 Architecture du modèle

Le modèle utilisé pour MNIST se compose de :

- **Couche 1 : Convolution 2D**
  - `nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)`
  - Reçoit une image 28×28 en niveaux de gris (1 canal)
  - Produit 16 cartes de caractéristiques (feature maps) de même taille grâce au padding
  - **Activation : ReLU**
  - **Réduction spatiale : MaxPooling (2x2)** → taille devient 14×14

- **Couche 2 : Convolution 2D**
  - `nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)`
  - Prend en entrée les 16 feature maps de la couche précédente
  - Produit 32 cartes → taille encore 14×14 → MaxPooling réduit à 7×7
  - **Activation : ReLU**
  - **MaxPooling (2x2)**

- **Flatten**
  - Aplatissement de la sortie des convolutions : `32 * 7 * 7 = 1568` neurones

- **Couche Fully Connected (dense)**
  - `nn.Linear(in_features=1568, out_features=128)`
  - **Activation : ReLU**

- **Sortie (Classification)**
  - `nn.Linear(in_features=128, out_features=10)`
  - Donne un vecteur de 10 scores (logits), un pour chaque chiffre (0 à 9)

#### 🧠 Fonctions clés

- **Activation : ReLU (Rectified Linear Unit)**
  - Permet au réseau d’apprendre des relations non linéaires
  - Aide à éviter le problème du "vanishing gradient"

- **Fonction de perte : `nn.CrossEntropyLoss`**
  - Combine `LogSoftmax` + `Negative Log Likelihood`
  - Adaptée à la **classification multi-classes**
  - Calcule la différence entre les vraies classes et les probabilités prédites

- **Optimiseur : `torch.optim.Adam`**
  - Méthode d’optimisation adaptative
  - Ajuste dynamiquement le learning rate de chaque paramètre
  - Efficace, stable, et largement utilisé pour l'entraînement de modèles profonds

#### 🧮 Dimensions résumées à chaque étape (entrée = 28×28)
| Étape | Taille sortie | Canaux |
|------|---------------|--------|
| Conv1 + Pooling | 14×14     | 16     |
| Conv2 + Pooling | 7×7       | 32     |
| Flatten         | 1568      | —      |
| FC1             | 128       | —      |
| FC2 (Sortie)    | 10        | —      |

---

📌 Remarque : Le modèle reste volontairement **simple et léger**, adapté à un petit dataset comme MNIST pour un entraînement rapide tout en illustrant les bases de la **convolution**, **pooling**, **flatten**, et **dense layers**.


### 4. Entraînement
- Boucle d’entraînement sur plusieurs epochs
- Affichage de la perte par epoch

### 5. Évaluation
- Prédictions sur le jeu de test
- Calcul de l’accuracy globale
- Génération de la matrice de confusion

---


## 📊 Métriques utilisées

### 🔹 Accuracy
- **Définition** : Proportion de prédictions correctes sur l’ensemble des échantillons.
- **Objectif** : Évaluer la précision globale du modèle de classification.
- **Utilité** : Très adaptée aux datasets équilibrés comme MNIST.

### 🔹 Matrice de confusion
- **Définition** : Tableau croisé indiquant les bonnes et mauvaises prédictions pour chaque classe.
- **Objectif** : Identifier quelles classes sont confondues entre elles.
- **Utilité** : Utile pour analyser les erreurs spécifiques (ex. : 4 prédits comme 9).

---

✅ **But commun de ces métriques** : Évaluer la capacité du modèle à **reconnaître correctement les images** et à **réduire les confusions entre classes**. Une haute accuracy et une matrice de confusion bien diagonale indiquent un bon modèle.

---

## ℹ️ Fonction d’activation utilisée : ReLU

- **Définition** : `ReLU(x) = max(0, x)`
- **Objectif** : Ajouter de la non-linéarité au réseau pour lui permettre d’apprendre des fonctions complexes.
- **Avantage** : Simple, rapide, et efficace pour les réseaux profonds. Elle évite le problème du gradient qui disparaît.

---
## 🧩 Visualisation du modèle avec `torchviz`

Afin de mieux comprendre la structure du modèle CNN construit avec PyTorch, il est possible de **générer un graphe visuel** du flux computationnel (calcul des sorties à partir des entrées). Cela aide à observer les **connexions entre les couches** et la façon dont les données traversent le réseau.

### ✅ Étapes pour afficher le graphe dans Google Colab

### 1. Installer la bibliothèque `torchviz`
```python
!pip install torchviz

---

## 📂 Tâche 3 : Classification multi-classes

### 📌 Description
Utilisation du dataset **Fashion-MNIST** pour entraîner un modèle de classification d’images avec un **CNN construit avec TensorFlow/Keras**. L’objectif est de reconnaître automatiquement des objets ou chiffres manuscrits à partir d’images.

---

## 🎯 Objectif
Concevoir un **réseau convolutif robuste avec régularisation et data augmentation** capable de distinguer plusieurs classes (0 à 9), en utilisant **Keras** (backend TensorFlow).

---

## 🛠️ Outils & Technologies

- Python 3.x
- TensorFlow / Keras
- Matplotlib / Seaborn
- Scikit-learn

---

## 📊 Dataset

- **Fashion MNIST** (vêtements)
  - 60 000 images d'entraînement, 10 000 de test
  - Images 28x28, en niveaux de gris
  - 10 classes
  - Accessible via `tensorflow.keras.datasets`

---

## ⚙️ Étapes du projet

### 1. Prétraitement des données
- Chargement via `keras.datasets.fashion_mnist`
- Normalisation des pixels `[0, 255] → [0, 1]`
- Reshape des images au format (28, 28, 1)
- Encodage one-hot des labels (`to_categorical`)

### 2. Construction du modèle CNN
- `Conv2D(32)` + ReLU + MaxPooling + Dropout
- `Conv2D(64)` + ReLU + MaxPooling + Dropout
- `Flatten` → `Dense(128)` + ReLU + Dropout
- `Dense(10)` avec activation `softmax` (sortie)

### 3. Compilation du modèle
- Optimiseur : `Adam`
- Fonction de perte : `categorical_crossentropy`
- Métrique : `accuracy`

### 4. Entraînement du modèle
Deux variantes sont testées :
- **Sans Data Augmentation**
- **Avec Data Augmentation** (`ImageDataGenerator` : rotation, shift, zoom...)

### 5. Évaluation
- `evaluate()` sur le jeu de test
- Matrice de confusion
- Rapport de classification (`precision`, `recall`, `f1-score`)
- Courbes de précision par epoch

---

## 📈 Résultats comparés

| Variante               | Accuracy (test) | F1-score moyen | Classe 6 (F1) |
|------------------------|----------------|----------------|---------------|
| ❌ Sans augmentation   | 89 %           | 0.89           | 0.69          |
| ✅ Avec augmentation   | 89 %           | 0.89           | 0.69          |

➡️ **Observation** : Les résultats sont très proches, mais l’usage de la data augmentation permet une meilleure **généralisation**, en particulier pour les classes sous-représentées ou confondues (comme la classe 6).

---

## 📊 Rapport de classification (extraits)

**Sans Data Augmentation** :
accuracy: 0.89
macro avg F1: 0.89
classe 6 F1: 0.69

**Avec Data Augmentation** :
accuracy: 0.89
macro avg F1: 0.89
classe 6 F1: 0.69

## 🎁 Data Augmentation

Ajout de transformations artificielles via `ImageDataGenerator` pour augmenter la robustesse du modèle :

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train)

---

## Fonctions utilisées
`ReLU` : activation non linéaire après chaque convolution

`Softmax` : activation de sortie pour la classification multi-classes

`Dropout` : couche de régularisation pour éviter le surapprentissage

`Data Augmentation` : technique pour améliorer la généralisation du modèle en simulant des variations réalistes d’images

