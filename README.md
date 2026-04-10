---
title: Walmart Sales Prediction
emoji: 📊
colorFrom: blue
colorTo: yellow
sdk: docker
sdk_version: "1.28.0"
app_file: hf_app.py
pinned: false
---

# 🎬 Walmart Sales Prediction Dashboard

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20App-yellow)](https://huggingface.co/spaces/msshaqura/walmart_project)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

👉 **Try the app here:**  
[https://huggingface.co/spaces/msshaqura/walmart_project](https://huggingface.co/spaces/msshaqura/walmart_project)

---

## 📊 Table des matières

1. [🎯 Objectif du projet](#-objectif-du-projet)
2. [📁 Structure du projet](#-structure-du-projet)
3. [📊 Dataset](#-dataset)
4. [🔧 Prérequis & Installation](#-prérequis--installation)
5. [🚀 Lancer l'application](#-lancer-lapplication)
6. [🧠 Modèles entraînés](#-modèles-entraînés)
7. [📈 Performances](#-performances)
8. [🔍 Découverte clé : Paradoxe de Simpson](#-découverte-clé--paradoxe-de-simpson)
9. [🛠️ Stack technique](#️-stack-technique)
10. [📦 Livrables](#-livrables)
11. [👤 Auteur](#-auteur)

---

## 🎯 Objectif du projet

**Prédire les ventes hebdomadaires** des magasins Walmart à partir d'indicateurs économiques (CPI, chômage, prix du carburant, température, etc.).

**Objectifs pédagogiques :**
- ✅ Réaliser une analyse exploratoire complète (EDA)
- ✅ Prétraiter les données (valeurs manquantes, outliers, features temporelles)
- ✅ Entraîner et comparer plusieurs modèles de régression
- ✅ Déployer l'application sur Hugging Face Spaces

---

## 📁 Structure du projet
walmart-sales-prediction/
├── app/
│ └── streamlit_app.py # Application Streamlit
├── data/
│ └── walmart_store_sales.csv # Dataset (150 lignes, 8 colonnes)
├── models/
│ ├── random_forest.pkl # Meilleur modèle (R²=0.745)
│ ├── linear_regression.pkl
│ ├── ridge.pkl
│ ├── lasso.pkl
│ └── features_names.pkl # Noms des features
├── src/
│ ├── data_preprocessing.py # Nettoyage et imputation
│ ├── train_models.py # Entraînement des modèles
│ └── save_models.py # Sauvegarde des modèles
├── notebooks/
│ └── 01_analysis_complete.ipynb # Analyse complète
├── Dockerfile # Conteneurisation
├── requirements.txt # Dépendances Python
├── README.md # Documentation
└── .gitattributes # Configuration Xet pour fichiers .pkl


---

## 📊 Dataset

**Source :** Données Walmart (adaptées du challenge Kaggle)

**Colonnes :**
| Colonne | Type | Description |
|---------|------|-------------|
| `Store` | Numeric | Identifiant du magasin (1-45) |
| `Date` | Date | Semaine d'enregistrement |
| `Weekly_Sales` | Numeric | **Cible** - Ventes de la semaine ($) |
| `Holiday_Flag` | Binaire | 1 = semaine avec jour férié |
| `Temperature` | Numeric | Température (°F) |
| `Fuel_Price` | Numeric | Prix du carburant ($) |
| `CPI` | Numeric | Indice des prix à la consommation |
| `Unemployment` | Numeric | Taux de chômage (%) |

**Taille :** 150 lignes → 131 lignes après nettoyage

---

## 🔧 Prérequis & Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/msshaqura/projet_walmart.git
cd projet_walmart

2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Installer les dépendances
pip install -r requirements.txt

🚀 Lancer l'application
En local
streamlit run app/streamlit_app.py

En ligne
👉 Hugging Face Space : Lien ci-haut

🧠 Modèles entraînés
Modèle	            Type	            Régularisation
Linear Regression	Baseline	        Aucune
Ridge	            L2	                alpha=1.0
Lasso	            L1	                alpha=1.0
Random Forest	    Non linéaire	    n_estimators=100, max_depth=10

📈 Performances
Tableau comparatif
Modèle	                R² Train        R² Test	    RMSE Test	    MAE Test
Linear Regression	    0.1337	        0.1520	    588 490 $	    541 390 $
Ridge	                0.1334	        0.1532	    588 069 $	    542 354 $
Lasso	                0.1337	        0.1520	    588 492 $	    541 393 $
Random Forest	        0.9399	        0.7448	    322 809 $	    247 209 $


🔍 Découverte clé : Paradoxe de Simpson
Le problème
Analyse	                Corrélation Unemployment vs Weekly_Sales
Globale	                POSITIVE (+0.241) ❌
Par magasin (Store 4)	NÉGATIVE (-0.508) ✅
Par magasin (Store 5)	NÉGATIVE (-0.549) ✅

Pourquoi c'est important ?
- Un modèle linéaire global aurait été totalement erroné

- Cela justifie le choix d'un modèle non linéaire (Random Forest)

- Après contrôle par Store, le coefficient redevient négatif (-21 998 $)

![Image 1](images/image-1.png)

![Image 2](images/image.png)


🛠️ Stack technique
Catégorie	        Technologies
Langage	            Python 3.9+
Data	            Pandas, NumPy
Visualisation	    Matplotlib, Seaborn
Machine Learning	Scikit-learn (LinearRegression, Ridge, Lasso, RandomForest)
Interface	        Streamlit
Déploiement	        Docker, Hugging Face Spaces
Versioning	        Git, GitHub


👤 Auteur
Mohammed SHAQURA
🎓 Data Analyst - Jedha Bootcamp