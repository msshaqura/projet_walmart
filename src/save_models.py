# ------------------------------
# FICHIER : src/save_models.py
# ------------------------------

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# DÉTERMINER LE CHEMIN CORRECT
# ============================================================
# Obtenir le chemin du dossier src
current_dir = os.path.dirname(os.path.abspath(__file__))
# Remonter à la racine du projet
project_root = os.path.dirname(current_dir)
# Chemin vers le fichier data
data_path = os.path.join(project_root, 'data', 'walmart_store_sales.csv')

print("=" * 70)
print("CHARGEMENT ET NETTOYAGE DES DONNÉES")
print("=" * 70)
print(f"Chemin du fichier: {data_path}")
print(f"Fichier existe: {os.path.exists(data_path)}")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
df = pd.read_csv(data_path)

print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Supprimer les lignes avec Weekly_Sales manquant
df = df.dropna(subset=['Weekly_Sales'])

# Convertir Date
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date'])

# Interpoler les dates
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

def interpoler_dates(groupe):
    timestamps = groupe['Date'].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    timestamps_interp = timestamps.interpolate(method='linear', limit_direction='both')
    groupe['Date'] = pd.to_datetime(timestamps_interp, unit='s')
    return groupe

df = df.groupby('Store', group_keys=False).apply(interpoler_dates)

# Créer les features temporelles
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Supprimer Date
df = df.drop(columns=['Date'])

# Supprimer les outliers dans Unemployment
Q1 = df['Unemployment'].quantile(0.25)
Q3 = df['Unemployment'].quantile(0.75)
IQR = Q3 - Q1
borne_sup = Q3 + 1.5 * IQR
df = df[(df['Unemployment'] <= borne_sup) | (df['Unemployment'].isna())]

# Imputer les valeurs manquantes
df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0)
df['Temperature'] = df.groupby('Store')['Temperature'].ffill()
df['Temperature'] = df.groupby('Store')['Temperature'].bfill()
df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].ffill()
df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].bfill()
df['CPI'] = df.groupby('Store')['CPI'].transform(lambda x: x.interpolate(method='linear'))
df['CPI'] = df.groupby('Store')['CPI'].ffill()
df['CPI'] = df.groupby('Store')['CPI'].bfill()
df['Unemployment'] = df.groupby('Store')['Unemployment'].transform(lambda x: x.interpolate(method='linear'))
df['Unemployment'] = df.groupby('Store')['Unemployment'].ffill()
df['Unemployment'] = df.groupby('Store')['Unemployment'].bfill()

print(f"✅ Données nettoyées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================
# PRÉPARATION POUR L'ENTRAÎNEMENT
# ============================================================
print("\n" + "=" * 70)
print("PRÉPARATION POUR L'ENTRAÎNEMENT")
print("=" * 70)

X = df.drop(columns=['Weekly_Sales'])
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set: {X_train.shape[0]} lignes")
print(f"Test set: {X_test.shape[0]} lignes")
print(f"Features: {X.columns.tolist()}")

# ============================================================
# ENTRAÎNEMENT DES MODÈLES
# ============================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 70)

# 1. Linear Regression
print("\n1. Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"   ✅ R² Train: {lr.score(X_train, y_train):.4f}")
print(f"   ✅ R² Test: {lr.score(X_test, y_test):.4f}")

# 2. Ridge
print("\n2. Ridge (alpha=1.0)...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
print(f"   ✅ R² Train: {ridge.score(X_train, y_train):.4f}")
print(f"   ✅ R² Test: {ridge.score(X_test, y_test):.4f}")

# 3. Lasso
print("\n3. Lasso (alpha=1.0)...")
lasso = Lasso(alpha=1.0, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
print(f"   ✅ R² Train: {lasso.score(X_train, y_train):.4f}")
print(f"   ✅ R² Test: {lasso.score(X_test, y_test):.4f}")

# 4. Random Forest (meilleur modèle)
print("\n4. Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"   ✅ R² Train: {rf.score(X_train, y_train):.4f}")
print(f"   ✅ R² Test: {rf.score(X_test, y_test):.4f}")

# ============================================================
# SAUVEGARDE DES MODÈLES
# ============================================================
print("\n" + "=" * 70)
print("SAUVEGARDE DES MODÈLES")
print("=" * 70)

# Créer le dossier models
models_path = os.path.join(project_root, 'models')
os.makedirs(models_path, exist_ok=True)

# Sauvegarder chaque modèle
joblib.dump(lr, os.path.join(models_path, 'linear_regression.pkl'))
print("✅ linear_regression.pkl sauvegardé")

joblib.dump(ridge, os.path.join(models_path, 'ridge.pkl'))
print("✅ ridge.pkl sauvegardé")

joblib.dump(lasso, os.path.join(models_path, 'lasso.pkl'))
print("✅ lasso.pkl sauvegardé")

joblib.dump(rf, os.path.join(models_path, 'random_forest.pkl'))
print("✅ random_forest.pkl sauvegardé")

# Sauvegarder aussi les features names
features_names = X.columns.tolist()
joblib.dump(features_names, os.path.join(models_path, 'features_names.pkl'))
print("✅ features_names.pkl sauvegardé")

# Sauvegarder aussi les coefficients pour référence
coeff_data = {
    'linear_regression_coef': dict(zip(features_names, lr.coef_)),
    'linear_regression_intercept': lr.intercept_,
    'random_forest_importance': dict(zip(features_names, rf.feature_importances_))
}
joblib.dump(coeff_data, os.path.join(models_path, 'model_info.pkl'))
print("✅ model_info.pkl sauvegardé")

print("\n" + "=" * 70)
print("✅ TOUS LES MODÈLES ONT ÉTÉ SAUVEGARDÉS")
print("=" * 70)
print(f"\n📁 Dossier des modèles: {models_path}")
print("""
Fichiers créés:
   - linear_regression.pkl
   - ridge.pkl
   - lasso.pkl
   - random_forest.pkl
   - features_names.pkl
   - model_info.pkl
""")