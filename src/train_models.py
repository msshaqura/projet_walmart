# ------------------------------
# FICHIER : src/train_models.py
# À copier-coller dans src/train_models.py
# ------------------------------

train_models_content = '''
# -*- coding: utf-8 -*-
"""
Module d'entraînement des modèles pour le projet Walmart
Auteur : Data Scientist
Date : Avril 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Importer le module de preprocessing
from data_preprocessing import prepare_data

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Diviser les données en train et test
    
    Parameters:
    X (pd.DataFrame): Features
    y (pd.Series): Target
    test_size (float): Proportion du test set
    random_state (int): Graine aléatoire
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Train set: {X_train.shape[0]} lignes")
    print(f"✅ Test set: {X_test.shape[0]} lignes")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """
    Entraîner une régression linéaire
    
    Returns:
    LinearRegression: Modèle entraîné
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Linear Regression entraîné")
    return model

def train_ridge(X_train, y_train, alpha=1.0):
    """
    Entraîner une régression Ridge
    
    Returns:
    Ridge: Modèle entraîné
    """
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    print(f"✅ Ridge (alpha={alpha}) entraîné")
    return model

def train_lasso(X_train, y_train, alpha=1.0):
    """
    Entraîner une régression Lasso
    
    Returns:
    Lasso: Modèle entraîné
    """
    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    print(f"✅ Lasso (alpha={alpha}) entraîné")
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Entraîner un Random Forest
    
    Returns:
    RandomForestRegressor: Modèle entraîné
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"✅ Random Forest (n_estimators={n_estimators}) entraîné")
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Évaluer un modèle sur train et test
    
    Returns:
    dict: Métriques de performance
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    return metrics, y_test_pred

def display_coefficients(model, feature_names):
    """
    Afficher les coefficients du modèle linéaire
    """
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        print("\\n📊 Coefficients du modèle:")
        for feature, coef in zip(feature_names, coefs):
            print(f"  {feature}: {coef:.2f}")

def display_feature_importance(model, feature_names):
    """
    Afficher l'importance des features pour Random Forest
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print("\\n📊 Importance des features:")
        for feature, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {imp:.4f}")

def save_model(model, filepath):
    """
    Sauvegarder un modèle avec joblib
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé: {filepath}")

def load_model(filepath):
    """
    Charger un modèle avec joblib
    """
    return joblib.load(filepath)

def main():
    """
    Pipeline principal d'entraînement
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES - WALMART")
    print("=" * 60)
    
    # 1. Préparer les données
    df = prepare_data('../data/walmart_store_sales.csv')
    
    # 2. Séparer X et y
    X = df.drop(columns=['Weekly_Sales'])
    y = df['Weekly_Sales']
    
    print(f"\\n📊 Features: {X.columns.tolist()}")
    print(f"📊 Target: Weekly_Sales")
    
    # 3. Division train/test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Entraîner les modèles
    models = {}
    results = {}
    
    # Linear Regression
    print("\\n" + "-" * 40)
    lr = train_linear_regression(X_train, y_train)
    models['linear_regression'] = lr
    metrics, _ = evaluate_model(lr, X_train, X_test, y_train, y_test)
    results['linear_regression'] = metrics
    print(f"   R² Test: {metrics['test_r2']:.4f}")
    display_coefficients(lr, X.columns)
    
    # Ridge
    print("\\n" + "-" * 40)
    ridge = train_ridge(X_train, y_train, alpha=1.0)
    models['ridge'] = ridge
    metrics, _ = evaluate_model(ridge, X_train, X_test, y_train, y_test)
    results['ridge'] = metrics
    print(f"   R² Test: {metrics['test_r2']:.4f}")
    
    # Lasso
    print("\\n" + "-" * 40)
    lasso = train_lasso(X_train, y_train, alpha=1.0)
    models['lasso'] = lasso
    metrics, _ = evaluate_model(lasso, X_train, X_test, y_train, y_test)
    results['lasso'] = metrics
    print(f"   R² Test: {metrics['test_r2']:.4f}")
    
    # Random Forest
    print("\\n" + "-" * 40)
    rf = train_random_forest(X_train, y_train)
    models['random_forest'] = rf
    metrics, _ = evaluate_model(rf, X_train, X_test, y_train, y_test)
    results['random_forest'] = metrics
    print(f"   R² Test: {metrics['test_r2']:.4f}")
    display_feature_importance(rf, X.columns)
    
    # 5. Sauvegarder les modèles
    print("\\n" + "-" * 40)
    print("SAUVEGARDE DES MODÈLES")
    save_model(lr, '../models/linear_regression.pkl')
    save_model(ridge, '../models/ridge.pkl')
    save_model(lasso, '../models/lasso.pkl')
    save_model(rf, '../models/random_forest.pkl')
    
    # 6. Résumé final
    print("\\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    print(f"\\n{'Modèle':<20} {'R² Train':<12} {'R² Test':<12} {'RMSE Test':<15}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f} {metrics['test_rmse']:<15.2f}")
    
    print("\\n✅ Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()
'''

print(train_models_content)
print("\n✅ Copie ce code dans src/train_models.py")