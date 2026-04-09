# ------------------------------
# FICHIER : src/data_preprocessing.py
# À copier-coller dans src/data_preprocessing.py
# ------------------------------

preprocessing_content = '''
# -*- coding: utf-8 -*-
"""
Module de preprocessing pour le projet Walmart
Auteur : Data Scientist
Date : Avril 2026
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Charger les données depuis un fichier CSV
    
    Parameters:
    filepath (str): Chemin vers le fichier CSV
    
    Returns:
    pd.DataFrame: DataFrame chargé
    """
    df = pd.read_csv(filepath)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def remove_missing_target(df, target_col='Weekly_Sales'):
    """
    Supprimer les lignes où la cible est manquante
    
    Parameters:
    df (pd.DataFrame): DataFrame source
    target_col (str): Nom de la colonne cible
    
    Returns:
    pd.DataFrame: DataFrame nettoyé
    """
    initial_shape = df.shape[0]
    df_clean = df.dropna(subset=[target_col])
    removed = initial_shape - df_clean.shape[0]
    print(f"✅ {removed} lignes supprimées (target manquante)")
    return df_clean

def convert_date(df):
    """
    Convertir la colonne Date en datetime
    
    Parameters:
    df (pd.DataFrame): DataFrame source
    
    Returns:
    pd.DataFrame: DataFrame avec Date convertie
    """
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    print(f"✅ Date convertie: {df['Date'].isna().sum()} dates invalides")
    return df

def interpolate_dates(df):
    """
    Interpoler les dates manquantes par Store
    
    Parameters:
    df (pd.DataFrame): DataFrame source
    
    Returns:
    pd.DataFrame: DataFrame avec dates interpolées
    """
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    def interpoler_groupe(groupe):
        timestamps = groupe['Date'].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        timestamps_interp = timestamps.interpolate(method='linear', limit_direction='both')
        groupe['Date'] = pd.to_datetime(timestamps_interp, unit='s')
        return groupe
    
    df = df.groupby('Store', group_keys=False).apply(interpoler_groupe)
    print(f"✅ Dates interpolées")
    return df

def create_date_features(df):
    """
    Créer les features temporelles (Year, Month, Day)
    
    Parameters:
    df (pd.DataFrame): DataFrame source
    
    Returns:
    pd.DataFrame: DataFrame avec features temporelles
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # DayOfWeek est constante (4 = Vendredi) -> on ne la garde pas
    print(f"✅ Features temporelles créées: Year, Month, Day")
    return df

def remove_outliers_unemployment(df):
    """
    Supprimer les outliers dans Unemployment (méthode IQR)
    
    Parameters:
    df (pd.DataFrame): DataFrame source
    
    Returns:
    pd.DataFrame: DataFrame sans outliers
    """
    Q1 = df['Unemployment'].quantile(0.25)
    Q3 = df['Unemployment'].quantile(0.75)
    IQR = Q3 - Q1
    borne_sup = Q3 + 1.5 * IQR
    
    initial_shape = df.shape[0]
    df_clean = df[(df['Unemployment'] <= borne_sup) | (df['Unemployment'].isna())]
    removed = initial_shape - df_clean.shape[0]
    print(f"✅ {removed} outliers supprimés dans Unemployment")
    return df_clean

def impute_missing_values(df):
    """
    Imputer les valeurs manquantes
    - Holiday_Flag: mode (0)
    - Temperature: ffill/bfill par Store
    - Fuel_Price: ffill/bfill par Store
    - CPI: interpolation linéaire par Store
    - Unemployment: interpolation linéaire par Store
    """
    # Holiday_Flag
    df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0)
    
    # Temperature
    df['Temperature'] = df.groupby('Store')['Temperature'].ffill()
    df['Temperature'] = df.groupby('Store')['Temperature'].bfill()
    
    # Fuel_Price
    df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].ffill()
    df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].bfill()
    
    # CPI
    df['CPI'] = df.groupby('Store')['CPI'].transform(lambda x: x.interpolate(method='linear'))
    df['CPI'] = df.groupby('Store')['CPI'].ffill()
    df['CPI'] = df.groupby('Store')['CPI'].bfill()
    
    # Unemployment
    df['Unemployment'] = df.groupby('Store')['Unemployment'].transform(lambda x: x.interpolate(method='linear'))
    df['Unemployment'] = df.groupby('Store')['Unemployment'].ffill()
    df['Unemployment'] = df.groupby('Store')['Unemployment'].bfill()
    
    print(f"✅ Imputation terminée")
    return df

def prepare_data(filepath):
    """
    Pipeline complet de preprocessing
    
    Parameters:
    filepath (str): Chemin vers le fichier CSV
    
    Returns:
    pd.DataFrame: DataFrame prêt pour le machine learning
    """
    print("=" * 50)
    print("PREPROCESSING - WALMART SALES")
    print("=" * 50)
    
    df = load_data(filepath)
    df = remove_missing_target(df)
    df = convert_date(df)
    df = interpolate_dates(df)
    df = create_date_features(df)
    df = remove_outliers_unemployment(df)
    df = impute_missing_values(df)
    
    # Supprimer la colonne Date
    df = df.drop(columns=['Date'])
    
    print("=" * 50)
    print(f"✅ Dataset final: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("=" * 50)
    
    return df

if __name__ == "__main__":
    # Test du pipeline
    df = prepare_data('../data/walmart_store_sales.csv')
    print(f"\\nColonnes finales: {df.columns.tolist()}")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
'''

print(preprocessing_content)
print("\n✅ Copie ce code dans src/data_preprocessing.py")