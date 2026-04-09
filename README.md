# Walmart Sales Prediction

## 🎯 Objectif du projet
Prédire les ventes hebdomadaires des magasins Walmart à partir d'indicateurs économiques.

## 📊 Dataset
- 150 lignes, 8 colonnes
- Features: Store, Date, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- Target: Weekly_Sales

## 🔧 Prérequis
```bash
pip install -r requirements.txt

📁 Structure du projet
walmart_sales_prediction/
├── data/                 # Dataset
├── notebooks/            # Analyse complète
├── src/                  # Code source
├── models/               # Modèles sauvegardés
├── app/                  # Application Streamlit
└── reports/              # Rapport

🚀 Modèles entraînés
Modèle	R² Test	RMSE
Linear Regression	0.152	588 490 $
Ridge	0.153	588 069 $
Lasso	0.152	588 492 $
Random Forest	0.745	322 809 $
🏆 Meilleur modèle
Random Forest - R² = 0.7448

🌐 Démo Streamlit
streamlit run app/streamlit_app.py

👥 Auteur
Mohammed SHAQURA - Projet réalisé dans le cadre de la formation Data Analysis.