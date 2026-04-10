import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================================================
# set_page_config DOIT ÊTRE LA PREMIÈRE COMMANDE
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# COULEURS OFFICIELLES WALMART
# ============================================================
WALMART_BLUE = "#0071CE"
WALMART_YELLOW = "#FFC220"
WALMART_WHITE = "#FFFFFF"
WALMART_DARK_BLUE = "#0056A3"
WALMART_DARK_YELLOW = "#E6A800"

# ============================================================
# STYLE (Sidebar non modifié, textes lisibles)
# ============================================================
import streamlit as st

st.markdown("""
<style>

section.main > div {
    background-color: white;
}

@media (prefers-color-scheme: dark) {
    section.main > div {
        background-color: transparent;
    }
}

section.main h1, 
section.main h2, 
section.main h3 {
    color: #0071CE;
}

section.main p, 
section.main label {
    color: #000000;
}

@media (prefers-color-scheme: dark) {
    section.main h1, 
    section.main h2, 
    section.main h3 {
        color: #FFC220;
    }

    section.main p, 
    section.main label {
        color: #FFFFFF;
    }
}

section.main .stButton > button {
    background-color: #0071CE;
    color: white;
    border-radius: 8px;
    border: none;
}

section.main .stButton > button:hover {
    background-color: #0056A3;
}

section.main .stAlert {
    border-left: 4px solid #0071CE;
}

section.main .stDataFrame {
    background-color: white;
}

div.stSlider > div > div > div > div {
     background-color: #FFCC00 !important;
}
div.stSlider > div > div > div > div > div {
    background-color: #FFCC00 !important;
}
div[role="radiogroup"] input:checked + label {
    background-color: #FFCC00 !important;
    color: black !important;           

</style>
""", unsafe_allow_html=True)

st.title("📊 Walmart Sales Prediction")
st.markdown("---")

# ============================================================
# CHARGEMENT DES MODÈLES (CACHÉ)
# ============================================================
@st.cache_resource
def load_models():
    """Charger les modèles sauvegardés"""
    import os
    import joblib
    import streamlit as st

    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    models = {}
    try:
        models['random_forest'] = joblib.load(os.path.join(models_path, 'random_forest.pkl'))
        models['linear_regression'] = joblib.load(os.path.join(models_path, 'linear_regression.pkl'))
        models['ridge'] = joblib.load(os.path.join(models_path, 'ridge.pkl'))
        models['lasso'] = joblib.load(os.path.join(models_path, 'lasso.pkl'))
        models['features'] = joblib.load(os.path.join(models_path, 'features_names.pkl'))
        st.success("✅ Modèles chargés avec succès!")
    except Exception as e:
        st.error(f"❌ Erreur de chargement des modèles: {e}")
        models = None
    
    return models

# ============================================================
# CHARGEMENT DES DONNÉES POUR EDA
# ============================================================
@st.cache_data
def load_and_clean_data():
    """Charger et nettoyer les données pour l'EDA"""
    try:
        import os
        import pandas as pd
    
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'walmart_store_sales.csv')
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['Weekly_Sales'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
            
        def interpoler_dates(groupe):
            timestamps = groupe['Date'].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
            timestamps_interp = timestamps.interpolate(method='linear', limit_direction='both')
            groupe['Date'] = pd.to_datetime(timestamps_interp, unit='s')
            return groupe
        
        df = df.groupby('Store', group_keys=False).apply(interpoler_dates)
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        # NE PAS supprimer Date ici - on la garde pour les graphiques temporels
        # df = df.drop(columns=['Date'])  <-- À SUPPRIMER ou commenter
        
        Q1 = df['Unemployment'].quantile(0.25)
        Q3 = df['Unemployment'].quantile(0.75)
        IQR = Q3 - Q1
        borne_sup = Q3 + 1.5 * IQR
        df = df[(df['Unemployment'] <= borne_sup) | (df['Unemployment'].isna())]
        
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
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement des données: {e}")
        return None

# Charger les modèles
models = load_models()

# Sidebar
with st.sidebar:
    st.image("Walmart-Logo.png", width=200)
    st.markdown("""
    <h1 style='color:#0071CE;'>Walmart Dashboard</h1>
    """, unsafe_allow_html=True)

    st.header("Navigation")
    page = st.radio(
        "Choisissez une page:",
        ["Accueil", "Prédiction", "Analyse des données (EDA)", "Performance", "Documentation"]
    )

    st.markdown("---")
    st.markdown("### 👨‍💻 Présenté par")
    st.markdown("**Mohammed SHAQURA**")
    st.markdown("Data Analyst | Walmart Project")
    st.markdown("**Jedha Bootcamp**")

# ============================================================
# PAGE ACCUEIL
# ============================================================
if page == "Accueil":
    st.header("🏠 Bienvenue")
    st.info("""
    💡Dans ce projet:  
    - J'ai choisi choisi Random Forest car il a montré d'excellentes performances (R² = 0.7448). 
    - Dans un contexte professionnel, je recommanderais de tester également XGBoost et LightGBM pour potentiellement 
      améliorer encore la précision.
   - Le choix final dépendrait des contraintes de temps et de performance attendues par Walmart.
    """)

    st.markdown("""
    ## 🎯 Objectif
    
    Prédire les **ventes hebdomadaires** des magasins Walmart.
    
    ## 🏆 Meilleur modèle
    
    **Random Forest** - R² = 0.7448
    
    ## 📊 Modèles disponibles
    
    - Linear Regression (baseline)
    - Ridge (régularisation L2)
    - Lasso (régularisation L1)
    - **Random Forest** (recommandé)
    """)

# ============================================================
# PAGE PRÉDICTION (AVEC MODÈLES RÉELS)
# ============================================================
elif page == "Prédiction":
    st.header("📈 Prédiction des ventes")
    
    if models is None:
        st.error("❌ Modèles non disponibles. Veuillez exécuter src/save_models.py d'abord.")
        st.stop()
    
    st.info("📅 **Période prédite:** Ventes de la SEMAINE complète (7 jours)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏪 Magasin")
        store = st.number_input("Numéro du magasin", min_value=1.0, max_value=45.0, value=1.0, step=1.0)
        holiday_flag = st.selectbox("Semaine de fête", [0, 1], format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non")
        
        st.subheader("🌡️ Température")
        temp_unit = st.radio("Unité:", ["Celsius (°C)", "Fahrenheit (°F)"], horizontal=True)
        
        if temp_unit == "Celsius (°C)":
            temp_celsius = st.slider("Température (°C)", -20.0, 50.0, 15.0, 0.5)
            temperature = (temp_celsius * 9/5) + 32
        else:
            temperature = st.slider("Température (°F)", 0.0, 120.0, 60.0, 0.5)
        
        fuel_price = st.slider("Prix du carburant ($)", 2.0, 5.0, 3.5, 0.05)
    
    with col2:
        st.subheader("📉 Économie")
        cpi = st.slider("CPI", 120.0, 240.0, 180.0, 1.0)
        unemployment = st.slider("Taux de chômage (%)", 3.0, 15.0, 7.0, 0.1)
        
        st.subheader("📅 Date")
        year = st.number_input("Année", 2010, 2013, 2011)
        month = st.number_input("Mois", 1, 12, 6)
        day = st.number_input("Jour", 1, 31, 15)
    
    # Choix du modèle
    st.subheader("🎯 Choix du modèle")
    model_choice = st.selectbox(
        "Sélectionnez un modèle:",
        options=['random_forest', 'linear_regression', 'ridge', 'lasso'],
        format_func=lambda x: {
            'random_forest': '🌲 Random Forest (Recommandé)',
            'linear_regression': '📏 Linear Regression (Baseline)',
            'ridge': '🔷 Ridge (L2)',
            'lasso': '🔶 Lasso (L1)'
        }[x]
    )
    
    if st.button("🚀 Prédire", type="primary", use_container_width=True):
        # Préparer les données d'entrée
        input_data = np.array([[store, holiday_flag, temperature, fuel_price, cpi, unemployment, year, month, day]])
        
        # Prédiction avec le modèle choisi
        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        prediction = max(prediction, 200000)  # Minimum raisonnable
        
        st.success(f"## 📊 Ventes estimées pour la semaine: **${prediction:,.2f}**")
        
        daily_avg = prediction / 7
        st.metric("Moyenne par jour", f"${daily_avg:,.2f}")
        
        # Afficher le modèle utilisé
        st.caption(f"Modèle utilisé: {model_choice.replace('_', ' ').title()}")

# ============================================================
# PAGE ANALYSE DES DONNÉES (EDA) - IDENTIQUE NOTEBOOK
# ============================================================
elif page == "Analyse des données (EDA)":
    st.header("📊 Analyse exploratoire des données (EDA)")
    
    df_clean = load_and_clean_data()
    
    if df_clean is None:
        st.error("Impossible de charger les données")
        st.stop()
    
    st.success(f"✅ Données chargées: {df_clean.shape[0]} lignes")
    
    # Choix du graphique
    graph_choice = st.selectbox(
        "Choisissez une visualisation:",
        [
            "1. Matrice de corrélation + Corrélation avec Weekly_Sales",
            "2. Relation Unemployment vs Weekly_Sales (Paradoxe de Simpson)",
            "3. Tendance temporelle (Chômage + Ventes)",
            "4. Corrélation par magasin (Unemployment vs Sales)"
        ]
    )
    
    # ============================================================
    # GRAPHIQUE 1: Matrice de corrélation + barres
    # ============================================================
    if graph_choice == "1. Matrice de corrélation + Corrélation avec Weekly_Sales":
        st.subheader("Analyse des corrélations après nettoyage")
        
        cols_numeriques = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        # Matrice de corrélation
        correlation_matrix = df_clean[cols_numeriques + ['Weekly_Sales']].corr()
        
        st.write("**Matrice de corrélation:**")
        st.dataframe(correlation_matrix.round(3), use_container_width=True)
        
        # Heatmap
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f', linewidths=0.5, ax=ax1)
        ax1.set_title('Matrice de corrélation entre les variables (après nettoyage)', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Corrélation avec la cible
        st.subheader("Corrélation avec la cible (Weekly_Sales)")
        corr_avec_cible = correlation_matrix['Weekly_Sales'].drop('Weekly_Sales').sort_values(ascending=False)
        
        for var, corr in corr_avec_cible.items():
            if abs(corr) > 0.5:
                niveau = "Forte corrélation"
            elif abs(corr) > 0.3:
                niveau = "Corrélation modérée"
            else:
                niveau = "Faible corrélation"
            st.write(f"  **{var}**: {corr:.3f} → {niveau}")
        
        # Barplot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors = ['steelblue' if x > 0 else 'salmon' for x in corr_avec_cible.values]
        corr_avec_cible.plot(kind='bar', color=colors, ax=ax2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0.3, color='red', linestyle='--', label='Seuil corrélation modérée (0.3)')
        ax2.axhline(y=-0.3, color='red', linestyle='--')
        ax2.set_title('Corrélation des variables avec Weekly_Sales', fontsize=14)
        ax2.set_ylabel('Coefficient de corrélation')
        ax2.set_xlabel('Variables')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.info("""
        **Résumé:** Les corrélations avec Weekly_Sales sont toutes FAIBLES (entre -0.3 et 0.3).
        Cela signifie qu'aucune variable prise isolément n'explique fortement les ventes.
        """)
    
    # ============================================================
    # GRAPHIQUE 2: Paradoxe de Simpson
    # ============================================================
    elif graph_choice == "2. Relation Unemployment vs Weekly_Sales (Paradoxe de Simpson)":
        st.subheader("Visualisation par Store - Paradoxe de Simpson")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Scatter plot par Store
        sns.scatterplot(data=df_clean, x='Unemployment', y='Weekly_Sales', 
                        hue='Store', palette='tab20', alpha=0.7, legend='brief', ax=ax)
        
        # Ligne de régression globale
        sns.regplot(data=df_clean, x='Unemployment', y='Weekly_Sales', 
                    scatter=False, color='black', label='Tendance globale', ax=ax)
        
        ax.set_title('Relation Unemployment vs Weekly_Sales par magasin', fontsize=14)
        ax.set_xlabel('Taux de chômage (%)', fontsize=12)
        ax.set_ylabel('Ventes hebdomadaires ($)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **Observation (Paradoxe de Simpson):**
        - La ligne NOIRE (tendance globale) est ascendante → corrélation positive
        - Mais chaque magasin montre une tendance descendante
        - Les magasins avec plus de chômage (Store 4, 5) ont aussi plus de ventes
        - Mais À L'INTÉRIEUR de chaque magasin, plus de chômage = moins de ventes
        """)
    
    # ============================================================
    # GRAPHIQUE 3: Tendance temporelle (2 graphiques côte à côte)
    # ============================================================
    elif graph_choice == "3. Tendance temporelle (Chômage + Ventes)":
        st.subheader("Analyse de la tendance temporelle")
        
        # Deux graphiques côte à côte
        col1, col2 = st.columns(2)
        
        # Graphique 1: Unemployment
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            for store in df_clean['Store'].unique()[:5]:
                df_store = df_clean[df_clean['Store'] == store].sort_values('Date')
                ax1.plot(df_store['Date'], df_store['Unemployment'], marker='o', label=f'Store {int(store)}')
            ax1.set_title('Tendance du taux de chômage par magasin', fontsize=12)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Unemployment (%)')
            ax1.legend()
            ax1.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
        
        # Graphique 2: Weekly_Sales
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for store in df_clean['Store'].unique()[:5]:
                df_store = df_clean[df_clean['Store'] == store].sort_values('Date')
                ax2.plot(df_store['Date'], df_store['Weekly_Sales'], marker='o', label=f'Store {int(store)}')
            ax2.set_title('Tendance des ventes par magasin', fontsize=12)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Weekly_Sales ($)')
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Corrélations temporelles
        st.subheader("Corrélations temporelles")
        st.write(f"📊 Corrélation entre Year et Unemployment: **{df_clean['Year'].corr(df_clean['Unemployment']):.3f}**")
        st.write(f"📊 Corrélation entre Year et Weekly_Sales: **{df_clean['Year'].corr(df_clean['Weekly_Sales']):.3f}**")
        
        st.caption("📌 Observation: Le Store 4 a un chômage plus bas mais des ventes plus élevées")
    
    # ============================================================
    # GRAPHIQUE 4: Corrélation par magasin (horizontal)
    # ============================================================
    else:
        st.subheader("Analyse par magasin - Corrélation Unemployment vs Sales")
        
        # Calculer les corrélations
        correlations = []
        for store in df_clean['Store'].unique():
            df_store = df_clean[df_clean['Store'] == store]
            if len(df_store) > 5:
                corr = df_store['Unemployment'].corr(df_store['Weekly_Sales'])
                correlations.append({'Store': int(store), 'Correlation': corr, 'n_samples': len(df_store)})
        
        df_corr = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        
        st.write("**Corrélations par magasin (triées par ordre décroissant):**")
        st.dataframe(df_corr, use_container_width=True, hide_index=True)
        
        # Résultats
        corr_positives = df_corr[df_corr['Correlation'] > 0]
        corr_negatives = df_corr[df_corr['Correlation'] < 0]
        
        st.write(f"✅ **Magasins avec corrélation POSITIVE:** {len(corr_positives)} → {corr_positives['Store'].tolist()}")
        st.write(f"✅ **Magasins avec corrélation NÉGATIVE:** {len(corr_negatives)} → {corr_negatives['Store'].tolist()}")
        
        # Graphique horizontal (barres)
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if c > 0 else 'red' for c in df_corr['Correlation']]
        ax.barh(df_corr['Store'].astype(str), df_corr['Correlation'], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("Coefficient de corrélation")
        ax.set_ylabel("Store")
        ax.set_title('Corrélation Unemployment vs Weekly_Sales par magasin (vert = positive, rouge = négative)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Top 5
        st.write("**TOP 5 magasins avec corrélation POSITIVE:**")
        for _, row in df_corr.head(5).iterrows():
            st.write(f"  Store {int(row['Store'])}: {row['Correlation']:.3f}")
        
        st.write("**TOP 5 magasins avec corrélation NÉGATIVE:**")
        for _, row in df_corr.tail(5).iterrows():
            st.write(f"  Store {int(row['Store'])}: {row['Correlation']:.3f}")

# ============================================================
# PAGE PERFORMANCE
# ============================================================
elif page == "Performance":
    st.header("📊 Performance des modèles")
    
    data = {
        "Modèle": ["Linear Regression", "Ridge", "Lasso", "Random Forest"],
        "R² Test": [0.1520, 0.1532, 0.1520, 0.7448],
        "RMSE ($)": [588490, 588069, 588492, 322809],
        "MAE ($)": [541390, 542354, 541393, 247209]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.success("🏆 **Random Forest** est le meilleur modèle avec R² = 0.7448")
    
    st.subheader("Comparaison R²")
    st.bar_chart(df.set_index("Modèle")["R² Test"])
    
    st.subheader("Importance des features (Random Forest)")
    importance = {
        "Feature": ["Unemployment", "Store", "CPI", "Month", "Temperature", "Fuel_Price", "Day", "Year", "Holiday_Flag"],
        "Importance": [0.2511, 0.2493, 0.1569, 0.1272, 0.0774, 0.0743, 0.0506, 0.0107, 0.0027]
    }
    df_imp = pd.DataFrame(importance)
    st.bar_chart(df_imp.set_index("Feature"))
    
    st.info("""
    💡 **Conclusion:** Random Forest est bien plus performant car il capture les relations NON LINÉAIRES.
    """)

# ============================================================
# PAGE DOCUMENTATION
# ============================================================
else:
    st.header("📚 Documentation")
    
    st.markdown("""
    ### Découverte clé: Paradoxe de Simpson
    
    La corrélation globale entre chômage et ventes est **positive** (+0.241),
    mais la corrélation par magasin est **négative** pour 8 magasins sur 13!
    
    
""")


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'> Walmart Dashboard - Prédiction des ventes | Créé avec Streamlit </p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'> *Projet fait pour la certifiction de BLOC 3* | Jedha Bootcamp </p>",
    unsafe_allow_html=True
)
