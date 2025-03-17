import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import xlsxwriter
import random
import time

# Configuration de la page
st.set_page_config(page_title="Contrôle Métrologique des Préemballages", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Styles CSS personnalisés
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----- FONCTIONS D'UTILITAIRES -----

# Calcul de l'EMT selon l'annexe 5 du guide DGCCRF
def calculer_emt(qn):
    """
    Calcule l'erreur maximale tolérée selon l'annexe 5 du guide DGCCRF
    """
    if qn < 50:
        return 9/100 * qn
    elif qn < 100:
        return 4.5
    elif qn < 200:
        return 4.5/100 * qn
    elif qn < 300:
        return 9
    elif qn < 500:
        return 3/100 * qn
    elif qn < 1000:
        return 15
    elif qn < 10000:
        return 1.5/100 * qn
    else:
        return 150

# Tableau pour l'Annexe 4
TABLE_POM_X = [0.184, 0.175, 0.167, 0.161, 0.155, 0.149, 0.144, 0.140, 0.136, 0.132, 0.129, 0.126, 
              0.123, 0.120, 0.118, 0.115, 0.113, 0.111, 0.109, 0.107, 0.105, 0.103, 0.102, 0.100, 
              0.099, 0.097, 0.096, 0.095, 0.093, 0.092, 0.091, 0.090, 0.089, 0.088, 0.087, 0.086, 
              0.085, 0.084, 0.083, 0.082, 0.081, 0.080, 0.079, 0.078, 0.077, 0.076, 0.075, 0.074, 
              0.073, 0.072]

TABLE_POM_Y = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 
              145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 
              230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 300]

# Tableau pour l'Annexe 3
TABLE_POM_X_ANNEX3 = [0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.94, 1.06, 1.17, 1.28, 1.39, 1.50, 
                      1.61, 1.72, 1.83, 1.94, 2.05, 2.17, 2.28, 2.39, 2.50, 2.61, 2.72, 2.83, 2.94, 
                      3.05, 3.16, 3.27, 3.38]

TABLE_POM_Y_ANNEX3 = [328, 108, 66, 44, 30, 23, 18, 15, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 3, 
                      3, 2, 2, 2, 2, 2, 2]

# Coefficients statistiques pour différentes tailles d'échantillons
COEF_G = {
    5: 0.686,
    7: 0.787,
    10: 0.902,
    15: 1.047,
    20: 1.14,
    30: 1.264,
    50: 1.402
}

# Fonction pour le test de normalité de Shapiro-Wilk
def shapiro_test(data):
    stat, p = stats.shapiro(data)
    alpha = 0.05
    if p > alpha:
        return "Distribution normale", p, True
    else:
        return "Distribution non normale", p, False

# Fonction pour calculer le POM à partir de l'annexe 4 du guide DGCCRF
def calcul_pom_defectueux(delta_sqrt_n):
    """
    Calcule le POM pour les défauts en utilisant l'annexe 4
    """
    # Limiter delta_sqrt_n aux bornes de TABLE_POM_X
    if delta_sqrt_n < min(TABLE_POM_X):
        delta_sqrt_n = min(TABLE_POM_X)
    elif delta_sqrt_n > max(TABLE_POM_X):
        delta_sqrt_n = max(TABLE_POM_X)

    # Interpolation linéaire pour les valeurs manquantes
    f = interp1d(TABLE_POM_X, TABLE_POM_Y, kind='linear', fill_value="extrapolate")

    # S'assurer que delta_sqrt_n est un nombre, pas un tableau
    if isinstance(delta_sqrt_n, (np.ndarray, np.float64)):
        delta_sqrt_n = float(delta_sqrt_n)

    # Forcer le résultat de l'interpolation en float
    pom = float(f(delta_sqrt_n))

    return round(pom)

# Fonction pour calculer le POM à partir de l'annexe 3 du guide DGCCRF
def calcul_pom_moyenne(delta_sqrt_n):
    """
    Calcule le POM pour la moyenne en utilisant l'annexe 3
    """
    # Limiter delta_sqrt_n aux bornes de TABLE_POM_X_ANNEX3
    if delta_sqrt_n < min(TABLE_POM_X_ANNEX3):
        delta_sqrt_n = min(TABLE_POM_X_ANNEX3)
    elif delta_sqrt_n > max(TABLE_POM_X_ANNEX3):
        delta_sqrt_n = max(TABLE_POM_X_ANNEX3)

    # Interpolation linéaire pour les valeurs manquantes
    f = interp1d(TABLE_POM_X_ANNEX3, TABLE_POM_Y_ANNEX3, kind='linear', fill_value="extrapolate")

    # S'assurer que delta_sqrt_n est un nombre, pas un tableau
    if isinstance(delta_sqrt_n, (np.ndarray, np.float64)):
        delta_sqrt_n = float(delta_sqrt_n)

    # Forcer le résultat de l'interpolation en float
    pom = float(f(delta_sqrt_n))

    return round(pom)

# Création d'un jeu de données de test
def generer_donnees_test(qn, sigma, tare_moyenne, tare_sigma, n_samples=50):
    """
    Génère un jeu de données de test pour la simulation
    """
    # Générer des poids de tare
    tares = np.random.normal(tare_moyenne, tare_sigma, n_samples)
    tares = np.clip(tares, tare_moyenne - 3*tare_sigma, tare_moyenne + 3*tare_sigma)
    
    # Générer des poids nets
    nets = np.random.normal(qn, sigma, n_samples)
    nets = np.clip(nets, qn - 3*sigma, qn + 3*sigma)
    
    # Calculer les poids bruts
    bruts = tares + nets
    
    # Créer le dataframe
    df = pd.DataFrame({
        "Tare (g)": tares,
        "Poids Brut (g)": bruts
    })
    
    return df

# ----- INTERFACE UTILISATEUR -----

# En-tête
st.title("🔍 Validation d'échantillonnage pour le contrôle métrologique")
st.markdown("---")

# Créer un onglet
tabs = st.tabs(["📊 Analyse", "🧪 Simulation", "📝 Documentation", "📥 Téléchargements"])

with tabs[0]:  # Onglet Analyse
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Hypothèses de travail")
        qn = st.number_input("Quantité Nominale (g):", value=500.0, min_value=1.0, help="Poids nominal indiqué sur l'emballage.")
        # Calculer automatiquement l'EMT selon la réglementation
        e_calc = calculer_emt(qn)
        e = st.number_input("Erreur maximale tolérée (g):", value=float(e_calc), min_value=0.1, help=f"Valeur calculée selon l'annexe 5 du guide DGCCRF: {e_calc:.2f}g")
        surpoids_max = st.number_input("Surpoids maximum toléré (g):", value=2.0, min_value=0.0, help="Surpoids maximal acceptable pour chaque pack.")
    
    with col2:
        st.subheader("Stratégie d'échantillonnage")
        n_options = list(COEF_G.keys())
        n_index = n_options.index(5) if 5 in n_options else 0
        n = st.selectbox("Effectif d'échantillon:", n_options, index=n_index, help="Nombre de packs à peser dans chaque échantillon.")
        frequence = st.number_input("Fréquence d'échantillonnage (par heure):", value=1.0, min_value=1.0, help="Nombre d'échantillons à contrôler par heure.")
        max_defectueux = st.number_input("Nombre max de défectueux tolérés pour 100 unités:", value=2.5, min_value=0.0, max_value=10.0, help="Selon la règle des 2,5%")

    # Téléchargement du fichier Excel
    st.subheader("📤 Téléchargement du fichier Excel de pesée")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx"])
    with col2:
        test_option = st.checkbox("Utiliser des données de test", value=False)
        if test_option:
            test_sigma = st.slider("Écart-type du processus (g):", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
            test_tare = st.slider("Tare moyenne (g):", min_value=1.0, max_value=50.0, value=15.0, step=0.5)
            test_tare_sigma = st.slider("Écart-type de la tare (g):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
            
            if st.button("Générer des données de test"):
                # Générer des données de test
                df_test = generer_donnees_test(qn, test_sigma, test_tare, test_tare_sigma)
                uploaded_file = BytesIO()
                with pd.ExcelWriter(uploaded_file, engine='xlsxwriter') as writer:
                    df_test.to_excel(writer, sheet_name='Pesées', index=False)
                uploaded_file.seek(0)
                
                # Afficher un message de succès
                st.success("Données de test générées avec succès!")
    with col3:
        st.markdown("**Besoin d'un modèle?**")
        st.markdown("""
        <a href="#" onclick="document.getElementById('tabs-4').click();" style="text-decoration:none;">
            <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                📥  Voir onglet téléchargement pour <br>un modèle Excel vierge
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Traitement lorsqu'un fichier est téléchargé
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="Pesées")
            
            # Vérification des colonnes
            required_cols = ["Tare (g)", "Poids Brut (g)"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"Le fichier doit contenir les colonnes suivantes: {', '.join(required_cols)}")
                st.stop()
                
            tare_poids = df["Tare (g)"].dropna().tolist()
            poids_bruts = df["Poids Brut (g)"].dropna().tolist()
            
            # Calcul des poids nets
            df["Poids Net (g)"] = df["Poids Brut (g)"] - df["Tare (g)"]
            poids_nets = df["Poids Net (g)"].dropna().tolist()
            
            # Identification des défectueux
            df["Défectueux"] = df["Poids Net (g)"] < (qn - e)
            nombre_defectueux = df["Défectueux"].sum()
            pourcentage_defectueux = (nombre_defectueux / len(df)) * 100
            
            # Configuration des colonnes pour l'affichage des résultats
            col1, col2 = st.columns(2)
            
            # Analyse de la tare
            with col1:
                st.subheader("📦 Analyse de la Tare")
                tare_moy = np.mean(tare_poids)
                tare_std = np.std(tare_poids)
                test_result, p_value, is_normal = shapiro_test(tare_poids)
                
                info_metrics = {
                    "Nombre d'emballages": len(tare_poids),
                    "Moyenne de la tare": f"{tare_moy:.2f} g",
                    "Écart-type de la tare": f"{tare_std:.2f} g",
                    "Test de normalité": f"{test_result} (p = {p_value:.3f})"
                }
                
                # Affichage des métriques
                for key, value in info_metrics.items():
                    st.metric(key, value)
                
                # Histogramme de la tare
                fig_tare = px.histogram(df, x="Tare (g)", title="Distribution de la Tare",
                                      labels={"Tare (g)": "Tare (g)", "count": "Fréquence"},
                                      color_discrete_sequence=["#3366CC"])
                fig_tare.add_vline(x=tare_moy, line_dash="dash", line_color="red",
                                annotation_text=f"Moyenne: {tare_moy:.2f} g",
                                annotation_position="top right")
                st.plotly_chart(fig_tare, use_container_width=True)
                
                # Alerte si la distribution n'est pas normale
                if not is_normal:
                    st.markdown(
                        """
                        <div class="warning-box">
                        ⚠️ <b>Attention:</b> La distribution de la tare n'est pas normale. 
                        Cela peut affecter les résultats de l'analyse.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Analyse du poids brut et net
            with col2:
                st.subheader("🏋️ Analyse du Poids")
                
                # Métriques de poids brut
                brut_moy = np.mean(poids_bruts)
                brut_std = np.std(poids_bruts)
                
                # Métriques de poids net
                net_moy = np.mean(poids_nets)
                net_std = np.std(poids_nets)
                test_result_net, p_value_net, is_normal_net = shapiro_test(poids_nets)
                
                # Affichage des métriques
                st.metric("Poids brut moyen", f"{brut_moy:.2f} g")
                st.metric("Écart-type du poids brut", f"{brut_std:.2f} g")
                st.metric("Poids net moyen", f"{net_moy:.2f} g")
                st.metric("Écart-type du poids net", f"{net_std:.2f} g")
                st.metric("Test de normalité (poids net)", f"{test_result_net} (p = {p_value_net:.3f})")
                
                # Histogramme du poids net
                fig_net = px.histogram(df, x="Poids Net (g)", title="Distribution du Poids Net",
                                      labels={"Poids Net (g)": "Poids Net (g)", "count": "Fréquence"},
                                      color_discrete_sequence=["#FF9900"])
                fig_net.add_vline(x=qn, line_dash="dash", line_color="green",
                                annotation_text=f"QN: {qn} g",
                                annotation_position="top right")
                fig_net.add_vline(x=qn-e, line_dash="dash", line_color="red",
                                annotation_text=f"TU1: {qn-e} g",
                                annotation_position="top left")
                st.plotly_chart(fig_net, use_container_width=True)
                
                # Alerte si la distribution n'est pas normale
                if not is_normal_net:
                    st.markdown(
                        """
                        <div class="warning-box">
                        ⚠️ <b>Attention:</b> La distribution du poids net n'est pas normale. 
                        Cela peut affecter les résultats de l'analyse.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Calcul des critères de conformité
            sigma_0 = np.std(poids_nets)  # Utilisation du poids net au lieu du poids brut
            
            # Calcul du seuil de centrage selon les règles DGCCRF
            ms = qn if sigma_0 <= e/2.05 else qn - e + 2.05 * sigma_0
            qc = ms + surpoids_max
            g = COEF_G[n]  # Coefficient pour la taille d'échantillon choisie
            tu1 = qn - e
            
            # Comparaison POM vs POl
            m2 = qn - e + 2.05 * sigma_0
            delta = (qc - m2) / sigma_0
            
            # Affichage des résultats de l'analyse
            st.subheader("📊 Résultats de l'analyse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Conformité du lot")
                
                # Affichage du nombre de défectueux
                if pourcentage_defectueux <= max_defectueux:
                    st.markdown(
                        f"""
                        <div class="success-box">
                        ✅ Le lot est conforme pour le critère des défectueux.<br>
                        Défectueux: {nombre_defectueux} unités ({pourcentage_defectueux:.2f}%) ≤ {max_defectueux}%
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Le lot n'est pas conforme pour le critère des défectueux.<br>
                        Défectueux: {nombre_defectueux} unités ({pourcentage_defectueux:.2f}%) > {max_defectueux}%
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Affichage de la moyenne du lot
                if net_moy >= qn - (g * sigma_0 / np.sqrt(n)):
                    st.markdown(
                        f"""
                        <div class="success-box">
                        ✅ Le lot est conforme pour le critère de la moyenne.<br>
                        Moyenne: {net_moy:.2f} g ≥ {qn - (g * sigma_0 / np.sqrt(n)):.2f} g
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Le lot n'est pas conforme pour le critère de la moyenne.<br>
                        Moyenne: {net_moy:.2f} g < {qn - (g * sigma_0 / np.sqrt(n)):.2f} g
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("#### Recommandations de production")
                
                # Affichage des valeurs de réglage
                st.metric("Seuil de centrage (ms)", f"{ms:.2f} g", 
                         delta=f"{ms-qn:.2f} g", delta_color="normal",
                         help="Valeur minimale de la moyenne du processus pour respecter le critère des défectueux")
                st.metric("Poids cible (QC)", f"{qc:.2f} g", 
                         delta=f"{qc-qn:.2f} g", delta_color="normal",
                         help="Valeur cible de remplissage incluant le surpoids maximum toléré")
                st.metric("Limite de défectueux (TU1)", f"{tu1:.2f} g", 
                         delta=f"{tu1-qn:.2f} g", delta_color="inverse",
                         help="Limite en dessous de laquelle un pack est considéré comme défectueux")
            
            # Vérification si delta est positif
            st.subheader("🔍 Validation de l'échantillonnage")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Validation de l'échantillonnage
                if delta < 0:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Le calcul de delta est négatif: {delta:.3f}<br>
                        Cela signifie que le processus est déjà dérégulé. Vous devez:<br>
                        1. Augmenter le surpoids maximum toléré, ou<br>
                        2. Réduire l'écart-type du processus
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    delta_sqrt_n = delta * np.sqrt(n)
                    
                    # Vérification des bornes de delta_sqrt_n
                    if delta_sqrt_n < min(TABLE_POM_X) or delta_sqrt_n > max(TABLE_POM_X):
                        st.markdown(
                            f"""
                            <div class="warning-box">
                            ⚠️ Attention: delta_sqrt_n ({delta_sqrt_n:.3f}) est en dehors des limites des tableaux d'interpolation.
                            Les résultats peuvent être moins précis.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Calcul du POM et POl
                    pom = calcul_pom_defectueux(delta_sqrt_n)
                    pol = frequence * 4  # 4 échantillons par heure max
                    
                    # Affichage des résultats
                    st.metric("POM (Période Opérationnelle Moyenne)", f"{pom} échantillons", help="Nombre moyen d'échantillons nécessaires pour détecter un déréglage")
                    st.metric("POl (Période Opérationnelle limite)", f"{pol} échantillons", help="Nombre maximum d'échantillons que vous pouvez contrôler par heure")
                    
                    # Validation de l'échantillonnage
                    if pom <= pol:
                        st.markdown(
                            f"""
                            <div class="success-box">
                            ✅ L'échantillonnage est validé!<br>
                            POM ({pom} échantillons) ≤ POl ({pol} échantillons)
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="error-box">
                            ❌ L'échantillonnage n'est pas suffisant.<br>
                            POM ({pom} échantillons) > POl ({pol} échantillons)<br>
                            Vous devez augmenter l'effectif ou la fréquence d'échantillonnage.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            
            with col2:
                if delta >= 0:
                    # Propositions d'alternatives
                    st.markdown("#### Alternatives d'échantillonnage")
                    
                    # Tableau des alternatives
                    alternatives = []
                    for n_alt in sorted(COEF_G.keys()):
                        delta_sqrt_n_alt = delta * np.sqrt(n_alt)
                        pom_alt = calcul_pom_defectueux(delta_sqrt_n_alt)
                        is_valid = pom_alt <= pol
                        
                        alternatives.append({
                            "Effectif": n_alt,
                            "POM": pom_alt,
                            "Validé": "✅" if is_valid else "❌"
                        })
                    
                    # Créer un DataFrame pour l'affichage
                    df_alternatives = pd.DataFrame(alternatives)
                    
                    # Graphique des alternatives
                    fig = px.bar(df_alternatives, x="Effectif", y="POM", 
                                color="Validé", 
                                title="POM selon l'effectif d'échantillon",
                                color_discrete_map={"✅": "#28a745", "❌": "#dc3545"})
                    
                    # Ajouter une ligne pour POl
                    fig.add_hline(y=pol, line_dash="dash", line_color="black",
                                  annotation_text=f"POl: {pol}")
                    
                    # Mise en forme
                    fig.update_layout(
                        xaxis_title="Effectif d'échantillon",
                        yaxis_title="POM (échantillons)",
                        legend_title="Validé"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Génération du rapport Excel
            st.subheader("💾 Télécharger le rapport d'analyse")
            if st.button("Générer le rapport"):
                # Création du rapport Excel
                output = BytesIO()
                workbook = xlsxwriter.Workbook(output)
                
                # Format pour les titres
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'bg_color': '#4472C4',
                    'font_color': 'white'
                })
                
                # Format pour les en-têtes
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D9E1F2',
                    'border': 1
                })
                
                # Format pour les cellules
                cell_format = workbook.add_format({
                    'border': 1
                })
                
                # Format pour les résultats positifs
                positive_format = workbook.add_format({
                    'border': 1,
                    'bg_color': '#C6EFCE',
                    'font_color': '#006100'
                })
                
                # Format pour les résultats négatifs
                negative_format = workbook.add_format({
                    'border': 1,
                    'bg_color': '#FFC7CE',
                    'font_color': '#9C0006'
                })
                
                # Feuille de résumé
                worksheet_resume = workbook.add_worksheet("Résumé")
                worksheet_resume.set_column('A:A', 30)
                worksheet_resume.set_column('B:B', 20)
                
                # Titre du rapport
                worksheet_resume.merge_range('A1:B1', 'RAPPORT DE VALIDATION D\'ÉCHANTILLONNAGE', title_format)
                worksheet_resume.write('A3', 'Date de l\'analyse:', header_format)
                worksheet_resume.write('B3', pd.Timestamp.now().strftime('%d/%m/%Y'), cell_format)
                
                # Paramètres d'entrée
                worksheet_resume.merge_range('A5:B5', 'PARAMÈTRES D\'ENTRÉE', header_format)
                params = [
                    ["Quantité Nominale (QN)", f"{qn} g"],
                    ["Erreur maximale tolérée (E)", f"{e} g"],
                    ["Surpoids maximum toléré", f"{surpoids_max} g"],
                    ["Effectif d'échantillon (n)", str(n)],
                    ["Fréquence d'échantillonnage", f"{frequence} /heure"]
                ]
                
                row = 6
                for param in params:
                    worksheet_resume.write(row, 0, param[0], cell_format)
                    worksheet_resume.write(row, 1, param[1], cell_format)
                    row += 1
                
                # Résultats d'analyse
                worksheet_resume.merge_range(f'A{row+1}:B{row+1}', 'RÉSULTATS D\'ANALYSE', header_format)
                row += 2
                
                results = [
                    ["Nombre total d'unités", len(poids_nets)],
                    ["Poids net moyen", f"{net_moy:.2f} g"],
                    ["Écart-type du processus", f"{sigma_0:.2f} g"],
                    ["Nombre de défectueux", nombre_defectueux],
                    ["Pourcentage de défectueux", f"{pourcentage_defectueux:.2f}%"],
                    ["Seuil de centrage (ms)", f"{ms:.2f} g"],
                    ["Poids cible (QC)", f"{qc:.2f} g"]
                ]
                
                for result in results:
                    worksheet_resume.write(row, 0, result[0], cell_format)
                    worksheet_resume.write(row, 1, result[1], cell_format)
                    row += 1
                
                # Validation de l'échantillonnage
                worksheet_resume.merge_range(f'A{row+1}:B{row+1}', 'VALIDATION DE L\'ÉCHANTILLONNAGE', header_format)
                row += 2
                
                if delta < 0:
                    worksheet_resume.write(row, 0, "État du processus", cell_format)
                    worksheet_resume.write(row, 1, "Dérégulé (delta < 0)", negative_format)
                else:
                    validation_results = [
                        ["Delta", f"{delta:.3f}"],
                        ["Delta × √n", f"{delta_sqrt_n:.3f}"],
                        ["POM", str(pom)],
                        ["POl", str(pol)]
                    ]
                    
                    for result in validation_results:
                        worksheet_resume.write(row, 0, result[0], cell_format)
                        worksheet_resume.write(row, 1, result[1], cell_format)
                        row += 1
                    
                    worksheet_resume.write(row, 0, "Validation", cell_format)
                    if pom <= pol:
                        worksheet_resume.write(row, 1, "VALIDÉ ✓", positive_format)
                    else:
                        worksheet_resume.write(row, 1, "NON VALIDÉ ✗", negative_format)
                
                # Feuille de données
                worksheet_data = workbook.add_worksheet("Données")
                
                # Ajouter les données brutes
                headers = ["Tare (g)", "Poids Brut (g)", "Poids Net (g)", "Défectueux"]
                for col, header in enumerate(headers):
                    worksheet_data.write(0, col, header, header_format)
                
                for row, (tare, brut, net, defect) in enumerate(zip(
                    df["Tare (g)"],
                    df["Poids Brut (g)"],
                    df["Poids Net (g)"],
                    df["Défectueux"]
                ), start=1):
                    worksheet_data.write(row, 0, tare, cell_format)
                    worksheet_data.write(row, 1, brut, cell_format)
                    worksheet_data.write(row, 2, net, cell_format)
                    
                    if defect:
                        worksheet_data.write(row, 3, "Oui", negative_format)
                    else:
                        worksheet_data.write(row, 3, "Non", positive_format)
                
                # Fermer le workbook et récupérer les données
                workbook.close()
                
                # Télécharger le rapport
                st.download_button(
                    label="Télécharger le rapport Excel",
                    data=output.getvalue(),
                    file_name=f"rapport_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
                
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier Excel: {str(e)}")
            st.stop()
    else:
        st.info("Veuillez télécharger un fichier Excel ou utiliser des données de test pour commencer l'analyse.")

with tabs[1]:  # Onglet Simulation
    st.subheader("🔬 Simulation de contrôle métrologique")
    
    st.markdown("""
    Cette section vous permet de simuler un processus de remplissage et d'observer l'impact des différents 
    paramètres sur la conformité des lots et la validation de l'échantillonnage.
    """)
    
    # Paramètres de simulation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres du processus")
        sim_qn = st.number_input("Quantité Nominale (g):", value=500.0, min_value=1.0, key="sim_qn")
        sim_e = st.number_input("Erreur maximale tolérée (g):", value=float(calculer_emt(sim_qn)), min_value=0.1, key="sim_e")
        sim_sigma = st.slider("Écart-type du processus (g):", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="sim_sigma")
        sim_centrage = st.slider("Centrage du processus (% de QN):", min_value=95.0, max_value=105.0, value=100.0, step=0.1, key="sim_centrage")
    
    with col2:
        st.subheader("Paramètres d'échantillonnage")
        sim_n_options = list(COEF_G.keys())
        sim_n_index = sim_n_options.index(5) if 5 in sim_n_options else 0
        sim_n = st.selectbox("Effectif d'échantillon:", sim_n_options, index=sim_n_index, key="sim_n")
        sim_frequence = st.number_input("Fréquence d'échantillonnage (par heure):", value=1.0, min_value=1.0, key="sim_freq")
        sim_surpoids = st.number_input("Surpoids maximum toléré (g):", value=2.0, min_value=0.0, key="sim_surpoids")
    
    # Bouton pour lancer la simulation
    if st.button("Lancer la simulation"):
        # Paramètres de simulation
        tare_moyenne = 15.0
        tare_sigma = 0.5
        
        # Générer les données simulées
        moyenne_reelle = sim_qn * (sim_centrage / 100)
        
        with st.spinner("Simulation en cours..."):
            # Simuler un léger délai pour donner l'impression de calcul
            time.sleep(1)
            
            # Création des données simulées
            sim_df = generer_donnees_test(moyenne_reelle, sim_sigma, tare_moyenne, tare_sigma, n_samples=100)
            
            # Calcul des poids nets
            sim_df["Poids Net (g)"] = sim_df["Poids Brut (g)"] - sim_df["Tare (g)"]
            
            # Identification des défectueux
            sim_df["Défectueux"] = sim_df["Poids Net (g)"] < (sim_qn - sim_e)
            
            # Affichage des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Résultats de la simulation")
                
                # Statistiques de base
                net_moy = np.mean(sim_df["Poids Net (g)"])
                net_std = np.std(sim_df["Poids Net (g)"])
                nombre_defectueux = sim_df["Défectueux"].sum()
                pourcentage_defectueux = (nombre_defectueux / len(sim_df)) * 100
                
                # Affichage des métriques
                st.metric("Poids net moyen", f"{net_moy:.2f} g", delta=f"{net_moy-sim_qn:.2f} g")
                st.metric("Écart-type réel", f"{net_std:.2f} g", delta=f"{net_std-sim_sigma:.2f} g" if abs(net_std-sim_sigma) > 0.01 else None)
                st.metric("Nombre de défectueux", f"{nombre_defectueux} unités")
                st.metric("Pourcentage de défectueux", f"{pourcentage_defectueux:.2f}%")
                
                # Calcul de conformité
                g = COEF_G[sim_n]
                seuil_moyenne = sim_qn - (g * net_std / np.sqrt(sim_n))
                
                # Affichage de la conformité
                st.subheader("Conformité du lot")
                
                # Critère des défectueux
                if pourcentage_defectueux <= 2.5:
                    st.markdown(
                        f"""
                        <div class="success-box">
                        ✅ Critère des défectueux : CONFORME<br>
                        {pourcentage_defectueux:.2f}% ≤ 2.5%
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Critère des défectueux : NON CONFORME<br>
                        {pourcentage_defectueux:.2f}% > 2.5%
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Critère de la moyenne
                if net_moy >= seuil_moyenne:
                    st.markdown(
                        f"""
                        <div class="success-box">
                        ✅ Critère de la moyenne : CONFORME<br>
                        {net_moy:.2f} g ≥ {seuil_moyenne:.2f} g
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Critère de la moyenne : NON CONFORME<br>
                        {net_moy:.2f} g < {seuil_moyenne:.2f} g
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Visualisation des données
                fig = px.histogram(sim_df, x="Poids Net (g)", title="Distribution des poids nets simulés",
                                  nbins=30, color_discrete_sequence=["#FF9900"])
                
                # Ajouter des lignes verticales
                fig.add_vline(x=sim_qn, line_dash="dash", line_color="green",
                            annotation_text=f"QN: {sim_qn} g",
                            annotation_position="top right")
                fig.add_vline(x=sim_qn-sim_e, line_dash="dash", line_color="red",
                            annotation_text=f"TU1: {sim_qn-sim_e} g",
                            annotation_position="top left")
                fig.add_vline(x=net_moy, line_dash="solid", line_color="blue",
                            annotation_text=f"Moyenne: {net_moy:.2f} g",
                            annotation_position="bottom right")
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse de l'échantillonnage
                sigma_0 = net_std
                ms = sim_qn if sigma_0 <= sim_e/2.05 else sim_qn - sim_e + 2.05 * sigma_0
                qc = ms + sim_surpoids
                m2 = sim_qn - sim_e + 2.05 * sigma_0
                delta = (qc - m2) / sigma_0
                
                # Validation de l'échantillonnage
                st.subheader("Validation de l'échantillonnage")
                
                if delta < 0:
                    st.markdown(
                        f"""
                        <div class="error-box">
                        ❌ Le processus est dérégulé (delta = {delta:.3f})<br>
                        Vous devez augmenter le surpoids ou réduire l'écart-type.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    delta_sqrt_n = delta * np.sqrt(sim_n)
                    pom = calcul_pom_defectueux(delta_sqrt_n)
                    pol = sim_frequence * 4
                    
                    # Affichage des résultats
                    st.metric("POM", f"{pom} échantillons")
                    st.metric("POl", f"{pol} échantillons")
                    
                    if pom <= pol:
                        st.markdown(
                            f"""
                            <div class="success-box">
                            ✅ L'échantillonnage est validé!<br>
                            POM ({pom}) ≤ POl ({pol})
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="error-box">
                            ❌ L'échantillonnage n'est pas suffisant.<br>
                            POM ({pom}) > POl ({pol})
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            
            # Tableau d'échantillonnage
            st.subheader("Table d'échantillonnage")
            
            # Créer un tableau pour différentes combinaisons
            table_data = []
            
            for n_sim in sorted(COEF_G.keys()):
                for freq_sim in [1, 2, 3, 4]:
                    if delta >= 0:
                        delta_sqrt_n_sim = delta * np.sqrt(n_sim)
                        pom_sim = calcul_pom_defectueux(delta_sqrt_n_sim)
                        pol_sim = freq_sim * 4
                        is_valid = pom_sim <= pol_sim
                        
                        table_data.append({
                            "Effectif (n)": n_sim,
                            "Fréquence (/h)": freq_sim,
                            "POM": pom_sim,
                            "POl": pol_sim,
                            "Validé": "✅" if is_valid else "❌"
                        })
            
            # Créer le DataFrame
            sim_table_df = pd.DataFrame(table_data)
            
            # Afficher le tableau avec formatage conditionnel
            st.dataframe(
                sim_table_df,
                column_config={
                    "Validé": st.column_config.TextColumn(
                        "Validé",
                        help="Indique si l'échantillonnage est validé"
                    )
                },
                use_container_width=True,
                hide_index=True
            )

with tabs[2]:  # Onglet Documentation
    st.subheader("📚 Documentation et références")
    
    st.markdown("""
    ### Guide de bonnes pratiques pour le contrôle métrologique des préemballages
    
    Cette application est basée sur le guide des bonnes pratiques pour le contrôle métrologique des préemballages
    publié par la DGCCRF (Direction Générale de la Concurrence, de la Consommation et de la Répression des Fraudes).
    
    #### Concepts clés
    
    * **Quantité Nominale (QN)** : Quantité indiquée sur l'emballage du produit.
    * **Erreur Maximale Tolérée (EMT ou E)** : Écart maximal admissible entre le poids réel et le poids nominal.
    * **Défectueux** : Unité dont le contenu est inférieur à QN - E.
    * **Critère des défectueux** : Au maximum 2,5% des préemballages peuvent être défectueux.
    * **Critère de la moyenne** : La moyenne du contenu réel des préemballages doit être au moins égale à la quantité nominale.
    
    #### Formules importantes
    
    * **Seuil de centrage (ms)** : 
      - Si σ₀ ≤ E/2.05, alors ms = QN
      - Si σ₀ > E/2.05, alors ms = QN - E + 2.05 × σ₀
    
    * **Poids cible (QC)** : QC = ms + surpoids
    
    * **Critère de la moyenne** : x̄ ≥ QN - (g × σ₀/√n)
      - où g est un coefficient statistique dépendant de la taille d'échantillon
    
    * **Delta** : δ = (QC - m2) / σ₀
      - où m2 = QN - E + 2.05 × σ₀
    
    * **POM** : Période Opérationnelle Moyenne (nombre d'échantillons avant détection d'un déréglage)
    
    * **POl** : Période Opérationnelle limite (fréquence × 4)
    
    #### Tableaux de référence
    
    **Tableau des coefficients g** :
    
    | Taille d'échantillon (n) | Coefficient g |
    |--------------------------|---------------|
    | 5                        | 0.686         |
    | 7                        | 0.787         |
    | 10                       | 0.902         |
    | 15                       | 1.047         |
    | 20                       | 1.140         |
    | 30                       | 1.264         |
    | 50                       | 1.402         |
    
    **Erreurs Maximales Tolérées (EMT)** :
    
    | Quantité nominale (QN) en grammes | EMT (E) |
    |------------------------------------|---------|
    | QN < 50                           | 9% de QN |
    | 50 ≤ QN < 100                     | 4.5 g    |
    | 100 ≤ QN < 200                    | 4.5% de QN |
    | 200 ≤ QN < 300                    | 9 g      |
    | 300 ≤ QN < 500                    | 3% de QN |
    | 500 ≤ QN < 1000                   | 15 g     |
    | 1000 ≤ QN < 10000                 | 1.5% de QN |
    | QN ≥ 10000                        | 150 g    |
    """)
    
    with st.expander("Qu'est-ce que le POM et le POl?"):
        st.markdown("""
        ### POM et POl
        
        **POM (Période Opérationnelle Moyenne)** : C'est le nombre moyen d'échantillons nécessaires pour détecter 
        un déréglage du processus. Plus cette valeur est faible, plus le plan d'échantillonnage est efficace pour détecter 
        rapidement les déréglages.
        
        **POl (Période Opérationnelle limite)** : C'est le nombre maximal d'échantillons que vous pouvez contrôler 
        pendant une durée déterminée (généralement 1 heure). Cette valeur est calculée comme le produit de la fréquence 
        d'échantillonnage par un facteur (généralement 4).
        
        **Validation de l'échantillonnage** : Pour qu'un plan d'échantillonnage soit considéré comme valide, 
        il faut que POM ≤ POl. Si ce n'est pas le cas, vous devez soit :
        - Augmenter l'effectif d'échantillon (n)
        - Augmenter la fréquence d'échantillonnage
        - Diminuer l'écart-type du processus
        - Augmenter le surpoids toléré
        """)
    
    with st.expander("Comment interpréter les résultats?"):
        st.markdown("""
        ### Interprétation des résultats
        
        #### Conformité du lot
        
        Un lot est considéré conforme s'il respecte deux critères :
        
        1. **Critère des défectueux** : le pourcentage d'unités défectueuses (poids < QN-E) ne doit pas dépasser 2,5%.
        
        2. **Critère de la moyenne** : la moyenne des poids nets doit être supérieure ou égale à QN - (g × σ₀/√n).
        
        #### Validation de l'échantillonnage
        
        L'échantillonnage est validé si POM ≤ POl, c'est-à-dire si le nombre moyen d'échantillons nécessaires pour 
        détecter un déréglage est inférieur ou égal au nombre maximal d'échantillons que vous pouvez contrôler.
        
        #### Recommandations de production
        
        - **Seuil de centrage (ms)** : C'est la valeur minimale que doit avoir la moyenne du processus pour respecter 
          le critère des défectueux.
        
        - **Poids cible (QC)** : C'est la valeur cible de remplissage qui inclut le surpoids maximum toléré.
        
        - **Delta (δ)** : Si delta est négatif, cela signifie que le processus est déjà dérégulé. Dans ce cas, 
          vous devez soit augmenter le surpoids maximum toléré, soit réduire l'écart-type du processus.
        """)
    
    with st.expander("Conseils pour améliorer votre processus"):
        st.markdown("""
        ### Conseils d'amélioration
        
        #### Si votre lot n'est pas conforme
        
        1. **Trop de défectueux** :
           - Augmentez le centrage du processus (moyenne)
           - Réduisez la variabilité du processus (écart-type)
           - Vérifiez l'étalonnage de vos équipements
        
        2. **Moyenne trop faible** :
           - Augmentez le centrage du processus
           - Vérifiez l'étalonnage de vos équipements
        
        #### Si votre échantillonnage n'est pas validé
        
        1. **POM > POl** :
           - Augmentez l'effectif d'échantillon (n)
           - Augmentez la fréquence d'échantillonnage
           - Réduisez l'écart-type du processus
           - Augmentez le surpoids toléré
        
        2. **Delta négatif** :
           - Augmentez le surpoids maximum toléré
           - Réduisez l'écart-type du processus
        
        #### Pour réduire les coûts
        
        1. **Réduire le surpoids** :
           - Réduisez l'écart-type du processus (meilleure maîtrise)
           - Utilisez des équipements plus précis
        
        2. **Optimiser l'échantillonnage** :
           - Utilisez un effectif d'échantillon plus grand mais une fréquence plus faible
        """)

# ----- FONCTIONS DE TÉLÉCHARGEMENT -----

# Fonction pour créer un fichier Excel modèle
def creer_fichier_excel_modele():
    """
    Crée un fichier Excel modèle avec les colonnes nécessaires pour l'analyse
    et quelques exemples de données.
    """
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output)
    
    # Format pour les en-têtes
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    # Format pour les cellules
    cell_format = workbook.add_format({
        'border': 1
    })
    
    # Format pour les instructions
    instruction_format = workbook.add_format({
        'italic': True,
        'font_color': '#666666'
    })
    
    # Créer la feuille de pesées
    worksheet = workbook.add_worksheet("Pesées")
    
    # Définir la largeur des colonnes
    worksheet.set_column('A:B', 15)
    worksheet.set_column('C:C', 30)
    
    # Ajouter les en-têtes
    worksheet.write('A1', 'Tare (g)', header_format)
    worksheet.write('B1', 'Poids Brut (g)', header_format)
    worksheet.write('C1', 'Notes (optionnel)', header_format)
    
    # Ajouter des exemples de données (5 lignes)
    exemple_tares = [15.2, 15.3, 15.1, 15.4, 15.2]
    exemple_bruts = [515.6, 516.2, 514.8, 515.9, 516.3]
    
    for i, (tare, brut) in enumerate(zip(exemple_tares, exemple_bruts), start=2):
        worksheet.write(f'A{i}', tare, cell_format)
        worksheet.write(f'B{i}', brut, cell_format)
        worksheet.write(f'C{i}', "", cell_format)
    
    # Ajouter des instructions
    worksheet.merge_range('A8:C8', 'Instructions d\'utilisation:', instruction_format)
    instructions = [
        "1. Saisissez les poids des emballages vides dans la colonne 'Tare (g)'.",
        "2. Saisissez les poids bruts des packs remplis dans la colonne 'Poids Brut (g)'.",
        "3. La colonne 'Notes' est optionnelle et peut être utilisée pour des commentaires.",
        "4. Vous pouvez supprimer les exemples et ajouter autant de lignes que nécessaire.",
        "5. Assurez-vous de conserver le nom de la feuille 'Pesées' et les en-têtes des colonnes."
    ]
    
    for i, instruction in enumerate(instructions, start=9):
        worksheet.merge_range(f'A{i}:C{i}', instruction, instruction_format)
    
    # Fermer le workbook
    workbook.close()
    
    return output.getvalue()

with tabs[3]:  # Onglet Téléchargements
    st.subheader("📥 Téléchargement des modèles et ressources")
    
    st.markdown("""
    Cette section vous permet de télécharger des fichiers modèles et des ressources pour vous aider dans votre contrôle métrologique.
    """)
    
    # Section pour le téléchargement du modèle Excel
    st.markdown("### Fichier Excel modèle pour les pesées")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Ce fichier Excel modèle contient les colonnes nécessaires pour l'analyse de vos pesées :
        
        - **Tare (g)** : Poids des emballages vides
        - **Poids Brut (g)** : Poids total du produit emballé
        - **Notes** : Champ optionnel pour vos commentaires
        
        Le fichier inclut également quelques exemples de données et des instructions d'utilisation.
        """)
    
    with col2:
        # Image d'aperçu du fichier Excel
        st.markdown("""
        <div style="border:1px solid #ddd; padding:10px; text-align:center;">
            <h4>Aperçu du modèle</h4>
            <div style="background-color:#f0f0f0; padding:10px; font-family:monospace; font-size:12px; text-align:left;">
                <b>Tare (g) | Poids Brut (g) | Notes</b><br>
                15.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 515.6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
                15.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 516.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
                15.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 514.8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
                ...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton pour télécharger le modèle
    excel_file = creer_fichier_excel_modele()
    st.download_button(
        label="📥 Télécharger le modèle Excel",
        data=excel_file,
        file_name="modele_pesees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Téléchargez ce fichier Excel et remplissez-le avec vos données de pesée."
    )
    
    # Instructions étape par étape
    with st.expander("Comment utiliser ce fichier Excel"):
        st.markdown("""
        ### Instructions d'utilisation pas à pas
        
        1. **Téléchargement du fichier modèle** :
           - Cliquez sur le bouton "Télécharger le modèle Excel" ci-dessus
           - Enregistrez le fichier sur votre ordinateur
        
        2. **Saisie de vos données** :
           - Ouvrez le fichier avec Microsoft Excel, LibreOffice Calc ou un logiciel équivalent
           - Supprimez les exemples de données si nécessaire
           - Saisissez les poids des emballages vides dans la colonne 'Tare (g)'
           - Saisissez les poids bruts des packs remplis dans la colonne 'Poids Brut (g)'
           - Ajoutez des notes dans la dernière colonne si nécessaire
        
        3. **Enregistrement du fichier complété** :
           - Enregistrez le fichier Excel complété
           - Assurez-vous de conserver le nom de la feuille 'Pesées' et les en-têtes des colonnes
        
        4. **Téléchargement dans l'application** :
           - Retournez dans l'onglet "Analyse" de cette application
           - Téléchargez votre fichier Excel complété via le sélecteur de fichier
           - L'application analysera automatiquement vos données
        """)
    
    # Section pour les ressources documentaires
    st.markdown("### Ressources documentaires")
    
    # Guide DGCCRF
    st.markdown("""
    #### Guide de bonnes pratiques DGCCRF
    
    Le guide de bonnes pratiques pour le contrôle métrologique des préemballages publié par la DGCCRF
    contient toutes les informations réglementaires et techniques nécessaires pour effectuer un contrôle
    métrologique conforme à la législation française et européenne.
    
    [Consulter le guide sur le site de la DGCCRF](https://www.economie.gouv.fr/files/files/directions_services/dgccrf/boccrf/2014/14_06/guide_autocontrole_metrologique.pdf)
    """)
    
    # Fiches techniques
    st.markdown("""
    #### Fiches techniques
    
    Ces fiches techniques résument les points essentiels du contrôle métrologique et peuvent être
    utilisées comme aide-mémoire dans votre processus de contrôle qualité.
    """)
    
    # Liste des fiches techniques (fictives pour l'instant)
    fiches = [
        {"titre": "Calcul de l'erreur maximale tolérée (EMT)", "description": "Comment calculer l'EMT selon la quantité nominale"},
        {"titre": "Interprétation des résultats statistiques", "description": "Guide d'interprétation des résultats d'analyse"},
        {"titre": "Optimisation du plan d'échantillonnage", "description": "Stratégies pour optimiser votre plan d'échantillonnage"}
    ]
    
    # Affichage des fiches techniques
    for i, fiche in enumerate(fiches):
        st.markdown(f"""
        <div style="border:1px solid #ddd; padding:10px; margin-bottom:10px; border-radius:5px;">
            <h5>{fiche['titre']}</h5>
            <p>{fiche['description']}</p>
            <button disabled style="background-color:#e0e0e0; padding:5px 10px; border:none; border-radius:3px;">
                Bientôt disponible
            </button>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("Développé pour le contrôle métrologique des préemballages | Version 2.0")
