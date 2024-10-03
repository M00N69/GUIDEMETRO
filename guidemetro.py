import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from io import BytesIO

# Fonction pour le test de normalité de Shapiro-Wilk
def shapiro_test(data):
    stat, p = stats.shapiro(data)
    alpha = 0.05
    if p > alpha:
        return "Distribution normale (p = {:.3f})".format(p)
    else:
        return "Distribution non normale (p = {:.3f})".format(p)

# Fonction pour calculer le POM à partir de l'annexe 3 du guide DGCCRF
def calcul_pom(delta_sqrt_n):
    table_pom_x = [0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.94, 1.06, 1.17, 1.28, 1.39, 1.50, 1.61, 1.72, 1.83, 1.94, 2.05, 2.17, 2.28, 2.39, 2.50, 2.61, 2.72, 2.83, 2.94, 3.05, 3.16, 3.27, 3.38]
    table_pom_y = [328, 108, 66, 44, 30, 23, 18, 15, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    f = interp1d(table_pom_x, table_pom_y, kind='linear', fill_value="extrapolate")
    if isinstance(delta_sqrt_n, (np.ndarray, np.float64)):
        delta_sqrt_n = float(delta_sqrt_n)
    return round(float(f(delta_sqrt_n)))

# Fonction pour calculer le POM pour les défauts
def calcul_pom_defectueux(delta_sqrt_n):
    table_pom_x = [0.184, 0.175, 0.167, 0.161, 0.155, 0.149, 0.144, 0.140, 0.136, 0.132, 0.129, 0.126, 0.123, 0.120, 0.118, 0.115, 0.113, 0.111, 0.109, 0.107, 0.105, 0.103, 0.102, 0.100, 0.099, 0.097, 0.096, 0.095, 0.093, 0.092, 0.091, 0.090, 0.089, 0.088, 0.087, 0.086, 0.085, 0.084, 0.083, 0.082, 0.081, 0.080, 0.080, 0.079, 0.078, 0.077, 0.077, 0.076, 0.075, 0.075, 0.074, 0.074]
    table_pom_y = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 300]
    f = interp1d(table_pom_x, table_pom_y, kind='linear', fill_value="extrapolate")
    if isinstance(delta_sqrt_n, (np.ndarray, np.float64)):
        delta_sqrt_n = float(delta_sqrt_n)
    return round(float(f(delta_sqrt_n)))

# Interface Streamlit
st.title("Validation d'échantillonnage pour le contrôle métrologique")

# Expander pour les explications détaillées
with st.expander("Explication : Comment utiliser cette application"):
    st.markdown("""
    ### Validation d'échantillonnage pour le contrôle métrologique
    
    Cette application vous aide à valider votre stratégie d'échantillonnage pour le contrôle métrologique de vos préemballages, conformément au guide de bonnes pratiques de la DGCCRF (Direction générale de la concurrence, de la consommation et de la répression des fraudes).
    
    #### Instructions pour la création du fichier Excel :
    - Créer un fichier Excel nommé `pesées.xlsx`.
    - Ajouter une feuille nommée `Pesées`.
    - Créer deux colonnes : `Tare (g)` et `Poids Brut (g)`.
    - Saisir les poids des emballages vides dans la colonne `Tare (g)`.
    - Saisir les poids bruts des packs remplis dans la colonne `Poids Brut (g)`.
    - Enregistrer le fichier Excel.
    
    #### Utilisation de l'application :
    1. Lancer l'application Streamlit.
    2. Saisir les hypothèses de travail suivantes :
       - **Quantité Nominale (g)** : poids nominal indiqué sur l'emballage.
       - **Erreur maximale tolérée (g)** : valeur de E selon l'annexe 5 du guide DGCCRF.
       - **Surpoids maximum toléré (g)** : surpoids maximal acceptable pour chaque pack.
       - **Effectif d'échantillon** : nombre de packs à peser dans chaque échantillon.
       - **Fréquence d'échantillonnage (par heure)** : nombre d'échantillons à contrôler par heure.
    3. Télécharger le fichier Excel contenant les poids des emballages vides et les poids bruts des packs.
    4. L'application affichera les résultats de l'analyse de la tare et du poids brut, ainsi que les critères de conformité calculés.
    5. L'application vérifiera si l'échantillonnage est validé ou non en comparant le POM (Période Opérationnelle Moyenne) et le POl (Période Opérationnelle Limite).
    6. Si l'échantillonnage n'est pas suffisant, l'application proposera des alternatives d'échantillonnage pour atteindre un POM acceptable.
    7. L'application générera un rapport Excel contenant les résultats de l'analyse, que vous pourrez télécharger.
    
    #### Fonctionnement de l'application :
    L'application utilise les bibliothèques Pandas, NumPy et SciPy pour effectuer des calculs statistiques et d'interpolation. Elle définit deux fonctions :
    
    - **shapiro_test** : réalise le test de normalité de Shapiro-Wilk sur les données fournies.
    - **calcul_pom** : calcule le POM (Période Opérationnelle Moyenne) à partir de la valeur de δ√n en utilisant une interpolation linéaire sur le tableau de l'Annexe 3 du guide DGCCRF.
    - **calcul_pom_defectueux** : calcule le POM (Période Opérationnelle Moyenne) à partir de la valeur de δ√n en utilisant une interpolation linéaire sur le tableau de l'Annexe 4 du guide DGCCRF, pour le critère des défectueux.
    
    #### Étapes de l'application :
    1. Saisie des hypothèses de travail par l'utilisateur.
    2. Upload du fichier Excel contenant les données de pesée.
    3. Analyse de la tare et du poids brut des données.
    4. Calcul des critères de conformité.
    5. Comparaison du POM et du POl pour valider l'échantillonnage.
    6. Génération d'un rapport Excel contenant les résultats de l'analyse.
    """)

# Lien pour télécharger le fichier d'exemple depuis Google Drive (lien direct)
st.markdown("Téléchargez le fichier d'exemple ici : [pesées.xlsx](https://drive.google.com/uc?export=download&id=1V-hd1YUOi512gwVJIKUjpoQ4OURzVIdM)")

# Hypothèses de travail
st.header("Saisie des hypothèses de travail")
qn = st.number_input("Quantité Nominale (g):", value=500, help="Poids nominal indiqué sur l'emballage.")
e = st.number_input("Erreur maximale tolérée (g):", value=15, help="Valeur de E selon l'annexe 5 du guide DGCCRF.")
surpoids_max = st.number_input("Surpoids maximum toléré (g):", value=2, help="Surpoids maximal acceptable pour chaque pack.")
n = st.number_input("Effectif d'échantillon:", value=5, help="Nombre de packs à peser dans chaque échantillon.")
frequence = st.number_input("Fréquence d'échantillonnage (par heure):", value=1, help="Nombre d'échantillons à contrôler par heure.")

# Téléchargement du fichier Excel
st.header("Téléchargement du fichier Excel de pesée")
uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Pesées")
        tare_poids = df["Tare (g)"].dropna().tolist()
        poids_bruts = df["Poids Brut (g)"].dropna().tolist()
    except Exception as e:
        st.error("Erreur lors du chargement du fichier Excel. Vérifiez le format du fichier.")
        st.write(e)
        st.stop()

    # Analyse de la tare
    st.subheader("Analyse de la Tare (g)")
    st.write(f"Nombre d'emballages : {len(tare_poids)}")
    st.write(f"Moyenne de la tare : {np.mean(tare_poids):.2f} g")
    st.write(f"Écart-type de la tare : {np.std(tare_poids):.2f} g")
    st.write(f"Test de normalité (Shapiro-Wilk) : {shapiro_test(tare_poids)}")

    # Analyse du poids brut
    st.subheader("Analyse du Poids Brut (g)")
    st.write(f"Nombre de packs : {len(poids_bruts)}")
    st.write(f"Moyenne du poids brut : {np.mean(poids_bruts):.2f} g")
    st.write(f"Écart-type du processus (σ₀) : {np.std(poids_bruts):.2f} g")

    # Calcul des critères de conformité
    sigma_0 = np.std(poids_bruts)
    ms = qn if sigma_0 <= e/2.05 else qn - e + 2.05 * sigma_0
    qc = ms + surpoids_max
    g = 0.686  # Coefficient pour n=5
    tu1 = qn - e

    # Comparaison POM vs POl
    m2 = qn - e + 2.05 * sigma_0
    delta = (qc - m2) / sigma_0
    delta_sqrt_n = delta * np.sqrt(n)
    pom = calcul_pom_defectueux(delta_sqrt_n)
    pol = frequence * 4  # 4 échantillons par heure max

    st.subheader("Validation de l'échantillonnage")
    st.write(f"Seuil de centrage (ms) : {ms:.2f} g")
    st.write(f"Poids cible (QC) : {qc:.2f} g")
    st.write(f"POM : {pom} échantillons")
    st.write(f"POl : {pol} échantillons")

    if pom <= pol:
        st.success("L'échantillonnage est validé!")
    else:
        st.error("L'échantillonnage n'est pas suffisant. Augmentez l'effectif ou la fréquence.")

    # Propositions d'échantillonnages alternatifs
    st.subheader("Propositions d'échantillonnages alternatifs")
    for n_alt in [7, 10, 15]:
        delta_sqrt_n_alt = delta * np.sqrt(n_alt)
        pom_alt = calcul_pom_defectueux(delta_sqrt_n_alt)
        st.write(f"Effectif (n) = {n_alt}, POM = {pom_alt} échantillons")

    # Génération du rapport Excel
    st.subheader("Télécharger le rapport d'analyse")
    if st.button("Générer le rapport"):
        df_resultats = pd.DataFrame({
            "Paramètre": ["Quantité Nominale (QN)", "Erreur maximale tolérée (E)", "Surpoids max", "Effectif d'échantillon (n)",
                          "Fréquence d'échantillonnage (par heure)", "Écart-type du processus (σ₀)", "Seuil de centrage (ms)",
                          "Poids cible (QC)", "Coefficient statistique (g)", "Limite de défectueux (TU1)", "Nouvelle valeur cible (m2)",
                          "δ", "δ√n", "POM", "POl"],
            "Valeur": [qn, e, surpoids_max, n, frequence, sigma_0, ms, qc, g, tu1, m2, delta, delta_sqrt_n, pom, pol],
            "Description": ["Poids nominal indiqué sur l'emballage.", "Valeur de E selon l'annexe 5 du guide DGCCRF.",
                            "Surpoids maximal acceptable pour chaque pack.", "Nombre de packs à peser dans chaque échantillon.",
                            "Nombre d'échantillons à contrôler par heure.", "Écart-type des poids bruts mesurés.",
                            "Valeur minimale de la moyenne du processus pour respecter le critère des défectueux.",
                            "Valeur cible de remplissage, incluant le surpoids maximum toléré.",
                            "Coefficient statistique pour un seuil de confiance de 90%.",
                            "Limite en dessous de laquelle un pack est considéré comme défectueux.",
                            "Valeur cible après un déréglage.",
                            "Déviation de la moyenne de l'échantillon par rapport à m2.",
                            "δ multiplié par la racine carrée de l'effectif d'échantillon.",
                            "Nombre moyen d'échantillons nécessaires pour détecter un déréglage.",
                            "Nombre maximum d'échantillons que vous pouvez contrôler par heure."]
        })

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df_resultats.to_excel(writer, sheet_name='Résultats', index=False)
        writer.close()

        st.download_button(
            label="Télécharger le rapport Excel",
            data=output.getvalue(),
            file_name="rapport_validation.xlsx",
            mime="application/vnd.ms-excel"
        )

else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
