import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from io import BytesIO

# Fonction pour le test de normalité de Shapiro-Wilk
def shapiro_test(data):
    """
    Réalise le test de normalité de Shapiro-Wilk sur les données fournies.

    Args:
        data (list): Une liste de valeurs numériques représentant les poids de tare ou les poids bruts.

    Returns:
        str: Une chaîne de caractères indiquant si la distribution est normale ou non, avec la valeur p.
    """
    stat, p = stats.shapiro(data)
    alpha = 0.05  # Seuil de signification
    if p > alpha:
        return "Distribution normale (p = {:.3f})".format(p)
    else:
        return "Distribution non normale (p = {:.3f})".format(p)

# Fonction pour calculer le POM à partir de l'annexe 3 du guide
def calcul_pom(delta_sqrt_n):
    """
    Calcule le POM (Période Opérationnelle Moyenne) à partir de la valeur de δ√n en utilisant
    une interpolation linéaire sur le tableau de l'Annexe 3 du guide DGCCRF.

    Args:
        delta_sqrt_n (float): La valeur de δ (déviation) multipliée par la racine carrée de n (effectif d'échantillon).

    Returns:
        int: La valeur du POM arrondie à l'entier le plus proche.
    """
    # Tableau de l'Annexe 3 du guide DGCCRF (valeurs réelles)
    table_pom_x = [0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.94, 1.06, 1.17, 1.28, 1.39, 1.50, 1.61, 1.72, 1.83, 1.94, 2.05, 2.17, 2.28, 2.39, 2.50, 2.61, 2.72, 2.83, 2.94, 3.05, 3.16, 3.27, 3.38]
    table_pom_y = [328, 108, 66, 44, 30, 23, 18, 15, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]

    # Interpolation linéaire pour les valeurs manquantes
    f = interp1d(table_pom_x, table_pom_y, kind='linear', fill_value="extrapolate")

    # S'assurer que delta_sqrt_n est un nombre simple (int ou float), pas un tableau ni float64
    if isinstance(delta_sqrt_n, (np.ndarray, np.float64)):
        delta_sqrt_n = float(delta_sqrt_n)

    pom = f(delta_sqrt_n)
    return round(pom)

# Interface Streamlit
st.title("Validation d'échantillonnage pour le contrôle métrologique")

# Introduction
st.write(
    """
    Cet outil vous aide à valider votre stratégie d'échantillonnage pour le contrôle 
    métrologique de vos préemballages, conformément au guide de bonnes pratiques 
    de la DGCCRF (Direction générale de la concurrence, de la consommation et de la 
    répression des fraudes).  
    """
)

# Instructions pour la création du fichier Excel
st.header("Instructions pour la création du fichier Excel")
st.write(
    """
    1. Créer un fichier Excel nommé `pesées.xlsx`.
    2. Ajouter une feuille nommée `Pesées`.
    3. Créer deux colonnes : `Tare (g)` et `Poids Brut (g)`.
    4. Saisir les poids des emballages vides dans la colonne `Tare (g)`.
    5. Saisir les poids bruts des packs remplis dans la colonne `Poids Brut (g)`.
    6. Enregistrer le fichier Excel.
    """
)

# Saisie des hypothèses de travail
st.header("Hypothèses de travail")
qn = st.number_input("Quantité Nominale (g):", value=1000, help="Poids nominal indiqué sur l'emballage.")
e = st.number_input("Erreur maximale tolérée (g):", value=15, help="Valeur de E selon l'annexe 5 du guide DGCCRF.")
surpoids_max = st.number_input("Surpoids maximum toléré (g):", value=2, help="Surpoids maximal acceptable pour chaque pack.")
n = st.number_input("Effectif d'échantillon:", value=5, help="Nombre de packs à peser dans chaque échantillon.")
frequence = st.number_input("Fréquence d'échantillonnage (par heure):", value=1, help="Nombre d'échantillons à contrôler par heure.")

# Upload du fichier Excel
st.header("Données de pesée")
st.write("Téléchargez le fichier Excel contenant les poids des emballages vides et les poids bruts des packs.")
uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx"])

# Vérification du format du fichier Excel
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Pesées")
        tare_poids = df["Tare (g)"].dropna().tolist()
        poids_bruts = df["Poids Brut (g)"].dropna().tolist()
    except Exception as e:
        st.error("Erreur lors du chargement du fichier Excel. Veuillez vérifier le format du fichier.")
        st.write(e)
        st.stop()

    # Analyse de la tare
    st.subheader("Résultats de l'analyse de la tare:")
    st.write("Nombre d'emballages:", len(tare_poids))
    st.write("Moyenne de la tare:", np.mean(tare_poids))
    st.write("Écart-type de la tare:", np.std(tare_poids))
    st.write("Test de normalité (Shapiro-Wilk):", shapiro_test(tare_poids))

    # Analyse du poids brut
    st.subheader("Résultats de l'analyse du poids brut:")
    st.write("Nombre de packs:", len(poids_bruts))
    st.write("Moyenne du poids brut:", np.mean(poids_bruts))
    st.write("Écart-type du processus (σ₀):", np.std(poids_bruts))

    # Calcul des critères de conformité
    sigma_0 = np.std(poids_bruts)
    ms = qn if sigma_0 <= e/2.05 else qn - e + 2.05 * sigma_0
    qc = ms + surpoids_max
    g = 0.686  # Pour n = 5, à adapter selon l'annexe 3 du guide
    tu1 = qn - e

    # Comparaison POM vs POl
    m1 = qn - 0.002 * qn  # Exemple de déréglage de 0.2%
    delta = (qc - m1) / sigma_0
    delta_sqrt_n = delta * np.sqrt(n)
    pom = calcul_pom(delta_sqrt_n)
    pol = frequence * 4  # 4 échantillons par heure maximum

    st.header("Validation de l'échantillonnage")
    st.write("**Seuil de centrage (ms):**", ms, "g.  C'est la valeur minimale de la moyenne du processus pour respecter le critère des défectueux.")
    st.write("**Poids cible (QC):**", qc, "g.  C'est la valeur cible de remplissage, incluant le surpoids maximum toléré.")
    st.write("**POM:**", pom, "échantillons. C'est le nombre moyen d'échantillons nécessaires pour détecter un déréglage de 0.2% de la quantité nominale.")
    st.write("**POl:**", pol, "échantillons. C'est le nombre maximum d'échantillons que vous pouvez contrôler par heure.")

    if pom <= pol:
        st.success("L'échantillonnage est validé!")
    else:
        st.error("L'échantillonnage n'est pas suffisant.")
        st.write("Augmentez l'effectif d'échantillon (n) ou la fréquence d'échantillonnage pour réduire le POM.")

    # Propositions d'échantillonnages
    st.header("Propositions d'échantillonnages")
    st.write("Voici quelques alternatives d'échantillonnage pour atteindre un POM acceptable:")
    for n_alt in [7, 10, 15]:
        delta_sqrt_n_alt = delta * np.sqrt(n_alt)
        pom_alt = calcul_pom(delta_sqrt_n_alt)
        st.write("- Effectif (n) = {}, POM = {} échantillons".format(n_alt, pom_alt))

    # Génération du rapport
    st.header("Télécharger le rapport")

    if st.button("Générer le rapport"):
        # Créer un DataFrame Pandas avec les résultats
        df_resultats = pd.DataFrame({
            "Paramètre": ["Quantité Nominale (QN)", "Erreur maximale tolérée (E)", "Surpoids max", "Effectif d'échantillon (n)", 
                         "Fréquence d'échantillonnage (par heure)", "Écart-type du processus (σ₀)", "Seuil de centrage (ms)", 
                         "Poids cible (QC)", "Coefficient statistique (g)", "Limite de défectueux (TU1)", "Nouvelle valeur cible (m1)", 
                         "δ", "δ√n", "POM", "POl"],
            "Valeur": [qn, e, surpoids_max, n, frequence, sigma_0, ms, qc, g, tu1, m1, delta, delta_sqrt_n, pom, pol],
            "Description": ["Poids nominal indiqué sur l'emballage.", "Valeur de E selon l'annexe 5 du guide DGCCRF.",
                           "Surpoids maximal acceptable pour chaque pack.", "Nombre de packs à peser dans chaque échantillon.",
                           "Nombre d'échantillons à contrôler par heure.", "Écart-type des poids bruts mesurés.",
                           "Valeur minimale de la moyenne du processus pour respecter le critère des défectueux.",
                           "Valeur cible de remplissage, incluant le surpoids maximum toléré.",
                           "Coefficient statistique pour un seuil de confiance de 90% (à adapter selon l'annexe 3 du guide).",
                           "Limite en dessous de laquelle un pack est considéré comme défectueux.",
                           "Valeur cible après un déréglage de 0.2% de la quantité nominale.",
                           "Déviation de la moyenne de l'échantillon par rapport à m1.",
                           "δ multiplié par la racine carrée de l'effectif d'échantillon.",
                           "Nombre moyen d'échantillons nécessaires pour détecter un déréglage de 0.2% de la quantité nominale.",
                           "Nombre maximum d'échantillons que vous pouvez contrôler par heure."]
        })

        # Créer un fichier Excel avec les résultats
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df_resultats.to_excel(writer, sheet_name='Résultats', index=False)
        writer.save()

        # Télécharger le fichier Excel
        st.download_button(
            label="Télécharger le rapport Excel",
            data=output.getvalue(),
            file_name="rapport_validation.xlsx",
            mime="application/vnd.ms-excel"
        )

else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
