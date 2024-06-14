Voici un document explicatif en format Markdown pour détailler les étapes de fonctionnement et d'utilisation de cette application :

# Validation d'échantillonnage pour le contrôle métrologique

Cette application Streamlit vous aide à valider votre stratégie d'échantillonnage pour le contrôle métrologique de vos préemballages, conformément au guide de bonnes pratiques de la DGCCRF (Direction générale de la concurrence, de la consommation et de la répression des fraudes).

## Instructions pour la création du fichier Excel

1. Créer un fichier Excel nommé `pesées.xlsx`.
2. Ajouter une feuille nommée `Pesées`.
3. Créer deux colonnes : `Tare (g)` et `Poids Brut (g)`.
4. Saisir les poids des emballages vides dans la colonne `Tare (g)`.
5. Saisir les poids bruts des packs remplis dans la colonne `Poids Brut (g)`.
6. Enregistrer le fichier Excel.

## Utilisation de l'application

1. Lancer l'application Streamlit.
2. Saisir les hypothèses de travail :
   - Quantité Nominale (g) : poids nominal indiqué sur l'emballage.
   - Erreur maximale tolérée (g) : valeur de E selon l'annexe 5 du guide DGCCRF.
   - Surpoids maximum toléré (g) : surpoids maximal acceptable pour chaque pack.
   - Effectif d'échantillon : nombre de packs à peser dans chaque échantillon.
   - Fréquence d'échantillonnage (par heure) : nombre d'échantillons à contrôler par heure.
3. Télécharger le fichier Excel contenant les poids des emballages vides et les poids bruts des packs.
4. L'application affichera les résultats de l'analyse de la tare et du poids brut, ainsi que les critères de conformité calculés.
5. L'application vérifiera si l'échantillonnage est validé ou non en comparant le POM (Période Opérationnelle Moyenne) et le POl (Période Opérationnelle Limite).
6. Si l'échantillonnage n'est pas suffisant, l'application proposera des alternatives d'échantillonnage pour atteindre un POM acceptable.
7. L'application générera un rapport Excel contenant les résultats de l'analyse, que vous pourrez télécharger.

## Fonctionnement de l'application

L'application utilise les bibliothèques Pandas, NumPy et SciPy pour effectuer des calculs statistiques et d'interpolation. Elle définit deux fonctions :

- `shapiro_test` : réalise le test de normalité de Shapiro-Wilk sur les données fournies.
- `calcul_pom` : calcule le POM (Période Opérationnelle Moyenne) à partir de la valeur de δ√n en utilisant une interpolation linéaire sur le tableau de l'Annexe 3 du guide DGCCRF.
- `calcul_pom_defectueux` : calcule le POM (Période Opérationnelle Moyenne) à partir de la valeur de δ√n en utilisant une interpolation linéaire sur le tableau de l'Annexe 4 du guide DGCCRF, pour le critère des défectueux.

L'application suit les étapes suivantes :

1. Saisie des hypothèses de travail par l'utilisateur.
2. Upload du fichier Excel contenant les données de pesée.
3. Analyse de la tare et du poids brut des données.
4. Calcul des critères de conformité.
5. Comparaison du POM et du POl pour valider l'échantillonnage.
6. Génération d'un rapport Excel contenant les résultats de l'analyse.

## Limites et améliorations possibles

- L'application ne vérifie pas la validité des données d'entrée. Il est recommandé de vérifier les données avant de les utiliser.
- L'application utilise une interpolation linéaire pour calculer le POM. Une méthode d'extrapolation plus robuste pourrait être utilisée pour les valeurs de δ√n en dehors de la plage du tableau.
- L'application ne prend pas en compte les autres critères de conformité mentionnés dans l'annexe 4 du guide DGCCRF. Il serait possible d'ajouter des fonctionnalités pour calculer ces critères supplémentaires.
- L'application ne propose pas de visualisation des données. Il serait possible d'ajouter des graphiques pour faciliter l'interprétation des résultats.
