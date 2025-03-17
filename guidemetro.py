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
    if delta >= 0:  # Cette ligne doit être indentée sous le bloc "with col2:"
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
