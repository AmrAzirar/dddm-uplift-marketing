import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Configuration page
st.set_page_config(
    page_title="Uplift Marketing Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('../data/hillstrom.csv')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['zip_code_enc'] = le.fit_transform(df['zip_code'])
    df['channel_enc'] = le.fit_transform(df['channel'])
    df['treatment'] = (df['segment'] != 'No E-Mail').astype(int)
    return df

@st.cache_data
def compute_uplift(df):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    FEATURES = ['recency', 'history', 'mens', 'womens',
                'newbie', 'zip_code_enc', 'channel_enc']

    X = df[FEATURES]
    y = df['conversion']
    t = df['treatment']

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42, stratify=y
    )

    model_treated = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
    model_control = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )

    model_treated.fit(X_train[t_train==1], y_train[t_train==1])
    model_control.fit(X_train[t_train==0], y_train[t_train==0])

    uplift_all = (model_treated.predict_proba(X)[:, 1] -
                  model_control.predict_proba(X)[:, 1])

    df['uplift_score'] = uplift_all

    p25 = np.percentile(uplift_all, 25)
    p75 = np.percentile(uplift_all, 75)

    def classify_persona(score, treatment):
        if score > p75:
            return 'Persuadable'
        elif score > 0 and treatment == 1:
            return 'Sure Thing'
        elif score < p25:
            return 'Sleeping Dog'
        else:
            return 'Lost Cause'

    df['persona'] = df.apply(
        lambda x: classify_persona(x['uplift_score'], x['treatment']),
        axis=1
    )

    return df, model_treated, model_control, FEATURES

# Chargement
df = load_data()
df, model_treated, model_control, FEATURES = compute_uplift(df)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
vue = st.sidebar.radio(
    "Choisir une vue :",
    ["Direction — KPIs", 
     "Marketing — Personas",
     "Operations — Scores",
     "Interpretabilite — SHAP",
     "Simulateur — Budget"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet DDDM — ENSIAS**")
st.sidebar.markdown("Uplift Marketing Hillstrom")

# ============================================================
# VUE 1 — DIRECTION
# ============================================================
if vue == "Direction — KPIs":
    st.title("Vue Direction — KPIs Globaux")
    st.markdown("---")

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)

    total_clients = len(df)
    persuadables = len(df[df['persona'] == 'Persuadable'])
    sure_things = len(df[df['persona'] == 'Sure Thing'])
    sleeping_dogs = len(df[df['persona'] == 'Sleeping Dog'])

    uplift_mens = 1.25 - 0.57
    uplift_womens = 0.88 - 0.57

    col1.metric("Total clients", f"{total_clients:,}")
    col2.metric("Persuadables", f"{persuadables:,}", 
                f"{persuadables/total_clients*100:.1f}%")
    col3.metric("Uplift email homme", f"+{uplift_mens:.2f}%")
    col4.metric("Budget economisable", 
                f"{(sure_things+sleeping_dogs)/total_clients*100:.0f}%")

    st.markdown("---")

    # ROI comparatif
    st.subheader("ROI — Campagne Blast vs Campagne Ciblee")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Campagne Blast")
        st.markdown(f"- Emails envoyes : **42 614**")
        st.markdown(f"- Cout total : **21 307 $**")
        st.markdown(f"- Convertis : **456**")
        st.markdown(f"- Revenu incremental : **~650 $**")

    with col2:
        st.markdown("### Campagne Ciblee (Uplift)")
        st.markdown(f"- Emails envoyes : **{persuadables:,}**")
        st.markdown(f"- Cout total : **{persuadables*0.5:,.0f} $**")
        st.markdown(f"- Memes convertis incrementaux")
        st.markdown(f"- Economie : **~{(42614-persuadables)*0.5:,.0f} $**")

    st.markdown("---")

    # Taux de conversion par groupe
    st.subheader("Taux de conversion par groupe experimental")
    conv_data = df.groupby('segment')['conversion'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['steelblue', 'gray', 'coral']
    ax.bar(conv_data.index, conv_data.values, color=colors)
    ax.set_ylabel('Taux de conversion (%)')
    ax.set_xlabel('Groupe')
    ax.axhline(y=0.57, color='red', linestyle='--', 
               label='Baseline (No Email)')
    ax.legend()
    st.pyplot(fig)

# ============================================================
# VUE 2 — MARKETING
# ============================================================
elif vue == "Marketing — Personas":
    st.title("Vue Marketing — Distribution des Personas")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nombre de clients par persona")
        persona_counts = df['persona'].value_counts()
        colors = {'Persuadable': '#e74c3c', 'Sure Thing': '#2ecc71',
                  'Lost Cause': '#95a5a6', 'Sleeping Dog': '#f39c12'}
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(persona_counts.index, persona_counts.values,
                      color=[colors[p] for p in persona_counts.index])
        ax.set_ylabel('Nombre de clients')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, persona_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 200,
                    f'{val:,}', ha='center', fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Taux de conversion par persona")
        persona_conv = df.groupby('persona')['conversion'].mean() * 100
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(persona_conv.index, persona_conv.values,
               color=[colors[p] for p in persona_conv.index])
        ax.set_ylabel('Taux de conversion (%)')
        ax.tick_params(axis='x', rotation=15)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Profil des Persuadables")

    persuadables_df = df[df['persona'] == 'Persuadable']
    col1, col2, col3 = st.columns(3)
    col1.metric("Recency moyenne", 
                f"{persuadables_df['recency'].mean():.1f} mois")
    col2.metric("History moyenne", 
                f"{persuadables_df['history'].mean():.0f} $")
    col3.metric("% Nouveaux clients", 
                f"{persuadables_df['newbie'].mean()*100:.0f}%")

    st.markdown("---")
    st.subheader("Distribution par zone geographique")
    zone_persona = pd.crosstab(df['zip_code'], df['persona'], 
                                normalize='index') * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    zone_persona.plot(kind='bar', ax=ax, 
                      color=['#95a5a6', '#f39c12', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Zone geographique')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='upper right')
    st.pyplot(fig)

# ============================================================
# VUE 3 — OPERATIONS
# ============================================================
elif vue == "Operations — Scores":
    st.title("Vue Operations — Scores Uplift")
    st.markdown("---")

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        zone_filter = st.selectbox(
            "Zone geographique",
            ['Toutes'] + list(df['zip_code'].unique())
        )
    with col2:
        canal_filter = st.selectbox(
            "Canal",
            ['Tous'] + list(df['channel'].unique())
        )
    with col3:
        persona_filter = st.selectbox(
            "Persona",
            ['Tous'] + list(df['persona'].unique())
        )

    # Application des filtres
    df_filtered = df.copy()
    if zone_filter != 'Toutes':
        df_filtered = df_filtered[df_filtered['zip_code'] == zone_filter]
    if canal_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['channel'] == canal_filter]
    if persona_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['persona'] == persona_filter]

    st.metric("Clients filtres", f"{len(df_filtered):,}")

    st.markdown("---")
    st.subheader("Distribution des scores uplift")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_filtered['uplift_score'], bins=50, 
            color='steelblue', edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', label='Seuil zero')
    ax.set_xlabel('Score uplift')
    ax.set_ylabel('Nombre de clients')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 20 clients a cibler (scores uplift les plus eleves)")
    top_clients = df_filtered.nlargest(20, 'uplift_score')[
        ['recency', 'history', 'channel', 'zip_code', 
         'persona', 'uplift_score']
    ].reset_index(drop=True)
    st.dataframe(top_clients, use_container_width=True)

# ============================================================
# VUE 4 — SHAP
# ============================================================
elif vue == "Interpretabilite — SHAP":
    st.title("Vue Interpretabilite — SHAP Values")
    st.markdown("---")

    st.info("Cette vue affiche les graphiques SHAP pre-calcules.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importance globale des features")
        if os.path.exists('../figures/08_shap_global.png'):
            st.image('../figures/08_shap_global.png')
        else:
            st.warning("Graphique non trouve — relancer le notebook 04")

    with col2:
        st.subheader("Impact local des features")
        if os.path.exists('../figures/09_shap_local.png'):
            st.image('../figures/09_shap_local.png')
        else:
            st.warning("Graphique non trouve — relancer le notebook 04")

    st.markdown("---")
    st.subheader("Interpretation des SHAP values")

    st.markdown("""
    | Feature | Importance | Interpretation |
    |---|---|---|
    | recency | Tres haute | Client recent = plus susceptible de convertir |
    | history | Tres haute | Gros acheteur = meilleur candidat |
    | newbie | Haute | Nouveau client reagit bien a l email |
    | womens | Haute | Acheteuse femme = profil specifique |
    | channel | Moyenne | Canal influence la conversion |
    | mens | Faible | Rayon homme moins discriminant |
    | zip_code | Faible | Zone geographique peu influente |
    """)

# ============================================================
# VUE 5 — SIMULATEUR
# ============================================================
elif vue == "Simulateur — Budget":
    st.title("Vue Simulateur — Budget Campagne")
    st.markdown("---")

    st.subheader("Parametres de la campagne")

    col1, col2 = st.columns(2)

    with col1:
        cout_email = st.slider(
            "Cout par email ($)", 
            min_value=0.1, max_value=2.0, 
            value=0.5, step=0.1
        )
        seuil_uplift = st.slider(
            "Seuil uplift minimum pour ciblage",
            min_value=0.0, max_value=0.05,
            value=0.008, step=0.001,
            format="%.3f"
        )

    with col2:
        revenu_moyen = st.slider(
            "Revenu moyen par conversion ($)",
            min_value=50, max_value=300,
            value=142, step=10
        )

    st.markdown("---")

    # Calculs
    clients_cibles = len(df[df['uplift_score'] >= seuil_uplift])
    clients_blast = len(df[df['treatment'] == 1])

    cout_blast = clients_blast * cout_email
    cout_cible = clients_cibles * cout_email
    economie = cout_blast - cout_cible
    pct_economie = economie / cout_blast * 100

    conv_blast = df[df['treatment']==1]['conversion'].sum()
    conv_cibles = df[df['uplift_score'] >= seuil_uplift]['conversion'].sum()

    revenu_blast = conv_blast * revenu_moyen
    revenu_cible = conv_cibles * revenu_moyen

    roi_blast = (revenu_blast - cout_blast) / cout_blast * 100
    roi_cible = (revenu_cible - cout_cible) / cout_cible * 100 if cout_cible > 0 else 0

    # Affichage resultats
    st.subheader("Resultats de la simulation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Clients cibles", f"{clients_cibles:,}",
                f"{clients_cibles - clients_blast:,} vs blast")
    col2.metric("Economie budget", f"{economie:,.0f} $",
                f"-{pct_economie:.0f}%")
    col3.metric("ROI cible vs blast",
                f"{roi_cible:.0f}%", 
                f"+{roi_cible - roi_blast:.0f}pts")

    st.markdown("---")

    # Comparaison visuelle
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cout
    axes[0].bar(['Blast', 'Ciblee'], 
                [cout_blast, cout_cible],
                color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Cout campagne ($)')
    axes[0].set_ylabel('Dollars')

    # ROI
    axes[1].bar(['Blast', 'Ciblee'],
                [roi_blast, roi_cible],
                color=['#e74c3c', '#2ecc71'])
    axes[1].set_title('ROI campagne (%)')
    axes[1].set_ylabel('%')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.info(f"En ciblant uniquement les clients avec uplift >= {seuil_uplift:.3f}, "
            f"tu economies {economie:,.0f}$ sur le budget campagne "
            f"({pct_economie:.0f}% de reduction).")