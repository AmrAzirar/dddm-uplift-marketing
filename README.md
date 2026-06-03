# Uplift Marketing — Qui achète GRÂCE à l'email ?
## Projet DDDM — ENSIAS | Juin 2026

**Auteur :** Azirar Amr  / Anas Houhou
**Module :** Data-Driven Decision Making  
**Professeur :** Pr. Y. Tabii  
**Dataset :** Hillstrom E-Mail Analytics (64 000 clients)

---

## Problématique

Un retailer américain envoie des emails promotionnels à sa base client.
Le problème : combien de ces achats auraient eu lieu sans l'email ?

Notre approche par **Uplift Modeling** identifie les 4 types de clients :

| Segment | Comportement | Action |
|---|---|---|
| Persuadables | Achètent GRÂCE à l'email | Cibler en priorité |
| Sure Things | Achètent sans email | Ne pas contacter |
| Lost Causes | N'achètent jamais | Ignorer |
| Sleeping Dogs | L'email les fait fuir | Éviter absolument |

---

## Résultats Clés

- 15 999 Persuadables identifiés sur 64 000 clients
- Budget campagne réduit de 62%
- ROI amélioré de 193% à 328%
- Économie de 13 308$ par campagne

---

## Structure du Projet

    dddm_project/
    ├── 01_business_case.ipynb    
    ├── 02_data_audit.ipynb       
    ├── 03_eda.ipynb              
    ├── 04_modeling.ipynb         
    ├── 06_decision.ipynb         
    ├── ab_test_plan.md           
    ├── requirements.txt          
    ├── dashboard/
    │   └── app.py               
    ├── data/
    │   └── hillstrom.csv        
    └── figures/                 

---

## Installation

### 1. Cloner le repository

    git clone https://github.com/AmrAzirar/dddm-uplift-marketing.git
    cd dddm-uplift-marketing

### 2. Installer les dépendances

    pip install -r requirements.txt

### 3. Lancer les notebooks

    jupyter notebook

Ouvrir les notebooks dans l'ordre : 01 → 02 → 03 → 04 → 06

### 4. Lancer le dashboard

    cd dashboard
    streamlit run app.py

Dashboard accessible sur : http://localhost:8501

---

## Technologies

| Technologie | Usage |
|---|---|
| Python | Langage principal |
| Pandas / NumPy | Manipulation données |
| Scikit-learn | Modèles uplift |
| SHAP | Interprétabilité |
| Streamlit | Dashboard interactif |
| Scipy | Tests statistiques |
| Matplotlib / Seaborn | Visualisations |
| Git / GitHub | Versioning |

---

## Modèles Entraînés

| Modèle | AUUC | Statut |
|---|---|---|
| S-Learner | 24.08 | Entraîné |
| T-Learner | 24.24 | Retenu |
| X-Learner | 23.19 | Entraîné |

---

## Impact Financier

| Scénario | Emails | Coût | ROI |
|---|---|---|---|
| Blast (avant) | 42 614 | 21 307$ | 193% |
| Ciblée (après) | 15 999 | 8 000$ | 328% |
| Économie | -26 615 | -13 308$ | +70% |

---

## Licence
Projet académique — ENSIAS 2026
