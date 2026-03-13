import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.predict import predict_risk
from src.visualize import (
    plot_histogram,
    plot_corr,
    plot_confusion_matrix,
    plot_roc_curve
)
from src.explain import (
    get_shap_values,
    plot_waterfall,
    generate_natural_explanation
)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="MediPredict",
    layout="wide"
)

# -----------------------------
# CHARGEMENT DES DONNÉES ET OBJETS
# -----------------------------
df = pd.read_csv("data/diabetes.csv")

model = joblib.load("model/medipredict_model.pkl")
scaler = joblib.load("model/scaler.pkl")
imputer = joblib.load("model/imputer.pkl")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

FEATURE_LABELS_FR = {
    "Pregnancies": "Grossesses",
    "Glucose": "Glucose",
    "BloodPressure": "Pression artérielle",
    "SkinThickness": "Épaisseur cutanée",
    "Insulin": "Insuline",
    "BMI": "Indice de masse corporelle",
    "DiabetesPedigreeFunction": "Antécédents familiaux",
    "Age": "Âge"
}

# Préparation dataset transformé pour SHAP + métriques modèle
X = df[FEATURES].copy()
y = df["Outcome"].copy()

X_imp = imputer.transform(X)
X_scaled = scaler.transform(X_imp)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# ÉTAT SESSION
# -----------------------------
if "consent" not in st.session_state:
    st.session_state.consent = False

if "risk_level" not in st.session_state:
    st.session_state.risk_level = None

# -----------------------------
# FONCTIONS UI
# -----------------------------
def display_risk_gauge(level: str, proba: float):
    """Affichage simple type jauge avec libellé texte."""
    st.subheader("Résultat")

    if level == "Faible":
        progress_value = min(int(proba * 100), 33)
        color_box = "🟢"
        advice = "Risque faible"
    elif level == "Modéré":
        progress_value = max(34, min(int(proba * 100), 66))
        color_box = "🟠"
        advice = "Risque modéré"
    else:
        progress_value = max(67, min(int(proba * 100), 100))
        color_box = "🔴"
        advice = "Risque élevé"

    st.metric("Niveau de risque", f"{color_box} {advice}")
    st.progress(progress_value / 100)
    st.caption(
        f"Estimation du risque : {progress_value}% — cette valeur correspond à un niveau indicatif de risque et ne constitue pas un diagnostic."
    )


def validate_user_inputs(user_data: dict):
    """Validation supplémentaire côté serveur."""
    errors = []

    if not (40 <= user_data["Glucose"] <= 250):
        errors.append("Le glucose doit être compris entre 40 et 250.")
    if not (40 <= user_data["BloodPressure"] <= 140):
        errors.append("La pression artérielle doit être comprise entre 40 et 140.")
    if not (5 <= user_data["SkinThickness"] <= 100):
        errors.append("L'épaisseur cutanée doit être comprise entre 5 et 100.")
    if not (15 <= user_data["Insulin"] <= 900):
        errors.append("L'insuline doit être comprise entre 15 et 900.")
    if not (10.0 <= user_data["BMI"] <= 70.0):
        errors.append("L'indice de masse corporelle doit être compris entre 10 et 70.")
    if not (0.05 <= user_data["DiabetesPedigreeFunction"] <= 3.0):
        errors.append("Le score d'antécédents familiaux doit être compris entre 0.05 et 3.0.")
    if not (21 <= user_data["Age"] <= 90):
        errors.append("L'âge doit être compris entre 21 et 90.")

    return errors


def generate_recommendations(user_data: dict):
    """Recommandations génériques, non médicales."""
    recommendations = []

    if user_data["Glucose"] > 140:
        recommendations.append(
            "Un taux de glucose élevé est un facteur de risque important. Une alimentation équilibrée et un suivi médical peuvent aider."
        )

    if user_data["BMI"] > 30:
        recommendations.append(
            "Un IMC élevé peut augmenter le risque. Une activité physique régulière et un accompagnement adapté peuvent être bénéfiques."
        )

    if user_data["BloodPressure"] > 90:
        recommendations.append(
            "Une pression artérielle élevée peut être un facteur de vigilance supplémentaire."
        )

    if user_data["Age"] > 45:
        recommendations.append(
            "L'âge est un facteur de risque non modifiable. La prévention et le suivi régulier sont donc particulièrement importants."
        )

    if user_data["DiabetesPedigreeFunction"] > 0.8:
        recommendations.append(
            "Des antécédents familiaux élevés peuvent renforcer le risque estimé. Cela justifie une vigilance accrue."
        )

    if not recommendations:
        recommendations.append(
            "Votre profil ne présente pas de facteur très marqué, mais il reste important de conserver de bonnes habitudes de vie."
        )

    recommendations.append(
        "Cet outil est une aide à la sensibilisation et ne remplace pas un professionnel de santé."
    )

    return recommendations


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à",
    [
        "Accueil",
        "Mon profil de risque",
        "Comprendre ma prédiction",
        "Explorer les données"
    ]
)

# -----------------------------
# PAGE 1 — ACCUEIL
# -----------------------------
if page == "Accueil":
    st.title("MediPredict")
    st.write(
        "Estimez votre niveau de risque de diabète de type 2 à partir d'indicateurs de santé anonymes."
    )

    st.warning(
        "Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical. "
        "En cas de doute, consultez un professionnel de santé."
    )

    st.subheader("Politique de confidentialité")
    st.write(
        "Les données saisies ne sont ni enregistrées ni partagées. "
        "Elles sont utilisées uniquement pendant votre session pour produire une estimation de risque. "
        "Aucun historique n'est conservé."
    )

    consent = st.checkbox("Je consens à utiliser cet outil de sensibilisation")
    if consent:
        st.session_state.consent = True
        st.success("Consentement enregistré.")
    else:
        st.session_state.consent = False

# -----------------------------
# PAGE 2 — MON PROFIL DE RISQUE
# -----------------------------
elif page == "Mon profil de risque":
    st.title("Mon profil de risque")

    if not st.session_state.consent:
        st.error("Vous devez d'abord donner votre consentement dans la page Accueil.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            pregnancies_option = st.selectbox(
                "Grossesses",
                ["Non applicable", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                help="Sélectionnez 'Non applicable' si cette variable ne vous concerne pas."
            )
            pregnancies = 0 if pregnancies_option == "Non applicable" else pregnancies_option

            glucose = st.number_input(
                "Glucose",
                min_value=40,
                max_value=250,
                value=100,
                help="Taux de glucose dans le sang. Valeur attendue entre 40 et 250."
            )

            bp = st.number_input(
                "Pression artérielle",
                min_value=40,
                max_value=140,
                value=70,
                help="Valeur attendue entre 40 et 140."
            )

            skin = st.number_input(
                "Épaisseur cutanée",
                min_value=5,
                max_value=100,
                value=20,
                help="Valeur attendue entre 5 et 100."
            )

        with col2:
            insulin = st.number_input(
                "Insuline",
                min_value=15,
                max_value=900,
                value=80,
                help="Valeur attendue entre 15 et 900."
            )

            bmi = st.number_input(
                "Indice de masse corporelle",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                help="Valeur attendue entre 10 et 70."
            )

            dpf = st.number_input(
                "Antécédents familiaux (score)",
                min_value=0.05,
                max_value=3.0,
                value=0.5,
                help="Score lié aux antécédents familiaux de diabète."
            )

            age = st.number_input(
                "Âge",
                min_value=21,
                max_value=90,
                value=30,
                help="Âge en années."
            )

        if st.button("Analyser mon profil"):
            user_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }

            errors = validate_user_inputs(user_data)

            if errors:
                for error in errors:
                    st.error(error)
            else:
                proba, level, df_input, X_scaled_input = predict_risk(user_data)

                st.session_state.user_data = user_data
                st.session_state.user_df = df_input
                st.session_state.user_scaled = X_scaled_input
                st.session_state.risk_level = level
                st.session_state.risk_proba = proba

                display_risk_gauge(level, proba)

# -----------------------------
# PAGE 3 — COMPRENDRE MA PRÉDICTION
# -----------------------------
elif page == "Comprendre ma prédiction":
    st.title("Comprendre ma prédiction")

    if st.session_state.risk_level is None or "user_scaled" not in st.session_state:
        st.info("Veuillez d'abord analyser votre profil dans la page 'Mon profil de risque'.")
    else:
        st.write(f"Niveau de risque estimé : **{st.session_state.risk_level}**")

        st.subheader("Graphique SHAP individualisé")
        shap_values = get_shap_values(X_train_s, st.session_state.user_scaled)
        fig_shap = plot_waterfall(shap_values)
        st.pyplot(fig_shap)

        st.subheader("Explication en langage naturel")
        explanation = generate_natural_explanation(
            shap_values,
            st.session_state.user_df
        )
        st.write(explanation)

        st.subheader("Comparaison de votre profil avec le dataset")
        user_data = st.session_state.user_data

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_histogram(df, "Glucose", user_data["Glucose"]))
            st.pyplot(plot_histogram(df, "BMI", user_data["BMI"]))
        with col2:
            st.pyplot(plot_histogram(df, "Age", user_data["Age"]))
            st.pyplot(plot_histogram(df, "BloodPressure", user_data["BloodPressure"]))

        st.subheader("Recommandations génériques")
        for rec in generate_recommendations(user_data):
            st.write(f"- {rec}")

# -----------------------------
# PAGE 4 — EXPLORER LES DONNÉES
# -----------------------------
elif page == "Explorer les données":
    st.title("Explorer les données")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())

    st.subheader("Visualisations descriptives")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_histogram(df, "Glucose"))
        st.pyplot(plot_histogram(df, "BMI"))
    with col2:
        st.pyplot(plot_histogram(df, "Age"))
        st.pyplot(plot_histogram(df, "BloodPressure"))

    st.subheader("Corrélations")
    st.pyplot(plot_corr(df))

    st.subheader("Performance du modèle")
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_confusion_matrix(model, X_test_s, y_test_s))
    with col4:
        st.pyplot(plot_roc_curve(model, X_test_s, y_test_s))

    st.subheader("Transparence")
    st.write("""
**Modèle utilisé :** Random Forest

**Pourquoi ce choix ?**  
Nous avons comparé une régression logistique et une Random Forest.  
La Random Forest a été retenue car elle obtenait de meilleures performances sur notre dataset, notamment en accuracy, F1-score et AUC-ROC.

**Pipeline de traitement :**  
- remplacement des zéros biologiquement impossibles par des valeurs manquantes  
- imputation par la médiane  
- standardisation avec StandardScaler  
- séparation train/test avec stratification  
- entraînement et évaluation du modèle

**Limites du modèle :**  
- dataset de petite taille  
- population non représentative de la population française  
- risque de biais selon l'âge ou les antécédents familiaux  
- outil de sensibilisation uniquement, pas de diagnostic médical

**Biais identifiés :**  
- biais de sélection de population  
- déséquilibre partiel des classes  
- présence initiale de valeurs aberrantes

**Cadre éthique :**  
Cet outil est destiné à la sensibilisation et ne remplace pas un avis médical.
""")