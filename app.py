import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.predict import predict_risk
from src.visualize import plot_histogram, plot_corr

st.set_page_config(page_title="MediPredict", layout="wide")

df = pd.read_csv("data/diabetes.csv")

if "consent" not in st.session_state:
    st.session_state.consent = False

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Accueil",
    "Mon profil de risque",
    "Comprendre ma prédiction",
    "Explorer les données"
])

# PAGE 1
if page == "Accueil":
    st.title("MediPredict")
    st.write("Estimez votre niveau de risque de diabète de type 2 à partir d'indicateurs de santé anonymes.")

    st.warning("Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical. En cas de doute, consultez un professionnel de santé.")

    st.subheader("Politique de confidentialité")
    st.write("Les données saisies ne sont ni enregistrées ni partagées. Elles sont utilisées uniquement pendant votre session pour produire une estimation de risque. Aucun historique n'est conservé.")

    consent = st.checkbox("Je consens à utiliser cet outil de sensibilisation")
    if consent:
        st.session_state.consent = True
        st.success("Consentement enregistré.")

# PAGE 2
elif page == "Mon profil de risque":
    if not st.session_state.consent:
        st.error("Vous devez d'abord donner votre consentement dans la page Accueil.")
    else:
        st.title("Mon profil de risque")

        pregnancies_option = st.selectbox("Grossesses", ["Non applicable", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pregnancies = 0 if pregnancies_option == "Non applicable" else pregnancies_option

        glucose = st.number_input("Glucose", min_value=40, max_value=250, value=100, help="Taux de glucose dans le sang")
        bp = st.number_input("Pression artérielle", min_value=40, max_value=140, value=70)
        skin = st.number_input("Épaisseur cutanée", min_value=5, max_value=100, value=20)
        insulin = st.number_input("Insuline", min_value=15, max_value=900, value=80)
        bmi = st.number_input("Indice de masse corporelle", min_value=10.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Antécédents familiaux (score)", min_value=0.05, max_value=3.0, value=0.5)
        age = st.number_input("Âge", min_value=21, max_value=90, value=30)

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

            proba, level, df_input, X_scaled = predict_risk(user_data)

            st.session_state.user_data = user_data
            st.session_state.user_df = df_input
            st.session_state.user_scaled = X_scaled
            st.session_state.risk_level = level
            st.session_state.risk_proba = proba

            st.subheader("Résultat")
            st.metric("Niveau de risque", level)

# PAGE 3
elif page == "Comprendre ma prédiction":
    st.title("Comprendre ma prédiction")

    if "risk_level" not in st.session_state:
        st.info("Veuillez d'abord analyser votre profil.")
    else:
        st.write(f"Niveau de risque estimé : **{st.session_state.risk_level}**")

        st.subheader("Explication simple")
        st.write("Le résultat dépend principalement du glucose, de l'IMC, de l'âge et des antécédents familiaux.")

        st.subheader("Comparaison avec les données")
        fig = plot_histogram(df, "Glucose")
        st.pyplot(fig)

        st.subheader("Recommandation générale")
        st.write("Un taux de glucose élevé peut être un facteur de risque modifiable. Une alimentation équilibrée et un suivi médical peuvent aider.")

# PAGE 4
elif page == "Explorer les données":
    st.title("Explorer les données")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())

    st.subheader("Corrélations")
    st.pyplot(plot_corr(df))

    st.subheader("Transparence")
    st.write("""
    Modèle utilisé : Random Forest.
    Limites : dataset réduit, non représentatif de la population française, risque de biais selon les groupes.
    Cet outil est destiné à la sensibilisation et non au diagnostic.
    """)