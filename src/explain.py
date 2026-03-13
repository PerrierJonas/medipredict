import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

model = joblib.load("model/medipredict_model.pkl")

def get_shap_values(X_background_scaled, X_input_scaled):
    explainer = shap.Explainer(model, X_background_scaled)
    shap_values = explainer(X_input_scaled)
    return shap_values

def plot_waterfall(shap_values):
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], max_display=8, show=False)
    return fig

def generate_natural_explanation(shap_values, user_df):
    values = shap_values[0, :, 1].values
    features = user_df.columns.tolist()

    shap_df = pd.DataFrame({
        "feature": features,
        "shap_value": values,
        "user_value": user_df.iloc[0].values
    })

    shap_df["abs_val"] = shap_df["shap_value"].abs()
    top_features = shap_df.sort_values("abs_val", ascending=False).head(3)

    parts = []
    for _, row in top_features.iterrows():
        direction = "augmente" if row["shap_value"] > 0 else "réduit"
        parts.append(f"{row['feature']} ({row['user_value']}) {direction} le risque estimé")

    return "Les facteurs les plus influents sont : " + ", ".join(parts) + "."