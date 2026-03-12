import shap
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model/medipredict_model.pkl")

def get_shap_values(X_train_scaled, X_input_scaled):
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_input_scaled)
    return shap_values

def plot_waterfall(shap_values):
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], max_display=8, show=False)
    return fig