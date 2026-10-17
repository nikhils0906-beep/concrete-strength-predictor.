import streamlit as st
st.markdown("""
<div style="text-align: right; font-size: 14px; color: gray;">
    Developed by:<br>
    <b>Nikhil</b><br>
    <b>Sufiyan</b>
</div>
""", unsafe_allow_html=True)

st.divider()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")

st.title("🏗️ Smart Concrete Compressive Strength Predictor")
st.caption("Educational machine-learning prototype | Estimated strength is not a laboratory result.")

@st.cache_resource
def load_and_train():
    ds = fetch_ucirepo(id=165)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rows.append({
            "Model": name,
            "R²": r2_score(y_test, pred),
            "MAE (MPa)": mean_absolute_error(y_test, pred),
            "RMSE (MPa)": np.sqrt(mean_squared_error(y_test, pred))
        })
        fitted[name] = m
    scores = pd.DataFrame(rows)
    best_name = scores.loc[scores["RMSE (MPa)"].idxmin(), "Model"]
    return X, scores, fitted, best_name

try:
    with st.spinner("Downloading dataset and training models..."):
        X, scores, fitted, best_name = load_and_train()
except Exception as exc:
    st.error("Could not load the dataset or train the models. Check your internet connection and try again.")
    st.code(str(exc))
    st.stop()

st.success(f"Models ready. Best model by held-out test RMSE: {best_name}")
st.subheader("1. Enter concrete mix details")
defaults = {
    "Cement": 350.0, "Blast Furnace Slag": 0.0, "Fly Ash": 0.0,
    "Water": 180.0, "Superplasticizer": 5.0, "Coarse Aggregate": 1000.0,
    "Fine Aggregate": 700.0, "Age": 28.0
}
units = {c: ("days" if c == "Age" else "kg/m³") for c in X.columns}
values = {}
left, right = st.columns(2)
for i, feature in enumerate(X.columns):
    col = left if i % 2 == 0 else right
    with col:
        max_value = 365.0 if feature == "Age" else max(float(X[feature].max()) * 1.5, defaults[feature] * 1.5)
        values[feature] = st.number_input(
            f"{feature} ({units[feature]})",
            min_value=0.0,
            max_value=float(max_value),
            value=float(defaults[feature]),
            step=1.0
        )

chosen = st.selectbox("Choose model", list(fitted.keys()), index=list(fitted.keys()).index(best_name))
if st.button("Predict Compressive Strength", type="primary"):
    inp = pd.DataFrame([[values[c] for c in X.columns]], columns=X.columns)
    pred = float(fitted[chosen].predict(inp)[0])
    st.subheader("2. Prediction")
    st.metric("Estimated Compressive Strength", f"{pred:.2f} MPa")
    st.warning("Educational estimate only. Do not use this result to approve a mix or determine structural safety. Confirm strength through appropriate laboratory testing and engineering review.")

st.subheader("3. Model comparison")
display_scores = scores.copy()
for c in ["R²", "MAE (MPa)", "RMSE (MPa)"]:
    display_scores[c] = display_scores[c].map(lambda v: round(float(v), 4 if c == "R²" else 2))
st.dataframe(display_scores, use_container_width=True, hide_index=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(scores["Model"], scores["R²"])
ax.set_title("R² Score Comparison")
ax.set_ylabel("R² score")
ax.set_ylim(0, 1)
plt.xticks(rotation=10)
st.pyplot(fig)
plt.close(fig)

with st.expander("About this project"):
    st.write("The UCI dataset contains historical concrete mix records. Models are evaluated on a held-out 20% test split. Scores may differ from earlier runs if the data split, dataset version or code changes.")
    st.write("Input quantities should follow the dataset units. Model output is an estimate, not a substitute for laboratory tests, standards or professional engineering judgement.")
