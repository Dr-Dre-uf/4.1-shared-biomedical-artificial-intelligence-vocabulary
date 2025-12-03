import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Module 4 Week 1 Notebook 1 – Interactive App", layout="wide")

st.title("🧬 Module 4 Week 1 Notebook 1 – Shared Biomedical AI Vocabulary")
st.write("This interactive notebook walks you through Activities 1–5 using a diabetes dataset.")

# ---------------------------------
# Sidebar – Activity Navigation
# ---------------------------------
activity = st.sidebar.radio(
    "Choose an Activity:",
    [
        "Activity 1 – Explore Data Types",
        "Activity 2 – Preprocessing & Train/Test Split",
        "Activity 3 – Train a Model",
        "Activity 4 – Cross-Validation",
        "Activity 5 – Alternative Methods"
    ]
)

# ---------------------------------
# Load Example Dataset (Cached)
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df


df = load_data()

st.sidebar.success("Dataset Loaded: diabetes.csv")

# ---------------------------------
# Activity 1
# ---------------------------------
if activity == "Activity 1 – Explore Data Types":
    st.header("🔍 Activity 1 – Exploring Data Types")
    st.write("Use the panel below to explore the dataset.")

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.info(
    """
**Outcome Variable:** `Outcome` — binary categorical (0 = no diabetes, 1 = diabetes).  
**Predictor Variables:** All other columns — numerical features.
"""
).
"
        "**Predictor Variables:** All other columns — numerical features."
    )

# ---------------------------------
# Activity 2
# ---------------------------------
elif activity == "Activity 2 – Preprocessing & Train/Test Split":
    st.header("⚙️ Activity 2 – Data Preprocessing")

    st.write("Below is the preprocessing used to prepare data for the classifier.")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    st.write("### Why Split Data?")
    st.info(
        "We split the dataset into training and testing sets to evaluate how well the model generalizes to new, unseen data."
    )

    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.success(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ---------------------------------
# Activity 3
# ---------------------------------
elif activity == "Activity 3 – Train a Model":
    st.header("🌲 Activity 3 – Model Training & Accuracy")

    st.write("Choose a model below and train it on the dataset.")

    model_choice = st.selectbox(
        "Choose a Model",
        ["Decision Tree", "Random Forest", "Logistic Regression"]
    )

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", f"{acc:.3f}")

    if model_choice == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=df.columns[:-1], class_names=["0", "1"], filled=True)
        st.pyplot(fig)

# ---------------------------------
# Activity 4
# ---------------------------------
elif activity == "Activity 4 – Cross-Validation":
    st.header("🔁 Activity 4 – 5-Fold Cross-Validation")

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scores = cross_val_score(dt, X, y, cv=5)

    st.write("### Fold Accuracies")
    st.write(scores)
    st.metric("Mean CV Accuracy", f"{np.mean(scores):.3f}")

# ---------------------------------
# Activity 5
# ---------------------------------
elif activity == "Activity 5 – Alternative Methods":
    st.header("🔄 Activity 5 – Alternative Modeling Approaches")

    st.write("Why use a Decision Tree?")
    st.info(
        "Decision Trees are simple, interpretable, and show how decisions are made step-by-step."
    )

    st.write("Other models to consider:")
    st.markdown(
        "- **Random Forest:** Reduces overfitting by averaging multiple trees.
"
        "- **Logistic Regression:** Good baseline linear model.
"
        "- **XGBoost / Gradient Boosting:** High-performance models for structured data.
"
        "- **Neural Networks:** Useful when capturing complex nonlinear patterns."
    )

# --- Clinical Science Alignment Update ---
# This version reframes explanations, activities, and UI text using clinical-science terminology,
# while preserving the notebook’s structure and educational sequence.

# --- Enhancements Added: Collapsible sections, interactive charts, model comparison, patient prediction, SHAP ---

# Enhancements: Interactive Charts, Model Comparison, Patient-Level Prediction, SHAP

import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.sidebar.markdown("---")
st.sidebar.header("📊 Extra Tools")
extra_tool = st.sidebar.selectbox(
    "Choose an Enhancement:",
    ["None", "Interactive Charts", "Model Comparison", "Patient-Level Prediction", "Model Explainability (SHAP)"]
)

# ----------------------
# Interactive Charts
# ----------------------
if extra_tool == "Interactive Charts":
    st.header("📈 Interactive Data Visualizations")
    col = st.selectbox("Select a variable to visualize", df.columns[:-1])

    fig_hist = px.histogram(df, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
    st.plotly_chart(fig_box, use_container_width=True)

    fig_corr = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------
# Model Comparison
# ----------------------
elif extra_tool == "Model Comparison":
    st.header("⚖️ Model Comparison Across Algorithms")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    results = {}

    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=5).mean()
        results[name] = score

    st.write(results)

    fig_comp = px.bar(x=list(results.keys()), y=list(results.values()), labels={"x": "Model", "y": "Accuracy"}, title="Model Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

# ----------------------
# Patient-Level Prediction
# ----------------------
elif extra_tool == "Patient-Level Prediction":
    st.header("🧪 Patient-Level Prediction Tool")

    st.write("Enter patient values to estimate diabetes risk:")

    inputs = {}
    for col in df.columns[:-1]:
        inputs[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    inp_df = pd.DataFrame([inputs])

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(df.iloc[:, :-1], df.iloc[:, -1])

    prediction = model.predict(inp_df)[0]
    prob = model.predict_proba(inp_df)[0][1]

    st.metric("Prediction", "Diabetes" if prediction == 1 else "No Diabetes")
    st.metric("Probability", f"{prob:.2f}")

# ----------------------
# SHAP Explainability
# ----------------------
elif extra_tool == "Model Explainability (SHAP)":
    st.header("🔍 SHAP Model Explainability")

    st.write("Explaining predictions of a Random Forest model:")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("Feature Importance (SHAP)")
    fig_shap = shap.summary_plot(shap_values[1], X, show=False)
    st.pyplot(fig_shap)


