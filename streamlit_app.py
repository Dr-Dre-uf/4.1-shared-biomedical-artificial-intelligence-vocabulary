import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import plotly.express as px
import psutil
import os

# --- MONITORING UTILITY ---
def display_performance_monitor():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB")

# ---------------------------------
# Page Config & Sidebar
# ---------------------------------
st.set_page_config(page_title="Shared Biomedical AI Vocabulary", layout="wide")

st.sidebar.markdown("### 1. Select Perspective")
perspective = st.sidebar.radio(
    "View application through the lens of:",
    ["Clinical Care", "Foundational Science"]
)

st.sidebar.markdown("### 2. Navigation")
activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Explore eICU Data",
        "Activity 2 - Preprocessing & Splitting",
        "Activity 3 - Decision Tree Training",
        "Activity 4 - Cross-Validation Analysis"
    ]
)

display_performance_monitor()

# ---------------------------------
# Contextual Variables
# ---------------------------------
if perspective == "Clinical Care":
    app_desc = "Explore the machine learning pipeline for critical care. This app uses Decision Trees and eICU data to predict in-hospital mortality, supporting ICU triage and patient survival."
    outcome_label = "In-Hospital Mortality"
    simulator_title = "Live ICU Patient Simulator"
    alert_high = "High Risk of In-Hospital Mortality"
    alert_low = "Low Risk of In-Hospital Mortality"
else:
    app_desc = "Explore the machine learning pipeline for biomedical research. This app uses Decision Trees and eICU data to discover critical biological thresholds and metabolic pathways."
    outcome_label = "Systemic Failure (Mortality)"
    simulator_title = "Live Biomarker Experiment"
    alert_high = "Systemic Failure Predicted"
    alert_low = "Physiological Stability Predicted"

st.title("Shared Biomedical AI Vocabulary")
st.write(app_desc)

# ---------------------------------
# Load Dataset & Align with eICU
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    icu_mapping = {
        "Pregnancies": "Prior ICU Admissions",
        "Glucose": "Admission Glucose (mg/dL)",
        "BloodPressure": "Mean Arterial Pressure",
        "SkinThickness": "Potassium Proxy (mEq/L)",
        "Insulin": "Creatinine Proxy (mg/dL)",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "Genetic Risk Index",
        "Age": "Age",
        "Outcome": "Target" 
    }
    df = df.rename(columns=icu_mapping)
    return df

df = load_data()

# --------------------
# Activity 1 - Data Exploration
# --------------------
if activity == "Activity 1 - Explore eICU Data":
    st.header("Activity 1 - Exploring eICU Data Types")
    st.write(f"Inspect the distributions to identify potential markers of {outcome_label.lower()}.")

    st.subheader("Data Preview & Types")
    n_rows = st.slider("Number of records to display", 1, 20, 5)
    
    display_df = df.copy()
    display_df.rename(columns={"Target": outcome_label}, inplace=True)
    st.dataframe(display_df.head(n_rows), use_container_width=True)
    
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select an eICU feature to visualize:", df.columns[:-1])
    
    fig = px.histogram(
        display_df, x=feature_to_plot, color=outcome_label, barmode="overlay",
        title=f"Distribution of {feature_to_plot} grouped by {outcome_label}",
        color_discrete_sequence=["#00CC96", "#EF553B"]
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 2 - Preprocessing
# --------------------
elif activity == "Activity 2 - Preprocessing & Splitting":
    st.header("Activity 2 - Data Preprocessing")
    st.write("We partition our data into a Training Set (to teach the model) and a Testing Set (to evaluate the model on unseen data).")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    test_size = st.slider("Hold-out test set size (percentage)", 0.1, 0.5, 0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    split_df = pd.DataFrame({
        "Set": ["Training Data (Model Learning)", "Testing Data (Model Evaluation)"],
        "Count": [len(X_train), len(X_test)]
    })
    fig = px.pie(split_df, values="Count", names="Set", title="eICU Cohort Data Split", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 3 - Train & Predict
# --------------------
elif activity == "Activity 3 - Decision Tree Training":
    st.header("Activity 3 - Decision Tree Training & Prediction")

    st.subheader("1. Configure the Decision Tree")
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 1, 15, 4)
    with col2:
        min_samples = st.slider("Min Samples to Split", 2, 50, 2)

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.metric("Test Set Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    st.divider()

    st.subheader(f"2. {simulator_title}")
    input_data = {}
    input_cols = st.columns(4)
    for idx, col in enumerate(df.columns[:-1]):
        with input_cols[idx % 4]:
            input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))

    user_df = pd.DataFrame([input_data])
    prediction = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0]
    
    if prediction == 1:
        st.error(f"**Model Alert:** {alert_high}")
    else:
        st.success(f"**Model Alert:** {alert_low}")
    st.progress(prob[1], text=f"Calculated Probability: {prob[1]*100:.1f}%")
    st.divider()

    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(model, feature_names=df.columns[:-1], class_names=["Stable", "Failure/Mortality"], filled=True, fontsize=10, max_depth=3)
    st.pyplot(fig)

# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-Validation Analysis":
    st.header("Activity 4 - Cross-Validation Analysis")
    st.write("Evaluate the Decision Tree across multiple folds to calculate Average Accuracy, Sensitivity, Specificity, and Precision.")

    cv_folds = st.slider("Number of Validation Folds", 2, 10, 5)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Calculate advanced metrics to match the notebook
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies, sensitivities, specificities, precisions = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        model.fit(X_tr, y_tr)
        y_p = model.predict(X_te)
        
        accuracies.append(accuracy_score(y_te, y_p))
        precisions.append(precision_score(y_te, y_p, zero_division=0))
        sensitivities.append(recall_score(y_te, y_p, zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(y_te, y_p).ravel()
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Accuracy", f"{np.mean(accuracies):.4f}")
    col2.metric("Average Sensitivity", f"{np.mean(sensitivities):.4f}")
    col3.metric("Average Specificity", f"{np.mean(specificities):.4f}")
    col4.metric("Average Precision", f"{np.mean(precisions):.4f}")

    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(cv_folds)], "Accuracy": accuracies})
    fig = px.bar(cv_df, x="Fold", y="Accuracy", title=f"{outcome_label} Accuracy per Fold", text_auto=".3f", range_y=[0, 1])
    fig.add_hline(y=np.mean(accuracies), line_dash="dash", line_color="red", annotation_text="Mean Accuracy")
    st.plotly_chart(fig, use_container_width=True)
