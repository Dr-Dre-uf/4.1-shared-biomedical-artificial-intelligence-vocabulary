import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, make_scorer, recall_score
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
st.set_page_config(page_title="Mod 4_Week 1_Notebook 1", layout="wide")

st.sidebar.markdown("### 1. Select Perspective")
perspective = st.sidebar.radio(
    "View application through the lens of:",
    ["Clinical Care", "Foundational Science"],
    help="Toggle this to see how the same machine learning pipeline is interpreted differently."
)

st.sidebar.markdown("### 2. Navigation")
activity = st.sidebar.radio(
    "🚀 Go to:",
    [
        "Activity 1 - Exploring data types",
        "Activity 2 - Data preprocessing",
        "Activity 3 - Model training",
        "Activity 4 - Cross-validation",
        "Activity 5 - Alternative methods"
    ]
)

display_performance_monitor()

# ---------------------------------
# Context Variables
# ---------------------------------
if perspective == "Clinical Care":
    app_desc = "You are part of a hospital’s clinical analytics team using a decision tree to predict in-hospital mortality using data from the eICU Collaborative Research Database."
    outcome_label = "In-Hospital Mortality (0=Survival, 1=Death)"
else:
    app_desc = "You are a computational biologist investigating pathophysiological mechanisms. Train Decision Tree models to discover critical thresholds in physiological biomarkers."
    outcome_label = "Systemic Failure (0=Stable, 1=Failure)"

st.title("🧬Module 4 Week 1 Notebook 1 – Shared biomedical artificial intelligence vocabulary")
st.write(app_desc)

# ---------------------------------
# Load Dataset (Exact Notebook format)
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()

# --------------------
# Activity 1 - Data Exploration
# --------------------
if activity == "Activity 1 - Exploring data types":
    st.header("🚀 Activity 1: Exploring data types")
    st.write(f"**8 model inputs (features):** The patient characteristics passed into the model.\n**1 model output (target):** {outcome_label}")

    st.subheader("Data Preview")
    n_rows = st.slider("Number of records to display", 1, 20, 5)
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    with st.expander("View Data Types (.dtypes)"):
        st.write(df.dtypes)
    
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to visualize:", df.columns[:-1])
    
    # ADA COMPLIANCE: Colorblind safe colors (Blue & Orange)
    fig = px.histogram(
        df, x=feature_to_plot, color="Outcome", barmode="overlay",
        title=f"Distribution of {feature_to_plot} grouped by Outcome",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"] 
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View chart data as text (Accessible Alternative)"):
        st.dataframe(df.groupby("Outcome")[feature_to_plot].describe())
        
    st.info("❓ **Questions:**\n1. Which is the outcome variable and what type of data it is?\n2. Which are the predictor variable and what type of data it is?")

# --------------------
# Activity 2 - Preprocessing
# --------------------
elif activity == "Activity 2 - Data preprocessing":
    st.header("🚀 Activity 2: Data preprocessing, model training and cross-validation")
    st.write("We prepare our data by splitting it into features (X) and target (y), followed by a train/test split.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    test_size = st.slider("Test set size (default is 0.2)", 0.1, 0.5, 0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    split_df = pd.DataFrame({
        "Set": ["Training Data", "Testing Data"],
        "Count": [len(X_train), len(X_test)]
    })
    
    fig = px.pie(
        split_df, values="Count", names="Set", title="Train/Test Split", hole=0.4,
        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View split proportions as text (Accessible Alternative)"):
        st.dataframe(split_df)
        
    st.info("❓ **Question:** Explain why doing this?")

# --------------------
# Activity 3 - Train & Predict
# --------------------
elif activity == "Activity 3 - Model training":
    st.header("🚀 Activity 3: Model training")
    
    # Fixed depth as per notebook: max_depth=4
    max_depth = st.slider("Max Depth", 1, 15, 4, help="Notebook default is 4 to prevent overfitting.")
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Accuracy (acc = accuracy_score)", f"{acc:.16f}")
    
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt, feature_names=df.columns[:-1], class_names=["0", "1"], filled=True, rounded=True, fontsize=10)
    st.pyplot(fig)
    
    with st.expander("View Decision Tree rules as text (Accessible Alternative)"):
        st.text(export_text(dt, feature_names=list(df.columns[:-1])))
        
    st.info("❓ **Questions:**\n1. What is the output. How to interpret?\n2. How is the performance? Do you believe it? Why?")

# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-validation":
    st.header("🚀 Activity 4: Cross-validation")
    st.write("Using 5-fold cross-validation to assess performance robustly.")

    cv_folds = st.slider("Folds (cv)", 2, 10, 5)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Exact CV implementation from Notebook
    scoring = {
        "accuracy": "accuracy",
        "sensitivity": "recall",                  
        "precision": "precision",
        "specificity": make_scorer(recall_score, pos_label=0)  
    }
    
    with st.spinner('Running cross-validation...'):
        cv_results = cross_validate(dt, X, y, cv=cv_folds, scoring=scoring)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Accuracy", f"{np.mean(cv_results['test_accuracy']):.4f}")
    col2.metric("Average Sensitivity", f"{np.mean(cv_results['test_sensitivity']):.4f}")
    col3.metric("Average Specificity", f"{np.mean(cv_results['test_specificity']):.4f}")
    col4.metric("Average Precision", f"{np.mean(cv_results['test_precision']):.4f}")

    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)], 
        "Accuracy": cv_results['test_accuracy']
    })
    
    fig = px.bar(
        cv_df, x="Fold", y="Accuracy", title="Accuracy per Fold", 
        text_auto=".3f", range_y=[0, 1], color_discrete_sequence=["#1f77b4"]
    )
    fig.add_hline(y=np.mean(cv_results['test_accuracy']), line_dash="dash", line_color="#ff7f0e", annotation_text="Mean")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View fold metrics as text (Accessible Alternative)"):
        st.dataframe(cv_df)

    st.info("❓ **Question:** How is the performance now?")

# --------------------
# Activity 5 - Alternative Methods
# --------------------
elif activity == "Activity 5 - Alternative methods":
    st.header("🚀 Activity 5: Alternative methods?")
    
    st.write("### 🌲 Decision Tree Characteristics")
    st.write("A decision tree predicts an outcome by repeatedly asking yes/no or threshold-based questions about the features.")
    st.write("**Decision mechanism:**\n* Start with all data at the root node.\n* Choose the best feature and threshold to split the data (using Gini impurity or Entropy).\n* Create branches and repeat until maximum depth is reached.")
    
    st.info("❓ **Questions:**\n1. Why using decision tree?\n2. What other methods do you think to use? Why?")
    
    st.write("*(Use your course knowledge to answer these reflection questions in your notebook or Canvas.)*")
