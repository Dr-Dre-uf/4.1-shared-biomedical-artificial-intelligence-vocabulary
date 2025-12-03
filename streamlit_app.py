import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import shap

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Module 4 Week 1 Notebook 1 – Interactive App", layout="wide")

st.title("Module 4 Week 1 Notebook 1 – Shared Biomedical AI Vocabulary")
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

# --------------------
# Activity 1 – Explore Data Types
# --------------------
if activity == "Activity 1 – Explore Data Types":
    st.header("Activity 1 – Exploring Data Types")

    st.subheader("Preview & Filter Data")
    selected_columns = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist())
    n_rows = st.slider("Number of rows to display", 1, 20, 5)
    st.dataframe(df[selected_columns].head(n_rows))

    st.subheader("Column Data Types")
    st.write(df[selected_columns].dtypes)

    st.info("""
Outcome Variable: `Outcome` — binary categorical (0 = no diabetes, 1 = diabetes).  
Predictor Variables: All other columns — numerical features.
""")

# --------------------
# Activity 2 – Preprocessing & Train/Test Split
# --------------------
elif activity == "Activity 2 – Preprocessing & Train/Test Split":
    st.header("Activity 2 – Data Preprocessing")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    st.write("Adjust Train/Test Split and Preprocessing")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    scale_option = st.selectbox("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if scale_option == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scale_option == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.success(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# --------------------
# Activity 3 – Train a Model
# --------------------
elif activity == "Activity 3 – Train a Model":
    st.header("Activity 3 – Model Training & Accuracy")

    model_choice = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "Logistic Regression"])

    if model_choice == "Decision Tree":
        max_depth = st.number_input("Decision Tree Max Depth", min_value=1, max_value=20, value=4)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_choice == "Random Forest":
        n_estimators = st.number_input("Random Forest n_estimators", min_value=10, max_value=500, value=100)
        max_depth = st.number_input("Random Forest Max Depth", min_value=1, max_value=20, value=4)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        max_iter = st.number_input("Logistic Regression max_iter", min_value=100, max_value=5000, value=1000)
        model = LogisticRegression(max_iter=max_iter)

    st.subheader("Optional: Modify first row values")
    edited_inputs = {}
    for col in df.columns[:-1]:
        edited_inputs[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].iloc[0]))
    st.write("Edited values (first row):", edited_inputs)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")

    if model_choice == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=df.columns[:-1], class_names=["0", "1"], filled=True)
        st.pyplot(fig)

# --------------------
# Activity 4 – Cross-Validation
# --------------------
elif activity == "Activity 4 – Cross-Validation":
    st.header("Activity 4 – Cross-Validation")

    model_choice = st.selectbox("Select Model for Cross-Validation", ["Decision Tree", "Random Forest", "Logistic Regression"], key="cv_model")
    cv_folds = st.slider("Number of folds", 2, 10, 5)

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_val_score(model, X, y, cv=cv_folds)

    st.write("Fold Accuracies")
    st.write(scores)
    st.metric("Mean CV Accuracy", f"{np.mean(scores):.3f}")

# --------------------
# Activity 5 – Alternative Methods
# --------------------
elif activity == "Activity 5 – Alternative Methods":
    st.header("Activity 5 – Alternative Modeling Approaches")

    st.write("Why use a Decision Tree?")
    st.info("Decision Trees are simple, interpretable, and show how decisions are made step-by-step.")

    st.write("Other models to consider:")
    st.markdown("""
- Random Forest: Reduces overfitting by averaging multiple trees.
- Logistic Regression: Good baseline linear model.
- XGBoost / Gradient Boosting: High-performance models for structured data.
- Neural Networks: Useful when capturing complex nonlinear patterns.
""")