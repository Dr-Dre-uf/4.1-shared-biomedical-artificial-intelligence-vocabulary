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
    """Tracks CPU and RAM usage of the current Streamlit process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    st.sidebar.caption("Tracks the resource usage of this app in real-time.")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%", help="Current CPU usage of the Streamlit server.")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB", help="Current RAM memory allocated to this app.")

# ---------------------------------
# Page Config & Sidebar
# ---------------------------------
st.set_page_config(page_title="Shared Biomedical AI Vocabulary", layout="wide")

st.sidebar.markdown("### 1. Select Perspective")
perspective = st.sidebar.radio(
    "View application through the lens of:",
    ["Clinical Care", "Foundational Science"],
    help="Toggle this to see how the same machine learning pipeline is interpreted differently depending on your scientific goals."
)

st.sidebar.markdown("### 2. Navigation")
activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Explore eICU Data",
        "Activity 2 - Preprocessing & Splitting",
        "Activity 3 - Decision Tree Training",
        "Activity 4 - Cross-Validation Analysis"
    ],
    help="Select a module to move through the different stages of the machine learning pipeline."
)

display_performance_monitor()

# ---------------------------------
# Contextual Variables
# ---------------------------------
if perspective == "Clinical Care":
    app_desc = "Explore the machine learning pipeline for critical care. This app uses Decision Trees and eICU data to predict in-hospital mortality, supporting ICU triage and patient survival."
    outcome_label = "In-Hospital Mortality"
    simulator_title = "Live ICU Patient Simulator"
    simulator_desc = "Adjust the clinical metrics below. The model will instantly predict mortality risk based on patient patterns."
    alert_high = "High Risk of In-Hospital Mortality"
    alert_low = "Low Risk of In-Hospital Mortality"
else:
    app_desc = "Explore the machine learning pipeline for biomedical research. This app uses Decision Trees and eICU data to discover critical biological thresholds and metabolic pathways."
    outcome_label = "Systemic Failure (Mortality)"
    simulator_title = "Live Biomarker Experiment"
    simulator_desc = "Adjust the biological assays below. The model will predict the likelihood of physiological systemic failure based on metabolic interactions."
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
    st.write(f"**Instructions:** Before training a model, it is crucial to understand the data. Inspect the raw data below and explore how different health metrics correlate with the {outcome_label.lower()} outcome.")

    st.subheader("Data Preview & Types")
    n_rows = st.slider(
        "Number of records to display", 
        min_value=1, max_value=20, value=5,
        help="Drag the slider to increase or decrease the number of rows visible in the table below."
    )
    
    display_df = df.copy()
    display_df.rename(columns={"Target": outcome_label}, inplace=True)
    st.dataframe(display_df.head(n_rows), use_container_width=True)
    
    st.subheader("Feature Distributions")
    st.write("Use the dropdown below to visualize how specific features vary between the two outcome classes. Notice if certain thresholds seem to separate the classes.")
    
    feature_to_plot = st.selectbox(
        "Select an eICU feature to visualize:", 
        df.columns[:-1],
        help="Choose a specific metric (like Admission Glucose or BMI) to see a histogram of its distribution."
    )
    
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
    st.write("**Instructions:** In this step, we split our data into a **Training Set** (used to teach the algorithm) and a **Testing Set** (hidden from the model and used only to evaluate its performance). Adjust the slider to see how the proportions change.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    test_size = st.slider(
        "Hold-out test set size (percentage)", 
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="The proportion of the dataset to reserve for testing. 0.2 means 20% of the data will be hidden from the model during training."
    )
    
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
    st.write("**Instructions:** Configure the hyperparameters of the Decision Tree, then evaluate its accuracy on the test set. Finally, use the interactive simulator to test the model yourself.")

    st.subheader("1. Configure the Decision Tree")
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider(
            "Max Depth", 
            min_value=1, max_value=15, value=4,
            help="Controls how deep the tree can grow. Deeper trees capture more complex patterns but risk memorizing the training data (overfitting)."
        )
    with col2:
        min_samples = st.slider(
            "Min Samples to Split", 
            min_value=2, max_value=50, value=2,
            help="The minimum number of samples required to create a new branch. Higher numbers force the tree to be more generalized."
        )

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.metric(
        "Test Set Accuracy", 
        f"{accuracy_score(y_test, y_pred):.3f}",
        help="The percentage of correct predictions made by the model on the unseen test dataset."
    )
    st.divider()

    st.subheader(f"2. {simulator_title}")
    st.write(simulator_desc)
    
    input_data = {}
    input_cols = st.columns(4)
    for idx, col in enumerate(df.columns[:-1]):
        with input_cols[idx % 4]:
            input_data[col] = st.slider(
                col, 
                float(df[col].min()), float(df[col].max()), float(df[col].median()),
                help=f"Adjust the {col} value to see how the model reacts."
            )

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
    st.write("The chart below maps out the exact mathematical rules the algorithm learned. Follow the branches (True goes left, False goes right) to see how it makes a decision.")
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(model, feature_names=df.columns[:-1], class_names=["Stable/Survival", "Failure/Mortality"], filled=True, fontsize=10, max_depth=3)
    st.pyplot(fig)

# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-Validation Analysis":
    st.header("Activity 4 - Cross-Validation Analysis")
    st.write("**Instructions:** A single Train/Test split might be lucky or unlucky. Cross-validation splits the data into multiple 'folds' and evaluates the model multiple times to calculate Average Accuracy, Sensitivity, Specificity, and Precision.")

    cv_folds = st.slider(
        "Number of Validation Folds", 
        min_value=2, max_value=10, value=5,
        help="How many pieces to divide the dataset into. 5 folds means the model trains and tests 5 separate times on different chunks of data."
    )
    
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
    col1.metric("Average Accuracy", f"{np.mean(accuracies):.4f}", help="Overall proportion of correct predictions.")
    col2.metric("Average Sensitivity", f"{np.mean(sensitivities):.4f}", help="True Positive Rate: Ability to correctly identify the positive class (e.g., mortality).")
    col3.metric("Average Specificity", f"{np.mean(specificities):.4f}", help="True Negative Rate: Ability to correctly identify the negative class (e.g., survival).")
    col4.metric("Average Precision", f"{np.mean(precisions):.4f}", help="Positive Predictive Value: Proportion of positive predictions that were actually correct.")

    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(cv_folds)], "Accuracy": accuracies})
    fig = px.bar(
        cv_df, x="Fold", y="Accuracy", 
        title=f"{outcome_label} Accuracy per Fold", 
        text_auto=".3f", range_y=[0, 1]
    )
    fig.add_hline(y=np.mean(accuracies), line_dash="dash", line_color="red", annotation_text="Mean Accuracy")
    st.plotly_chart(fig, use_container_width=True)
