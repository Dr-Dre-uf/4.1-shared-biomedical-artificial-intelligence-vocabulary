import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
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
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB")

# ---------------------------------
# Page Config & Sidebar
# ---------------------------------
st.set_page_config(page_title="eICU Mortality Prediction App", layout="wide")

st.sidebar.markdown("### 1. Select Perspective")
perspective = st.sidebar.radio(
    "Choose your scientific context:",
    ["Clinical Care", "Foundational Science"],
    help="Switching this changes the terminology and focus of the application."
)

st.sidebar.markdown("### 2. Navigation")
activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Explore eICU Data",
        "Activity 2 - Preprocessing & Splitting",
        "Activity 3 - Model Training (Neural Nets)",
        "Activity 4 - Cross-Validation Analysis"
    ]
)

display_performance_monitor()

# ---------------------------------
# Contextual Variables
# ---------------------------------
if perspective == "Clinical Care":
    app_title = "Predicting In-Hospital Mortality"
    app_desc = "Explore the machine learning pipeline for critical care. This app models in-hospital mortality using eICU data to support ICU triage and patient survival, serving as a stepping stone toward complex CNNs."
    outcome_label = "In-Hospital Mortality"
    simulator_title = "Live ICU Patient Simulator"
    simulator_desc = "Adjust the critical care metrics below. The model will instantly predict mortality risk based on patient patterns."
    alert_high = "High Risk of In-Hospital Mortality"
    alert_low = "Low Risk of In-Hospital Mortality"
else:
    app_title = "Modeling Systemic Failure Mechanisms"
    app_desc = "Explore the machine learning pipeline for biomedical research. This app models physiological collapse using eICU data to discover critical biomarker interactions and pathways."
    outcome_label = "Systemic Failure (Mortality)"
    simulator_title = "Live Biomarker Experiment"
    simulator_desc = "Adjust the biological assays below. The model will predict the likelihood of physiological systemic failure based on metabolic interactions."
    alert_high = "Systemic Failure Predicted"
    alert_low = "Physiological Stability Predicted"

st.title(app_title)
st.write(app_desc)

# ---------------------------------
# Load Dataset & Align with eICU / Mortality
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    
    # Rename columns to simulate the notebook's required eICU features (glucose, creatinine, potassium, etc.)
    icu_mapping = {
        "Pregnancies": "Prior ICU Admissions",
        "Glucose": "Admission Glucose (mg/dL)",
        "BloodPressure": "Mean Arterial Pressure",
        "SkinThickness": "Potassium Proxy (mEq/L)",
        "Insulin": "Creatinine Proxy (mg/dL)",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "Genetic Risk Index",
        "Age": "Age",
        "Outcome": "Target" # Kept generic here, renamed dynamically in charts
    }
    df = df.rename(columns=icu_mapping)
    return df

df = load_data()

# --------------------
# Activity 1 - Data Exploration
# --------------------
if activity == "Activity 1 - Explore eICU Data":
    st.header("Activity 1 - Exploring eICU Data Types")
    if perspective == "Clinical Care":
        st.write("Before building a neural network, inspect the vital signs and lab results to see how they correlate with patient survival.")
    else:
        st.write("Before building a neural network, inspect the biological distributions to identify potential markers of physiological collapse.")

    st.subheader("Data Preview & Types")
    n_rows = st.slider("Number of records to display", min_value=1, max_value=20, value=5)
    
    # Rename outcome column just for display based on perspective
    display_df = df.copy()
    display_df.rename(columns={"Target": outcome_label}, inplace=True)
    st.dataframe(display_df.head(n_rows), use_container_width=True)
    
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select an eICU feature to visualize:", df.columns[:-1])
    
    fig = px.histogram(
        display_df, 
        x=feature_to_plot, 
        color=outcome_label, 
        barmode="overlay",
        title=f"Distribution of {feature_to_plot} grouped by {outcome_label}",
        color_discrete_sequence=["#00CC96", "#EF553B"]
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 2 - Preprocessing
# --------------------
elif activity == "Activity 2 - Preprocessing & Splitting":
    st.header("Activity 2 - Data Preprocessing")
    st.write("Neural Networks are highly sensitive to unscaled data. Here, we apply feature scaling to ensure variables like 'Genetic Risk Index' and 'Admission Glucose' are on the same mathematical scale.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Hold-out test set size (percentage)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    with col2:
        scale_option = st.selectbox(
            "Feature Scaling Technique", 
            ["StandardScaler", "MinMaxScaler", "None"],
            help="Deep Learning algorithms usually require scaling to converge properly."
        )

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

    # Visualize the split
    split_df = pd.DataFrame({
        "Set": ["Training Data (Model Learning)", "Testing Data (Model Evaluation)"],
        "Count": [len(X_train), len(X_test)]
    })
    fig = px.pie(split_df, values="Count", names="Set", title="eICU Cohort Data Split", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 3 - Train & Predict
# --------------------
elif activity == "Activity 3 - Model Training (Neural Nets)":
    st.header("Activity 3 - Model Training & Prediction")
    st.write("Train an algorithm. You can choose a standard Decision Tree or a Deep Neural Network (MLP), which forms the foundational architecture for advanced Convolutional Neural Networks (CNNs).")

    # 1. Choose & Train Model
    st.subheader("1. Configure the Model")
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox("Choose an Algorithm", ["Deep Neural Network (MLP)", "Decision Tree", "Random Forest"])
    
    with col2:
        if model_choice == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 4)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_choice == "Random Forest":
            max_depth = st.slider("Max Depth", 1, 20, 4)
            model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
        else:
            hidden_layers = st.selectbox("Hidden Layer Architecture", [(64, 32), (128, 64, 32), (32, 16)])
            st.caption("Multiple hidden layers allow the network to learn complex physiological representations.")
            model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)

    # Train behind the scenes
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Force scaling for Neural Net to prevent convergence warnings
    if model_choice == "Deep Neural Network (MLP)":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    with st.spinner('Training model...'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    st.metric("Test Set Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")

    st.divider()

    # 2. Live Simulator
    st.subheader(f"2. {simulator_title}")
    st.write(simulator_desc)
    
    input_data = {}
    input_cols = st.columns(4)
    for idx, col in enumerate(df.columns[:-1]):
        with input_cols[idx % 4]:
            input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))

    # Make Prediction
    user_df = pd.DataFrame([input_data])
    
    if model_choice == "Deep Neural Network (MLP)":
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        prob = model.predict_proba(user_scaled)[0]
    else:
        prediction = model.predict(user_df)[0]
        prob = model.predict_proba(user_df)[0]
    
    if prediction == 1:
        st.error(f"**Model Alert:** {alert_high}")
    else:
        st.success(f"**Model Alert:** {alert_low}")
        
    st.progress(prob[1], text=f"Calculated Probability: {prob[1]*100:.1f}%")

# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-Validation Analysis":
    st.header("Activity 4 - Cross-Validation Analysis")
    st.write("Cross-validation splits the data into multiple 'folds' to ensure our model is stable and reproducible across different data subpopulations.")

    model_choice = st.selectbox("Select Model for Cross-Validation", ["Deep Neural Network (MLP)", "Decision Tree", "Random Forest"])
    cv_folds = st.slider("Number of Validation Folds", 2, 10, 5)

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if model_choice == "Deep Neural Network (MLP)":
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    
    with st.spinner('Running cross-validation...'):
        scores = cross_val_score(model, X, y, cv=cv_folds)

    st.metric("Mean CV Accuracy", f"{np.mean(scores):.3f}")

    # Plot the results
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(cv_folds)], "Accuracy": scores})
    
    fig = px.bar(cv_df, x="Fold", y="Accuracy", title=f"{outcome_label} Prediction Accuracy per Validation Fold", text_auto=".3f", range_y=[0, 1])
    fig.add_hline(y=np.mean(scores), line_dash="dash", line_color="red", annotation_text="Mean Accuracy")
    st.plotly_chart(fig, use_container_width=True)
