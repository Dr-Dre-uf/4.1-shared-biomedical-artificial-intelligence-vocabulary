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
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB")

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Interactive ML App", layout="wide")

st.title("Interactive Biomedical AI Pipeline")
st.write("Explore the machine learning pipeline dynamically. Adjust the parameters in each section to instantly see how they affect the data, the models, and the predictions.")

# ---------------------------------
# Sidebar - Activity Navigation
# ---------------------------------
st.sidebar.markdown("### Navigation")

activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Interactive Data Exploration",
        "Activity 2 - Preprocessing & Splitting",
        "Activity 3 - Live Model Training & Prediction",
        "Activity 4 - Cross-Validation Analysis"
    ]
)

display_performance_monitor()

# ---------------------------------
# Load Example Dataset
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()
st.sidebar.success("Dataset Loaded: diabetes.csv")

# --------------------
# Activity 1 - Data Exploration
# --------------------
if activity == "Activity 1 - Interactive Data Exploration":
    st.header("Activity 1 - Interactive Data Exploration")

    st.subheader("Data Preview")
    n_rows = st.slider("Number of rows to display", min_value=1, max_value=20, value=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.subheader("Feature Distributions")
    st.write("Select a feature below to see how its values differ between patients with and without diabetes.")
    
    feature_to_plot = st.selectbox("Select a feature to visualize:", df.columns[:-1])
    
    # Interactive Plotly chart
    fig = px.histogram(
        df, 
        x=feature_to_plot, 
        color="Outcome", 
        barmode="overlay",
        title=f"Distribution of {feature_to_plot} grouped by Outcome",
        color_discrete_sequence=["#00CC96", "#EF553B"]
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 2 - Preprocessing
# --------------------
elif activity == "Activity 2 - Preprocessing & Splitting":
    st.header("Activity 2 - Preprocessing & Splitting")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (percentage)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    with col2:
        scale_option = st.selectbox("Feature Scaling Technique", ["None", "StandardScaler", "MinMaxScaler"])

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
        "Set": ["Training Data", "Testing Data"],
        "Count": [len(X_train), len(X_test)]
    })
    fig = px.pie(split_df, values="Count", names="Set", title="Train vs Test Split Proportion", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Activity 3 - Train & Predict
# --------------------
elif activity == "Activity 3 - Live Model Training & Prediction":
    st.header("Activity 3 - Live Model Training & Prediction")

    # 1. Choose & Train Model
    st.subheader("1. Configure the Model")
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "Logistic Regression"])
    
    with col2:
        if model_choice == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 4)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_choice == "Random Forest":
            max_depth = st.slider("Max Depth", 1, 20, 4)
            model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
        else:
            max_iter = st.slider("Max Iterations", 100, 2000, 1000)
            model = LogisticRegression(max_iter=max_iter)

    # Train behind the scenes
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.metric("Test Set Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")

    st.divider()

    # 2. Live Patient Simulator
    st.subheader("2. Live Patient Simulator")
    st.write("Adjust the metrics below. The model will instantly predict whether this hypothetical patient has diabetes based on the patterns it just learned.")
    
    input_data = {}
    input_cols = st.columns(4)
    
    for idx, col in enumerate(df.columns[:-1]):
        with input_cols[idx % 4]:
            # Use sliders for interactive feel
            input_data[col] = st.slider(
                col, 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].median())
            )

    # Format the user input for prediction
    user_df = pd.DataFrame([input_data])
    
    # Make Prediction
    prediction = model.predict(user_df)[0]
    
    if prediction == 1:
        st.error(f"**Model Prediction:** Diabetes Detected (Class 1)")
    else:
        st.success(f"**Model Prediction:** No Diabetes Detected (Class 0)")
        
    # Show probability if the model supports it
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(user_df)[0]
        st.progress(prob[1], text=f"Probability of Diabetes: {prob[1]*100:.1f}%")

    if model_choice == "Decision Tree":
        st.divider()
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(15, 6))
        plot_tree(model, feature_names=df.columns[:-1], class_names=["No Diabetes", "Diabetes"], filled=True, fontsize=10, max_depth=3)
        st.pyplot(fig)

# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-Validation Analysis":
    st.header("Activity 4 - Cross-Validation Analysis")

    model_choice = st.selectbox(
        "Select Model for Cross-Validation", 
        ["Decision Tree", "Random Forest", "Logistic Regression"]
    )
    cv_folds = st.slider("Number of Validation Folds", 2, 10, 5)

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    with st.spinner('Running cross-validation...'):
        scores = cross_val_score(model, X, y, cv=cv_folds)

    st.metric("Mean CV Accuracy", f"{np.mean(scores):.3f}")

    # Plot the results
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
        "Accuracy": scores
    })
    
    fig = px.bar(
        cv_df, 
        x="Fold", 
        y="Accuracy", 
        title="Accuracy per Validation Fold",
        text_auto=".3f",
        range_y=[0, 1]
    )
    fig.add_hline(y=np.mean(scores), line_dash="dash", line_color="red", annotation_text="Mean Accuracy")
    st.plotly_chart(fig, use_container_width=True)
