import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import psutil
import os

# --- MONITORING UTILITY ---
def display_performance_monitor():
    """Tracks CPU and RAM usage of the current Streamlit process."""
    process = psutil.Process(os.getpid())
    # Resident Set Size (Physical Memory) in MB
    mem_mb = process.memory_info().rss / (1024 * 1024)
    # CPU usage over a short interval
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
st.set_page_config(page_title="Module 4 Week 1 Notebook 1 - Interactive App", layout="wide")

st.title("Shared Biomedical AI Vocabulary")
st.write("This interactive app acts as a companion to your notebook. Follow the instructions in each section to explore the diabetes dataset, preprocess data, train models, and evaluate their performance.")

# ---------------------------------
# Sidebar - Activity Navigation & Monitor
# ---------------------------------
st.sidebar.markdown("### Navigation")
st.sidebar.write("Use the menu below to move through the different activities.")

activity = st.sidebar.radio(
    "Choose an Activity:",
    [
        "Activity 1 - Explore Data Types",
        "Activity 2 - Preprocessing & Train/Test Split",
        "Activity 3 - Train a Model",
        "Activity 4 - Cross-Validation",
        "Activity 5 - Alternative Methods"
    ],
    help="Select a module to learn about different stages of the machine learning pipeline."
)

# Load performance monitor into sidebar
display_performance_monitor()

# ---------------------------------
# Load Example Dataset (Cached)
# ---------------------------------
@st.cache_data
def load_data():
    # Caching stores this in RAM to avoid re-downloading
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()
st.sidebar.success("Dataset Loaded: diabetes.csv")

# --------------------
# Activity 1 - Explore Data Types
# --------------------
if activity == "Activity 1 - Explore Data Types":
    st.header("Activity 1 - Exploring Data Types")
    st.write("Before training a model, it is crucial to understand the data you are working with. Below, you can inspect the raw data and check the data types (e.g., integers, floats) of each column.")

    st.subheader("Preview & Filter Data")
    selected_columns = st.multiselect(
        "Select columns to view", 
        df.columns.tolist(), 
        default=df.columns.tolist(),
        help="Click and select/deselect specific features to isolate the data you want to look at."
    )
    n_rows = st.slider(
        "Number of rows to display", 
        min_value=1, max_value=20, value=5,
        help="Drag the slider to increase or decrease the number of rows visible in the table below."
    )
    st.dataframe(df[selected_columns].head(n_rows))

    st.subheader("Column Data Types")
    st.write("Machine learning models require numerical inputs. Let's verify that our data types are correct:")
    st.write(df[selected_columns].dtypes)

    st.info("""
**Outcome Variable:** `Outcome` — binary categorical (0 = no diabetes, 1 = diabetes).  
**Predictor Variables:** All other columns — numerical features used to predict the outcome.
""")

# --------------------
# Activity 2 - Preprocessing & Train/Test Split
# --------------------
elif activity == "Activity 2 - Preprocessing & Train/Test Split":
    st.header("Activity 2 - Data Preprocessing")
    st.write("In this step, we split our data into a **Training Set** (to teach the model) and a **Testing Set** (to evaluate the model). We also apply feature scaling so that all predictor variables share a similar scale, which helps many algorithms perform better.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    test_size = st.slider(
        "Test set size", 
        min_value=0.1, max_value=0.5, value=0.2,
        help="The proportion of the dataset to reserve for testing. 0.2 means 20% of the data will be hidden from the model during training."
    )
    
    scale_option = st.selectbox(
        "Feature Scaling", 
        ["None", "StandardScaler", "MinMaxScaler"],
        help="StandardScaler shifts data to have a mean of 0 and standard deviation of 1. MinMaxScaler compresses data to a set range, usually 0 to 1."
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

    st.success(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# --------------------
# Activity 3 - Train a Model
# --------------------
elif activity == "Activity 3 - Train a Model":
    st.header("Activity 3 - Model Training & Accuracy")
    st.write("Now we will train a machine learning algorithm to predict diabetes. You can choose different models and adjust their settings (hyperparameters) to see how it affects the final accuracy on the test set.")

    model_choice = st.selectbox(
        "Choose a Model", 
        ["Decision Tree", "Random Forest", "Logistic Regression"],
        help="Select the algorithm you want to train on the diabetes dataset."
    )

    if model_choice == "Decision Tree":
        max_depth = st.number_input(
            "Decision Tree Max Depth", 
            min_value=1, max_value=20, value=4,
            help="Controls how deep the tree can grow. A higher depth can lead to a more complex model but risks overfitting to the training data."
        )
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_choice == "Random Forest":
        n_estimators = st.number_input(
            "Random Forest n_estimators", 
            min_value=10, max_value=500, value=100,
            help="The number of individual decision trees to build. More trees usually improve performance but require more computing power."
        )
        max_depth = st.number_input(
            "Random Forest Max Depth", 
            min_value=1, max_value=20, value=4,
            help="The maximum depth of each individual tree in the forest."
        )
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        max_iter = st.number_input(
            "Logistic Regression max_iter", 
            min_value=100, max_value=5000, value=1000,
            help="The maximum number of iterations for the solver to converge. If your model fails to converge, increase this number."
        )
        model = LogisticRegression(max_iter=max_iter)

    st.subheader("Optional: Modify a specific patient's inputs")
    st.write("Adjust the values below to see how standardizing or modifying a single row affects the pipeline conceptually (these values represent an example first row).")
    edited_inputs = {}
    cols = df.columns[:-1]
    
    # Create columns for inputs to save space
    input_cols = st.columns(4)
    for idx, col in enumerate(cols):
        with input_cols[idx % 4]:
            edited_inputs[col] = st.number_input(
                f"{col}", 
                float(df[col].min()), float(df[col].max()), float(df[col].iloc[0]),
                help=f"Modify the {col} value within the dataset's min/max range."
            )

    # Re-run train test split behind the scenes for the overall test set
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Test Set Accuracy", f"{acc:.3f}", help="The percentage of correct predictions made by the model on the unseen test dataset.")

    if model_choice == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        st.write("The chart below maps out the exact mathematical rules the decision tree learned to classify patients.")
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=df.columns[:-1], class_names=["0", "1"], filled=True, fontsize=10)
        st.pyplot(fig)
        
# --------------------
# Activity 4 - Cross-Validation
# --------------------
elif activity == "Activity 4 - Cross-Validation":
    st.header("Activity 4 - Cross-Validation")
    st.write("A single Train/Test split might be lucky or unlucky. Cross-validation splits the data into multiple 'folds' and trains the model multiple times, giving us a more reliable average accuracy.")

    model_choice = st.selectbox(
        "Select Model for Cross-Validation", 
        ["Decision Tree", "Random Forest", "Logistic Regression"], 
        key="cv_model",
        help="Select the model to evaluate using cross-validation."
    )
    cv_folds = st.slider(
        "Number of folds", 
        min_value=2, max_value=10, value=5,
        help="How many pieces to divide the dataset into. 5 folds means the model trains 5 separate times, testing on a different 20% chunk each time."
    )

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    st.info("Running cross-validation triggers multiple training cycles. Check the System Monitor in the sidebar to see the CPU usage spike!")
    scores = cross_val_score(model, X, y, cv=cv_folds)

    st.write("Fold Accuracies")
    st.write(scores)
    st.metric("Mean CV Accuracy", f"{np.mean(scores):.3f}", help="The average accuracy across all the cross-validation folds. This represents true expected performance.")

# --------------------
# Activity 5 - Alternative Methods
# --------------------
elif activity == "Activity 5 - Alternative Methods":
    st.header("Activity 5 - Alternative Modeling Approaches")

    st.write("### Why use a Decision Tree?")
    st.info("Decision Trees are simple, highly interpretable, and clearly show how decisions are made step-by-step.")

    st.write("### Other models to consider:")
    st.markdown("""
- **Random Forest:** Reduces overfitting by averaging multiple trees together (this is known as an Ensemble method).
- **Logistic Regression:** A solid baseline linear model commonly used for binary classification tasks.
- **XGBoost / Gradient Boosting:** Highly complex and performant tree-based models often used to win machine learning competitions.
- **Neural Networks:** Useful when trying to capture complex, non-linear patterns within extremely large datasets.
""")
