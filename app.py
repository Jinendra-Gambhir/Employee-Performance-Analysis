# Employee Performance Analysis Streamlit App

# Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set the page title
st.title("Employee Performance Analysis")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Model Building", "Results"])

# Load the data
df = pd.read_csv('HR-Employee-Attrition.csv')

# Data Overview Section
if pages == "Data Overview":
    st.header("Dataset Overview")
    
    # Display first few rows of the dataset
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Display the shape and description of the data
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("Data Description")
    st.write(df.describe())

    # Check for missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

# Data Visualization Section
elif pages == "Visualizations":
    st.header("Attrition and Feature Visualizations")

    # Attrition Distribution Pie Chart
    st.subheader("Attrition Distribution")
    attrition_count = pd.DataFrame(df['Attrition'].value_counts())
    fig1, ax1 = plt.subplots()
    ax1.pie(attrition_count['Attrition'], labels=['No', 'Yes'], explode=(0.2, 0), autopct='%1.1f%%', startangle=90)
    st.pyplot(fig1)

    # Countplot for Attrition
    st.subheader("Countplot of Attrition")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Attrition', ax=ax2)
    st.pyplot(fig2)

    # Drop unnecessary columns
    df.drop(['EmployeeCount', 'EmployeeNumber'], axis=1, inplace=True)

    # Attrition dummies creation
    attrition_dummies = pd.get_dummies(df['Attrition'])
    df = pd.concat([df, attrition_dummies], axis=1)
    df.drop(['Attrition', 'No'], axis=1, inplace=True)

    # Gender vs Attrition
    st.subheader("Gender vs Attrition")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=df, x='Gender', y='Yes', ax=ax3)
    st.pyplot(fig3)

    # Department vs Attrition
    st.subheader("Department vs Attrition")
    fig4, ax4 = plt.subplots()
    sns.barplot(data=df, x='Department', y='Yes', ax=ax4)
    st.pyplot(fig4)

    # Business Travel vs Attrition
    st.subheader("Business Travel vs Attrition")
    fig5, ax5 = plt.subplots()
    sns.barplot(data=df, x='BusinessTravel', y='Yes', ax=ax5)
    st.pyplot(fig5)

    # Heatmap of correlations
    st.subheader("Heatmap of Correlation between Numeric Features")
    plt.figure(figsize=(10, 6))
    fig6, ax6 = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax6)
    st.pyplot(fig6)

# Model Building Section
elif pages == "Model Building":
    st.header("Random Forest Classifier Model")

    # Drop unnecessary features for model training
    df.drop(['Age', 'JobLevel'], axis=1, inplace=True)

    # Label encoding for categorical features
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = LabelEncoder().fit_transform(df[column])

    # Define X and y for model building
    X = df.drop(['Yes'], axis=1)
    y = df['Yes']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)

    # Model training accuracy
    train_score = rf.score(X_train, y_train)
    st.subheader("Training Accuracy")
    st.write(f"Training Accuracy: {train_score:.2f}")

# Results Section
elif pages == "Results":
    st.header("Model Results on Test Data")

    # Predictions
    pred = rf.predict(X_test)

    # Accuracy score
    accuracy = accuracy_score(y_test, pred)
    st.subheader("Test Data Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    st.write(cm)

    # Confusion Matrix Visualization
    fig7, ax7 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax7)
    st.pyplot(fig7)

# Footer
st.sidebar.info("Employee Performance Analysis App using Machine Learning and Data Visualization")
