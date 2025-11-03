#!/usr/bin/env python3

"""
This script builds a simple machine learning model to predict passenger survival
on the RMS Titanic. It reads data from the './inputs' directory, trains a
Logistic Regression model, and saves the predictions for the test set to the
'./outputs' directory.
"""

import os
import sys
import subprocess

# --- Dependency Management ---

def check_and_install_packages():
    """
    Checks if required packages are installed and prompts the user to install
    them if they are not.
    """
    required_packages = ['pandas', 'sklearn']
    missing_packages = []
    try:
        for package in required_packages:
            __import__(package)
    except ImportError as e:
        missing_packages.append(e.name)

    if missing_packages:
        print("Some required libraries are not installed.")
        print(f"Missing: {', '.join(missing_packages)}")
        response = input("Would you like to install them now? (yes/no): ").lower()
        if response in ['yes', 'y']:
            print("Installing required libraries with pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                print("Installation successful. Please re-run the script.")
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Installation failed: {e}")
                print("Please install the libraries manually using: 'pip install pandas scikit-learn'")
                sys.exit(1)
        else:
            print("Installation declined. The program cannot continue.")
            sys.exit(1)

# Run the check before attempting to import the packages for the main script
check_and_install_packages()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- File and Directory Paths ---

INPUT_DIR = 'inputs'
OUTPUT_DIR = 'outputs'
TRAIN_FILE = os.path.join(INPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(INPUT_DIR, 'test.csv')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test_result.csv')

# --- Feature Engineering and Preprocessing ---

def preprocess_data(df):
    """
    Performs preprocessing on the Titanic dataset.
    - Fills missing values for 'Age', 'Fare', and 'Embarked'.
    - Converts categorical features 'Sex' and 'Embarked' to numerical format.
    """
    # Fill missing 'Age' with the median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing 'Fare' with the median (important for the test set)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Fill missing 'Embarked' with the most frequent value (mode)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Convert 'Sex' to a binary feature
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode the 'Embarked' feature
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True)

    return df

# --- Main Program Logic ---

def main():
    """
    Main function to execute the model training and prediction pipeline.
    """
    print("--- Titanic Survival Prediction ---")

    # 1. Load Data
    print(f"Loading training data from '{TRAIN_FILE}'...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found - {e.filename}")
        print("Please ensure 'inputs/train.csv' and 'inputs/test.csv' exist.")
        sys.exit(1)

    # Store PassengerId for the final output file
    test_passenger_ids = test_df['PassengerId']

    # 2. Preprocess Data
    print("Preprocessing data (handling missing values, encoding features)...")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # 3. Define Features and Target
    # Select features that are known to be predictive and are now in numerical format.
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    target = 'Survived'

    # Align columns between train and test sets to ensure consistency
    # This handles cases where a category might be present in train but not test
    train_labels = train_df[target]
    train_ids = train_df['PassengerId'] # Not used for training, just for alignment
    
    # Drop columns that are not features
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
    train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
    test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns and col != 'Survived'])

    # Align columns after one-hot encoding
    train_cols = train_df.columns
    test_cols = test_df.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        test_df[c] = 0
        
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        train_df[c] = 0
        
    test_df = test_df[train_cols] # Ensure order is the same

    X = train_df
    y = train_labels
    X_test = test_df

    # 4. Train and Evaluate Model
    # Split data for validation to get a sense of model performance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training a Logistic Regression model...")
    # Note: For this small dataset and simple model, GPU is not necessary or used.
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    # 5. Display Model Summary
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    
    coefficients = pd.DataFrame(model.coef_[0], X_train.columns, columns=['Coefficient'])
    
    print("\n--- Model Summary ---")
    print("Model Type: Logistic Regression")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nFeature Coefficients (Importance):")
    print(coefficients.sort_values(by='Coefficient', ascending=False))
    print("\n(Positive coefficients increase survival chance, negative decrease it)")
    print("-----------------------\n")

    # 6. Make Predictions on Test Set
    print(f"Predicting survival for {len(X_test)} passengers in the test set...")
    # Retrain the model on the full training data for best performance
    model.fit(X, y)
    test_predictions = model.predict(X_test)

    # 7. Save Results
    output_df = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': test_predictions
    })

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving prediction results to '{OUTPUT_FILE}'...")
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\nProcess complete. The output file has been generated successfully.")


if __name__ == "__main__":
    main()
