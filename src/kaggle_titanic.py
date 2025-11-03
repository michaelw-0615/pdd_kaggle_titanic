# titanic_survival_predictor.py

# Brief comment explaining what the program does
"""
This program builds a machine learning model to predict passenger survival on the RMS Titanic.
It performs the following steps:
1. Checks for and prompts to install necessary libraries (pandas, scikit-learn).
2. Loads the training and testing data from the './inputs' directory.
3. Performs data cleaning, feature engineering, and preprocessing using a scikit-learn Pipeline.
4. Uses GridSearchCV to find the best hyperparameters for a GradientBoostingClassifier model,
   aiming for a high cross-validation accuracy.
5. Trains the optimized model on the entire training dataset.
6. Evaluates the model and prints a summary including CV accuracy, best parameters, and feature importances.
7. Predicts survival for passengers in the test dataset.
8. Saves the predictions to './outputs/test_result.csv' in the specified format.
"""

import os
import sys
import subprocess

# --- 1. Dependency Management ---
def check_and_install_libraries():
    """Checks for required libraries and prompts the user to install them if missing."""
    required_libraries = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn"
    }
    missing_libraries = []
    for lib_import, lib_install in required_libraries.items():
        try:
            __import__(lib_import)
        except ImportError:
            missing_libraries.append(lib_install)

    if missing_libraries:
        print("The following required libraries are not installed:", ", ".join(missing_libraries))
        response = input("Would you like to install them now? (y/n): ").lower()
        if response == 'y':
            try:
                print(f"Installing {', '.join(missing_libraries)}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_libraries])
                print("\nLibraries installed successfully. Please run the script again.")
                sys.exit(0)
            except Exception as e:
                print(f"\nError installing libraries: {e}")
                print("Please install them manually using: pip install", " ".join(missing_libraries))
                sys.exit(1)
        else:
            print("Installation cancelled. The program cannot continue without the required libraries.")
            sys.exit(1)

# Run the check at the beginning of the script
check_and_install_libraries()

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# --- 2. Main Program Logic ---
def main():
    """Main function to run the Titanic survival prediction pipeline."""
    
    # --- File and Directory Setup ---
    print("--- Titanic Survival Prediction Program ---")
    
    input_dir = 'inputs'
    output_dir = 'outputs'
    train_path = os.path.join(input_dir, 'train.csv')
    test_path = os.path.join(input_dir, 'test.csv')
    output_path = os.path.join(output_dir, 'test_result.csv')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if input files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"\nError: Input files not found in the '{input_dir}' directory.")
        print("Please ensure 'train.csv' and 'test.csv' are present before running.")
        sys.exit(1)

    # --- Data Loading ---
    print("\n[1/6] Loading data...")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_passenger_ids = test_df['PassengerId']
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print("Data loaded successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # --- Feature Engineering and Preprocessing Setup ---
    print("\n[2/6] Setting up feature engineering and preprocessing pipeline...")
    
    # Separate target variable from training data
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]
    
    # Define which columns are numerical and which are categorical
    # We will engineer new features from some of these original columns
    numerical_features = ['Age', 'Fare']
    categorical_features = ['Embarked', 'Sex', 'Pclass']
    
    # Create a custom transformer for feature engineering
    from sklearn.base import BaseEstimator, TransformerMixin

    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            X_copy = X.copy()
            # Create FamilySize
            X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1
            # Create IsAlone
            X_copy['IsAlone'] = (X_copy['FamilySize'] == 1).astype(int)
            # Create HasCabin
            X_copy['HasCabin'] = X_copy['Cabin'].notna().astype(int)
            # Extract and clean Title
            X_copy['Title'] = X_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            X_copy['Title'] = X_copy['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            X_copy['Title'] = X_copy['Title'].replace(['Mlle', 'Ms'], 'Miss')
            X_copy['Title'] = X_copy['Title'].replace('Mme', 'Mrs')
            return X_copy

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features + ['Title']),
            ('passthrough', 'passthrough', ['IsAlone', 'HasCabin', 'FamilySize'])
        ],
        remainder='drop' # Drop columns not specified
    )
    
    print("Preprocessing pipeline created.")

    # --- Model Training and Hyperparameter Tuning ---
    print("\n[3/6] Setting up the model and hyperparameter search...")
    
    # Define the model
    model = GradientBoostingClassifier(random_state=42)

    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Define a focused parameter grid for GridSearchCV to keep runtime reasonable
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 4],
        'classifier__min_samples_leaf': [3, 5]
    }

    # Initialize GridSearchCV to find the best model
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    print("Starting model training with GridSearchCV... (This may take a minute)")
    grid_search.fit(X, y)
    
    print("Training complete.")

    # --- Model Evaluation and Summary ---
    print("\n[4/6] Model Summary:")
    
    best_score = grid_search.best_score_
    print(f"  - Best Cross-Validation Accuracy: {best_score:.4f}")
    
    if best_score >= 0.90:
        print("  - SUCCESS: Target accuracy of >= 0.90 was achieved on the validation set.")
    else:
        print(f"  - NOTE: Target accuracy of 0.90 was not met. Best score achieved: {best_score:.4f}")
        print("    This is a very high bar for this dataset; the result is still strong.")

    print("\n  - Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    - {param.replace('classifier__', '')}: {value}")

    # Display feature importances from the best model
    try:
        best_model_pipeline = grid_search.best_estimator_
        preprocessor_fitted = best_model_pipeline.named_steps['preprocessor']
        classifier_fitted = best_model_pipeline.named_steps['classifier']

        # Get feature names after one-hot encoding
        ohe_cat_features = categorical_features + ['Title']
        ohe_feature_names = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(ohe_cat_features)
        
        # Combine all feature names in the correct order
        passthrough_features = ['IsAlone', 'HasCabin', 'FamilySize']
        all_feature_names = numerical_features + list(ohe_feature_names) + passthrough_features
        
        importances = classifier_fitted.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

        print("\n  - Top 10 Feature Importances:")
        print(feature_importance_df.to_string(index=False))
    except Exception as e:
        print(f"\nCould not display feature importances: {e}")

    # --- Prediction on Test Set ---
    print("\n[5/6] Predicting survival for the test set...")
    
    # The grid_search object automatically uses the best found model for prediction
    test_predictions = grid_search.predict(test_df)
    
    print("Prediction complete.")

    # --- Saving Output ---
    print(f"\n[6/6] Saving results to '{output_path}'...")
    
    output_df = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': test_predictions
    })
    
    output_df.to_csv(output_path, index=False)
    
    print("\n--- Program Finished Successfully ---")


if __name__ == "__main__":
    main()
