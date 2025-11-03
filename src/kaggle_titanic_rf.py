# titanic_survival_predictor.py
# This program builds a machine learning model to predict passenger survival on the RMS Titanic.
# It reads training data, preprocesses features, trains a RandomForestClassifier,
# evaluates it, and then uses the trained model to predict survival for a separate test set,
# saving the results to a CSV file.

import os
import sys

# --- Dependency Checker ---
# A helper function to ensure required libraries are installed.
def check_and_install_libraries():
    """Checks for required libraries and prompts for installation if they are missing."""
    required_libraries = {
        "pandas": "pandas",
        "sklearn": "scikit-learn"
    }
    missing = []
    for lib_import, lib_install in required_libraries.items():
        try:
            __import__(lib_import)
        except ImportError:
            missing.append(lib_install)
    
    if missing:
        print("Error: The following required libraries are not installed:")
        for lib in missing:
            print(f"- {lib}")
        print("\nPlease install them by running the following command:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)

# --- Main Program Logic ---
def main():
    """Main function to run the data processing, model training, and prediction."""
    
    # --- 1. Setup and Data Loading ---
    print("--- Titanic Survival Prediction ---")
    
    # Define file paths
    INPUT_DIR = "inputs"
    OUTPUT_DIR = "outputs"
    TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
    TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_result.csv")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[1/6] Loading data from '{TRAIN_FILE}' and '{TEST_FILE}'...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error: Input file not found at {e.filename}.")
        print("Please ensure 'inputs/train.csv' and 'inputs/test.csv' exist.")
        sys.exit(1)

    # --- 2. Feature Engineering and Preprocessing ---
    print("[2/6] Preprocessing data and engineering features...")

    # We will use a combination of numerical and categorical features.
    # Dropping 'Name', 'Ticket', and 'Cabin' for simplicity, as they require more complex processing.
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    X_train = train_df[features]
    y_train = train_df['Survived']
    X_test = test_df[features]

    # Define preprocessing steps for different column types
    # Using pipelines is a best practice to prevent data leakage and simplify code.
    
    # For numerical columns: fill missing values (e.g., Age, Fare) with the median.
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    numerical_transformer = SimpleImputer(strategy='median')

    # For categorical columns: fill missing values with the most frequent value and then one-hot encode.
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 3. Model Selection and Pipeline Creation ---
    print("[3/6] Defining the model...")
    
    # We'll use a RandomForestClassifier, which is a robust and effective model for this type of problem.
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=2)

    # Create the full pipeline by chaining the preprocessor and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- 4. Model Training and Evaluation ---
    print("[4/6] Training the model on the training data...")
    pipeline.fit(X_train, y_train)
    
    print("[5/6] Evaluating model performance using cross-validation...")
    # Use cross-validation on the training set to get a more robust performance estimate.
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # --- Model Summary ---
    print("\n--- Model Summary ---")
    print(f"Model Type: RandomForestClassifier")
    print(f"Features Used: {features}")
    print("\nValidation Performance:")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Display feature importances for interpretability
    try:
        # Get feature names after one-hot encoding
        ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)
        
        importances = pipeline.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(5)
        
        print("\nTop 5 Feature Importances:")
        print(feature_importance_df.to_string(index=False))
    except Exception:
        print("\nCould not retrieve feature importances.")
    print("---------------------\n")

    # --- 5. Prediction on Test Set ---
    print("[6/6] Making predictions on the test set...")
    predictions = pipeline.predict(X_test)

    # --- 6. Output Generation ---
    output_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Predictions saved to '{OUTPUT_FILE}'.")


if __name__ == '__main__':
    # First, ensure necessary libraries are available
    check_and_install_libraries()
    
    # Import libraries after checking they exist
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # Run the main program
    main()