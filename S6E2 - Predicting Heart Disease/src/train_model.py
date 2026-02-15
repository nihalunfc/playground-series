# ==============================================================================
# PROJECT: Heart Disease Risk Prediction (Kaggle S6E2)
# SCRIPT: train_model.py
# DESCRIPTION: Production script for the Winning Stacking Ensemble (Trial 07)
# AUTHOR: Nihal V S
# ==============================================================================

import pandas as pd
import numpy as np
import os
import warnings

# Machine Learning Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "submissions"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_production.csv")

def load_data():
    """Loads train and test datasets from the data directory."""
    print(f"Loading data from {DATA_DIR}...")
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Data files not found in {DATA_DIR}. Please check paths.")
    
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    print(f" - Train Shape: {train.shape}")
    print(f" - Test Shape: {test.shape}")
    return train, test

def preprocess_data(train, test):
    """
    Performs preprocessing steps:
    1. Target Mapping (Absence/Presence -> 0/1)
    2. Feature Separation
    3. Categorical Encoding
    """
    print("Preprocessing data...")
    
    # 1. Target Mapping
    if 'Heart Disease' in train.columns:
        target_mapping = {'Absence': 0, 'Presence': 1}
        y = train['Heart Disease'].map(target_mapping)
        X = train.drop(['id', 'Heart Disease'], axis=1)
    else:
        raise ValueError("Target column 'Heart Disease' not found in training data.")
        
    X_test = test.drop(['id'], axis=1)

    # 2. Categorical Encoding (Label Encoding)
    # Combining train/test to ensure consistent encoding
    combined = pd.concat([X, X_test], axis=0)
    
    for col in combined.select_dtypes(include=['object']).columns:
        combined[col] = combined[col].astype('category').cat.codes
        
    # Split back into X and X_test
    X = combined.iloc[:len(train)]
    X_test = combined.iloc[len(train):]
    
    print(" - Categorical features encoded.")
    return X, y, X_test

def train_stacking_ensemble(X, y):
    """
    Trains the Trial 07 Stacking Architecture:
    - Level 0: Optimized XGBoost + Standard LightGBM
    - Level 1: Logistic Regression
    """
    print("Initializing Stacking Ensemble...")

    # Base Model 1: Optimized XGBoost (Trial 04 Parameters)
    xgb_model = XGBClassifier(
        n_estimators=3000, 
        learning_rate=0.0524336, 
        max_depth=4, 
        subsample=0.817428, 
        colsample_bytree=0.50969, 
        min_child_weight=10,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    # Base Model 2: Standard LightGBM (Trial 07 Configuration)
    lgb_model = LGBMClassifier(
        n_estimators=1000, 
        learning_rate=0.03, 
        num_leaves=63, 
        verbosity=-1,
        n_jobs=-1,
        random_state=42
    )

    base_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ]

    # Meta-Learner: Logistic Regression
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5, 
        stack_method='predict_proba',
        n_jobs=-1
    )

    print("Training models... (This may take a few minutes)")
    stack_model.fit(X, y)
    print(" - Training Complete.")
    
    return stack_model

def generate_submission(model, X_test, test_ids):
    """Generates predictions and saves the submission file."""
    print("Generating predictions...")
    
    # Predict probabilities for class 1 (Presence)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    submission = pd.DataFrame({'id': test_ids, 'Heart Disease': probs})
    submission.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"✅ Success! Submission file saved to: {SUBMISSION_FILE}")

def main():
    """Main execution flow."""
    try:
        # 1. Load
        train_df, test_df = load_data()
        
        # 2. Preprocess
        X, y, X_test = preprocess_data(train_df, test_df)
        
        # 3. Train
        model = train_stacking_ensemble(X, y)
        
        # 4. Submit
        generate_submission(model, X_test, test_df['id'])
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
