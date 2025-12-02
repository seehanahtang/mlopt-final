import sys
import os

# --- 0. PATH SETUP ---
# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root by going up one level
project_root = os.path.dirname(script_dir)

# Add project root to sys.path so Python can find 'utils'
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import numpy as np
import pandas as pd
import opticl
from pyomo.environ import ConcreteModel

# --- 1. Imports ---
from interpretableai import iai
from utils.opticl_adapter import make_opticl_compatible, detect_model_type

def generate_dummy_data(iai_model, model_path, n_samples=100, embedding_method='bert'):
    """
    Loads real training data with actual feature names and targets properly excluded.
    """
    try:
        # Try to load actual training data from clean_data
        embedding_suffix = embedding_method.lower()
        train_file = os.path.join(project_root, "clean_data", f"train_{embedding_suffix}.csv")
        
        if os.path.exists(train_file):
            print(f"Loading real training data from {train_file}...")
            df = pd.read_csv(train_file, nrows=n_samples)
            
            # Identify and exclude BOTH target columns
            target_cols = {'clicks', 'conversion', 'conversions', 'Conv. value', 'Clicks'}
            feature_cols = [c for c in df.columns if c.lower() not in target_cols]
            
            X = df[feature_cols].copy()
            
            # Get y from either 'Clicks' or 'Conv. value'
            if 'Clicks' in df.columns:
                y = df['Clicks'].values
            elif 'Conv. value' in df.columns:
                y = df['Conv. value'].values
            else:
                y = np.random.rand(len(X))
            
            print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
            print(f"Features: {list(X.columns)}")
            print(f"Target shape: {y.shape}")
            return X, y
        else:
            print(f"Training data not found at {train_file}, generating dummy data...")
    except Exception as e:
        print(f"Error loading training data: {e}, generating dummy data...")
    
    # Fallback: Generate dummy data if real data not available
    try:
        # Accessing internal structure of IAI linear model
        if hasattr(iai_model, 'get_learners'):
            params = iai_model.get_learners()[0].get_prediction_weights()
        else:
            params = iai_model.get_prediction_weights()
        
        if isinstance(params, tuple):
            # Count weights from both continuous and categorical
            continuous_weights, categorical_weights = params
            n_features = len(continuous_weights) + sum(len(v) for v in categorical_weights.values())
        else:
            n_features = len(params)
    except:
        # Fallback: Read JSON
        with open(model_path, 'r') as f:
            raw_data = json.load(f)
        n_features = raw_data['fits_'][0]['beta']['n']

    print(f"Detected {n_features} features. Generating dummy data...")
    
    X = pd.DataFrame(np.random.rand(n_samples, n_features))
    X.columns = [f"feature_{i}" for i in range(n_features)]
    y = np.random.rand(n_samples)
    
    return X, y

def main():
    # --- Configuration ---
    model_filename = "lr_bert_conversion.json"
    embedding_method = model_filename.split('_')[1]  # Extract 'bert' from 'lr_bert_conversion.json'
    model_path = os.path.join(project_root, "models", model_filename)
    
    # --- 2. Load the IAI Model ---
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file at: {model_path}")

    iai_model = iai.read_json(model_path)

    # --- 3. Detect Model Type ---
    model_type = detect_model_type(iai_model)
    print(f"Detected Model Type: {model_type}")

    # --- 4. Prepare Data ---
    X_train, y_train = generate_dummy_data(iai_model, model_path, embedding_method=embedding_method)

    # --- 6. Wrap Model for OptiCL Compatibility ---
    print("Wrapping model for OptiCL compatibility...")
    compatible_model = make_opticl_compatible(iai_model, feature_names=X_train.columns.tolist())

    # --- 7. Initialize Constraint Learning ---
    print("Initializing OptiCL ConstraintLearning...")
    constraintL = opticl.ConstraintLearning(X_train, y_train, compatible_model, model_type)

    # --- 8. Generate Constraints ---
    print("Extrapolating constraints...")
    constraint_add = constraintL.constraint_extrapolation('continuous')

    # --- 9. Save Results ---
    # We include the model type in the filename for clarity
    output_filename = f"learned_constraints_{model_type}.csv"
    output_file = os.path.join(script_dir, output_filename)
    
    print(f"Saving constraints to {output_file}...")
    constraint_add.to_csv(output_file, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()