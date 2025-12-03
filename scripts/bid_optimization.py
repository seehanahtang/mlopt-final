"""
Bid Optimization with Linear Programming
==========================================
Optimizes keyword bids using Gurobi linear programming solver.
Maximizes profit by setting optimal bids for keywords across regions and match types.

Requires:
- Gurobi solver and license: https://www.gurobi.com/
- gurobipy: pip install gurobipy
- Pre-trained models (lr_{embedding}_conversion.json, lr_{embedding}_clicks.json)
- Embeddings file will be auto-generated if missing

Usage:
    python bid_optimization.py --embedding-method bert --budget 68096.51 --max-bid 100
    python bid_optimization.py --embedding-method tfidf --budget 68096.51
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_embeddings

# Check for required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not found. Install with: pip install gurobipy")
    print("Note: Gurobi requires a valid license.")
    sys.exit(1)

try:
    from utils.iai_setup import iai
except ImportError:
    print("ERROR: Could not set up IAI. Install with: pip install iai")
    print("Note: IAI requires a valid license.")
    sys.exit(1)


def load_embeddings_data(keywords, embedding_method='bert', output_dir='data'):
    """Load embeddings from file."""
    embeddings_file = Path(output_dir) / f'unique_keyword_embeddings_{embedding_method}.csv'
    
    # Use the utility function to load embeddings
    embedding_df = load_embeddings(
        embeddings_file,
        embedding_method=embedding_method,
        keywords=keywords
    )
    
    return embedding_df


def load_models(embedding_method='bert', alg_conv='lr', alg_clicks='lr', models_dir='models'):
    """Load pre-trained prediction models based on embedding method and algorithm type.
    
    Args:
        embedding_method: 'bert' or 'tfidf'
        alg_conv: algorithm type for conversion model - 'lr' (linear regression), 'ort' (optimal regression tree), 
                  'rf' (random forest), 'xgb' (xgboost)
        alg_clicks: algorithm type for clicks model - same options as alg_conv
        models_dir: directory containing model files
    
    Returns:
        tuple of (conversion_model, clicks_model)
    """
    print(f"Loading models for embedding method '{embedding_method}'...")
    print(f"  Conversion model: {alg_conv}")
    print(f"  Clicks model: {alg_clicks}")
    
    conversion_model = Path(models_dir) / f'{alg_conv}_{embedding_method}_conversion.json'
    clicks_model = Path(models_dir) / f'{alg_clicks}_{embedding_method}_clicks.json'
    
    if not conversion_model.exists():
        raise FileNotFoundError(f"Conversion model not found: {conversion_model}")
    if not clicks_model.exists():
        raise FileNotFoundError(f"Clicks model not found: {clicks_model}")
    
    lnr_conv = iai.read_json(str(conversion_model))
    lnr_clicks = iai.read_json(str(clicks_model))
    
    print(f"  Loaded conversion model from {conversion_model}")
    print(f"  Loaded clicks model from {clicks_model}")
    
    return lnr_conv, lnr_clicks


def extract_weights(lnr_conv, lnr_clicks, embedding_method='bert', n_embeddings=50):
    """Extract ALL weights from trained models (not just embeddings)."""
    print(f"Extracting model weights...")
    
    # Get weights using IAI's method
    weights_conv_tuple = lnr_conv.get_prediction_weights()
    weights_clicks_tuple = lnr_clicks.get_prediction_weights()
    
    # Handle tuple format (continuous, categorical)
    if isinstance(weights_conv_tuple, tuple):
        weights_conv = weights_conv_tuple[0]
    else:
        weights_conv = weights_conv_tuple
    
    if isinstance(weights_clicks_tuple, tuple):
        weights_clicks = weights_clicks_tuple[0]
    else:
        weights_clicks = weights_clicks_tuple
    
    conv_const = lnr_conv.get_prediction_constant()
    clicks_const = lnr_clicks.get_prediction_constant()
    
    print(f"\n  Conversion model weights:")
    for key, val in sorted(weights_conv.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    print(f"\n  Clicks model weights:")
    for key, val in sorted(weights_clicks.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    return {
        'conv_const': conv_const,
        'conv_weights': weights_conv,
        'clicks_const': clicks_const,
        'clicks_weights': weights_clicks
    }


def create_feature_matrix(keyword_df, embedding_method='bert', target_day=None, regions=None, match_types=None, training_data_path='clean_data/ad_opt_data_bert.csv', weights_dict=None):
    """Create feature matrix for all keyword-region-match combinations for a specific day.
    
    Merges on exact keyword-match type-region combinations from historical data.
    If target_day is None, uses the latest date in data and adjusts date features for today.
    
    Includes:
    - Embeddings from keyword_df
    - Region
    - Match type
    - All historical features (Avg. CPC, Competition, etc.)
    - Date-adjusted features (day_of_week, month, days_to_next_course_start, is_public_holiday, is_weekend)
    
    Args:
        keyword_df: DataFrame with keywords and embeddings
        embedding_method: 'bert' or 'tfidf'
        target_day: str, date in format 'YYYY-MM-DD' (e.g., '2024-11-04'). If None, uses latest date.
        regions: list of regions (default: ["USA", "Region_A", "Region_B", "Region_C"])
        match_types: list of match types (default: ["broad match", "exact match", "phrase match"])
        training_data_path: path to training data file
        weights_dict: dict with 'conv_weights' and 'clicks_weights' to filter features. If None, keeps all.
    """
    if regions is None:
        regions = ["USA", "A", "B", "C"]
    if match_types is None:
        match_types = ["Broad match", "Exact match", "Phrase match"]
    
    # Extract model weights if provided
    if weights_dict is None:
        weights_dict = {}
    conv_weights = weights_dict.get('conv_weights', {})
    clicks_weights = weights_dict.get('clicks_weights', {})
    
    num_keywords = len(keyword_df)
    n_combos = num_keywords * len(regions) * len(match_types)
    
    print(f"Creating feature matrix...")
    print(f"  Target day: {target_day}")
    print(f"  Keywords: {num_keywords}, Regions: {len(regions)}, Matches: {len(match_types)}")
    print(f"  Total combinations: {n_combos}")
    
    # Load training data
    print(f"  Loading training data from {training_data_path}...")
    training_data = pd.read_csv(training_data_path)
    training_data['Day'] = pd.to_datetime(training_data['Day'])
    
    # Determine which date to use
    if target_day is not None:
        if isinstance(target_day, str):
            target_day = pd.to_datetime(target_day)
        filter_day = target_day
        print(f"  Using target day: {target_day.date()}")
    else:
        # Use latest date in data
        filter_day = training_data['Day'].max()
        print(f"  Using latest date in data: {filter_day.date()}")
    
    # Build all combinations we need
    combinations = []
    for kw in keyword_df['Keyword']:
        for region in regions:
            for match in match_types:
                combinations.append({
                    'Keyword': kw,
                    'Region': region,
                    'Match type': match,
                })
    
    combo_df = pd.DataFrame(combinations)
    combo_df['Day'] = filter_day
    print(f"  Created {len(combo_df)} keyword-region-match combinations")
    
    # Use merge_asof to match on (Keyword, Match type, Region) with nearest date
    print(f"  Merging on exact (Keyword, Match type, Region) with nearest date...")
    
    # Merge using asof on date, exact match on categorical columns
    result = pd.merge_asof(
        combo_df.sort_values('Day'),
        training_data.sort_values('Day'),
        on='Day',
        by=['Keyword', 'Match type', 'Region'],
        direction='nearest'
    )
    
    # Drop combinations not found in training data (will have NaN in feature columns)
    initial_rows = len(combo_df)
    result = result.dropna(subset=['Avg. CPC'])
    final_rows = len(result)
    dropped_rows = initial_rows - final_rows
    
    print(f"  Matched {final_rows} combinations with features (nearest date), {dropped_rows} combinations not found in data and dropped")
    
    # If target_day != latest date in data, we need to adjust date features
    if target_day is None:
        from utils.date_features import calculate_date_features
        
        target_date = pd.to_datetime(datetime.today())
        date_features = calculate_date_features(target_date, regions=regions)
        
        print(f"  Adjusting date features to today ({target_date.date()})...")
        result['day_of_week'] = date_features['day_of_week']
        result['is_weekend'] = date_features['is_weekend']
        result['month'] = date_features['month']
        result['is_public_holiday'] = date_features['is_public_holiday']
        result['days_to_next_course_start'] = date_features['days_to_next_course_start']
        
        print(f"  Adjusted temporal features for {target_date.date()}")
    else:
        print(f"  Using date features from {filter_day.date()}")
    
    # Extract the order information for rows that actually made it into the feature matrix
    # (BEFORE dropping Day/Keyword columns so we still have access to them)
    keyword_idx_list = []
    region_list = []
    match_list = []
    
    for _, row in result.iterrows():
        kw_idx = keyword_df[keyword_df['Keyword'] == row['Keyword']].index[0]
        keyword_idx_list.append(kw_idx)
        region_list.append(row['Region'])
        match_list.append(row['Match type'])
    
    # Drop day column
    result = result.drop(columns=['Day', 'Keyword'])

    # Reset index
    result.reset_index(drop=True, inplace=True)
    
    # Convert column names: replace dots with underscores to match model feature names
    result.columns = result.columns.str.replace('.', '_', regex=False)
    
    # One-hot encode categorical columns (if any)
    categorical_cols = result.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"  One-hot encoding categorical columns: {categorical_cols}")
        result = pd.get_dummies(result, columns=categorical_cols, drop_first=False)
    
    # Convert all columns to float for numeric operations
    result = result.astype(float)
    
    print(f"  Final feature matrix shape: {result.shape}")
    print(f"  Columns: {result.columns.tolist()[:15]}...")
    
    return result, keyword_idx_list, region_list, match_list


def optimize_bids(X, weights_dict, budget=68096.51, max_bid=100, n_active=14229):
    """Solve bid optimization problem using Gurobi.
    
    Uses all numeric features from the trained model.
    Feature matrix X should already be numeric (one-hot encoded if needed).
    """
    print(f"\nSolving bid optimization problem...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Max bid: ${max_bid:.2f}")
    print(f"  Max active keywords: {n_active}")
    
    n = len(X)
    
    # Extract constants and weights
    conv_const = weights_dict['conv_const']
    conv_weights = weights_dict['conv_weights']
    clicks_const = weights_dict['clicks_const']
    clicks_weights = weights_dict['clicks_weights']
    
    # Feature matrix as numpy array
    X_arr = X.values
    
    # Build prediction vectors by applying all model weights
    conv_predictions = np.full(n, conv_const, dtype=float)
    clicks_predictions = np.full(n, clicks_const, dtype=float)
    
    print(f"\n  Applying conversion model weights...")
    # Apply feature weights for conversion
    for feature_name, weight in conv_weights.items():
        feature_str = str(feature_name).strip()
        
        # Try to find the column in X
        if feature_str in X.columns:
            col_idx = X.columns.get_loc(feature_str)
            col_data = X_arr[:, col_idx].astype(float)
            conv_predictions += weight * col_data
            print(f"    Applied weight for '{feature_str}': {weight:.4f}")
        elif feature_str == 'Avg. CPC' or feature_str == 'Avg_ CPC':
            # CPC is handled separately in objective
            pass
        else:
            # Try to find similar name (handle naming variations)
            matching_cols = [col for col in X.columns if feature_str.lower() in col.lower()]
            if matching_cols:
                print(f"    Note: Feature '{feature_str}' not found exactly, but found similar: {matching_cols}")
    
    print(f"  Applying clicks model weights...")
    for feature_name, weight in clicks_weights.items():
        feature_str = str(feature_name).strip()
        
        if feature_str in X.columns:
            col_idx = X.columns.get_loc(feature_str)
            col_data = X_arr[:, col_idx].astype(float)
            clicks_predictions += weight * col_data
            print(f"    Applied weight for '{feature_str}': {weight:.4f}")
        elif feature_str == 'Avg. CPC' or feature_str == 'Avg_ CPC':
            pass
        else:
            matching_cols = [col for col in X.columns if feature_str.lower() in col.lower()]
            if matching_cols:
                print(f"    Note: Feature '{feature_str}' not found exactly, but found similar: {matching_cols}")
    
    # Get CPC weights if they exist
    conv_cpc_weight = conv_weights.get('Avg. CPC', conv_weights.get('Avg_ CPC', 0.0))
    clicks_cpc_weight = clicks_weights.get('Avg. CPC', clicks_weights.get('Avg_ CPC', 0.0))
    
    print(f"\n  CPC weights: conv={conv_cpc_weight:.4f}, clicks={clicks_cpc_weight:.4f}")
    
    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', 1)  # Show solver output
    
    # Decision variables
    b = model.addMVar(shape=n, lb=0, name='bid')  # bid amounts
    z = model.addMVar(shape=n, vtype=GRB.BINARY, name='active')  # binary active indicator
    
    # Objective: Maximize profit
    # profit = (conv_predictions + conv_cpc_weight * bid) - (clicks_predictions + clicks_cpc_weight * bid) * bid
    # profit = conv_predictions + conv_cpc_weight * bid - clicks_predictions * bid - clicks_cpc_weight * bid^2
    profit = gp.quicksum(
        conv_predictions[i] + conv_cpc_weight * b[i] - (clicks_predictions[i] + clicks_cpc_weight * b[i]) * b[i]
        for i in range(n)
    )
    
    model.setObjective(profit, GRB.MAXIMIZE)
    
    # Constraints
    # Budget constraint
    model.addConstr(gp.quicksum(b) <= budget, name='budget')
    
    # Bid linking constraints (b[i] is only positive if z[i] = 1)
    model.addConstrs((b[i] <= max_bid * z[i] for i in range(n)), name='bid_ub')
    
    # Activity constraint
    model.addConstr(gp.quicksum(z) <= n_active, name='max_active')
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print(f"  Status: OPTIMAL")
        print(f"  Optimal profit: ${model.objVal:,.2f}")
        return model, b, z
    else:
        print(f"  Status: {model.status}")
        return model, b, z

def embed_lr(model, lnr, y, X, b, target):
    """
    Embeds LR constraints with STRICT validation.
    Raises ValueError if X is missing any column required by the weights.
    """
    
    # --- 1. Get Weights & Constant ---
    weights_tuple = lnr.get_prediction_weights()
    
    numeric_weights = {}
    categorical_weights = {}

    if isinstance(weights_tuple, tuple):
        numeric_weights = weights_tuple[0]
        if len(weights_tuple) > 1:
            categorical_weights = weights_tuple[1]
    else:
        numeric_weights = weights_tuple
        
    const = lnr.get_prediction_constant()

    # --- 2. Create Variables for THIS target ---
    indices = [(target, i) for i in range(len(X))]
    new_y = model.addVars(indices, vtype=GRB.CONTINUOUS, name=f'y_{target}', lb=-GRB.INFINITY)
    y.update(new_y)
    
    # --- 3. Add Constraints ---
    for i in range(len(X)):
        
        rhs_expr = const
        
        # A. Numeric Features (Strict Check)
        for feature, coef in numeric_weights.items():
            if feature == 'Avg_ CPC':
                rhs_expr += coef * b[i]
                continue

            elif feature not in X.columns:
                raise ValueError(f"Error: Numeric column '{feature}' is missing from X dataframe.")
            
            else:
                rhs_expr += coef * X.iloc[i][feature]

        # B. Categorical Features (Strict Check)
        for feature, level_dict in categorical_weights.items():
            for level_name, coef in level_dict.items():
                
                # Construct OHE name (Match type_Broad match)
                ohe_col_name = f"{feature}_{level_name}"
                
                if ohe_col_name not in X.columns:
                    raise ValueError(f"Error: One-Hot encoded column '{ohe_col_name}' is missing from X dataframe.")
                
                rhs_expr += coef * X.iloc[i][ohe_col_name]

        model.addConstr(new_y[(target, i)] == rhs_expr, name=f"LR_constr_{target}_{i}")

    return model

def optimize_bids_embedded(X, lnr_conv, lnr_clicks, budget=400, max_bid=50.0, max_active=1000):
    """Solve bid optimization problem using Gurobi with embedded LR constraints.
    
    Embeds linear regression predictions directly into the Gurobi model as constraints,
    then optimizes bid allocations.
    
    Args:
        X: Feature matrix (rows = keyword-region-match combos, columns = features)
        lnr_conv: Trained linear regression model for conversion
        lnr_clicks: Trained linear regression model for clicks
        budget: Total budget for bids
        max_bid: Maximum individual bid
        max_active: Maximum number of active keywords
    
    Returns:
        tuple of (model, b, z, y) where:
        - model: Gurobi model object (solved)
        - b: Bid decision variables (MVar)
        - z: Binary active indicator variables (MVar)
        - y: Tupledict with optimized prediction variables y[('conversion', i)] and y[('clicks', i)]
    """
    print(f"\nSolving bid optimization with embedded LR constraints...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Max bid: ${max_bid:.2f}")
    print(f"  Max active keywords: {max_active}")
    print(f"  Keywords: {len(X)}")
    
    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', 1)  # Show solver output

    y = gp.tupledict()  # Predicted conversion and clicks variables
    b = model.addMVar(shape=len(X), lb=0, ub=max_bid, name='bid')  # bid amounts

    # Embed LR constraints for both targets
    embed_lr(model, lnr_conv, y, X, b, target='conversion')
    embed_lr(model, lnr_clicks, y, X, b, target='clicks')
    
    # Objective: Maximize profit. Sum of y_conversion - y_clicks * bid
    profit = gp.quicksum(
        y[('conversion', i)] - y[('clicks', i)] * b[i]
        for i in range(len(X))
    )
    
    model.setObjective(profit, GRB.MAXIMIZE)
    
    # Constraints
    # Budget constraint
    model.addConstr(gp.quicksum(b) <= budget, name='budget')
    
    model.update()
    model.write("my_model_debug.lp")
    
    # Optimize
    model.setParam("NonConvex", 2)  # Due to y[clicks] * b (if substituted would be quadratic)
    model.optimize()
    
    return model, b, y


def extract_solution(model, b, y, keyword_df, keyword_idx_list, region_list, match_list, X=None, weights_dict=None):
    """Extract non-zero bids from solution with predictions."""
    b_vals = b.X
    
    print(f"\nSolution Summary:")
    print(f"  Total spend: ${np.sum(b_vals):,.2f}")
    print(f"  Predicted profit: ${model.objVal:,.2f}")
    
    # Build result DataFrame
    bids_df = pd.DataFrame({
        'bid': b_vals,
        'keyword': [keyword_df.iloc[keyword_idx_list[i]]['Keyword'] for i in range(len(b_vals))],
        'region': [region_list[i] for i in range(len(b_vals))],
        'match': [match_list[i] for i in range(len(b_vals))],
    })
    
    # Add predictions if y variables are available (from embedded LR)
    if y is not None and len(y) > 0:
        conv_preds = []
        clicks_preds = []
        profits = []
        
        for idx in range(len(b_vals)):
            # Get y values directly from the optimized solution
            conv_val = y[('conversion', idx)].X
            clicks_val = y[('clicks', idx)].X
            bid_val = b_vals[idx]
            
            # Profit is exactly as optimized: conversion - clicks * bid
            profit = conv_val - clicks_val * bid_val
            
            conv_preds.append(conv_val)
            clicks_preds.append(clicks_val)
            profits.append(profit)
        
        bids_df['predicted_conv_value'] = conv_preds
        bids_df['predicted_clicks'] = clicks_preds
        bids_df['predicted_profit'] = profits
    elif X is not None and weights_dict is not None:
        # Fallback: compute from features and weights if y not available
        conv_const = weights_dict['conv_const']
        conv_weights = weights_dict['conv_weights']
        clicks_const = weights_dict['clicks_const']
        clicks_weights = weights_dict['clicks_weights']
        
        # Get CPC weights if they exist
        conv_cpc_weight = conv_weights.get('Avg. CPC', conv_weights.get('Avg_ CPC', 0.0))
        clicks_cpc_weight = clicks_weights.get('Avg. CPC', clicks_weights.get('Avg_ CPC', 0.0))
        
        # Compute predictions for active keywords
        conv_preds = []
        clicks_preds = []
        profits = []
        
        for i, idx in enumerate(range(len(b_vals))):
            bid_val = b_vals[idx]
            
            # Compute base conversion value (features + constant, WITHOUT bid term yet)
            conv_base = conv_const
            for feat_name, weight in conv_weights.items():
                if feat_name not in ['Avg. CPC', 'Avg_ CPC'] and feat_name in X.columns:
                    conv_base += weight * X.loc[idx, feat_name]
            
            # Compute base clicks (features + constant, WITHOUT bid term yet)
            clicks_base = clicks_const
            for feat_name, weight in clicks_weights.items():
                if feat_name not in ['Avg. CPC', 'Avg_ CPC'] and feat_name in X.columns:
                    clicks_base += weight * X.loc[idx, feat_name]
            
            # Now add bid terms properly:
            # y_conv = conv_base + conv_cpc_weight * bid
            # y_clicks = clicks_base + clicks_cpc_weight * bid
            conv_val = conv_base + conv_cpc_weight * bid_val
            clicks_val = clicks_base + clicks_cpc_weight * bid_val
            
            # Profit exactly matches optimizer objective:
            # profit = y_conv - y_clicks * bid
            #        = (conv_base + conv_cpc_weight * bid) - (clicks_base + clicks_cpc_weight * bid) * bid
            #        = conv_base + conv_cpc_weight * bid - clicks_base * bid - clicks_cpc_weight * bid^2
            profit = conv_val - clicks_val * bid_val
            
            conv_preds.append(conv_val)
            clicks_preds.append(clicks_val)
            profits.append(profit)
        
        bids_df['predicted_conv_value'] = conv_preds
        bids_df['predicted_clicks'] = clicks_preds
        bids_df['predicted_profit'] = profits
    
    # Sort by bid (descending)
    bids_df = bids_df.sort_values('bid', ascending=False).reset_index(drop=True)
    
    return bids_df


def main():
    parser = argparse.ArgumentParser(
        description="Optimize keyword bids using linear programming."
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='bert',
        choices=['tfidf', 'bert'],
        help='Embedding method used for models (default: bert)'
    )
    parser.add_argument(
        '--alg-conv',
        type=str,
        default='lr',
        choices=['lr', 'ort', 'rf', 'xgb'],
        help='Algorithm type for conversion model: lr (linear regression), ort (optimal tree), rf (random forest), xgb (xgboost) (default: lr)'
    )
    parser.add_argument(
        '--alg-clicks',
        type=str,
        default='lr',
        choices=['lr', 'ort', 'rf', 'xgb'],
        help='Algorithm type for clicks model: lr (linear regression), ort (optimal tree), rf (random forest), xgb (xgboost) (default: lr)'
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=400,
        help='Total budget for bids (default: 400)'
    )
    parser.add_argument(
        '--max-bid',
        type=float,
        default=50.0,
        help='Maximum individual bid (default: 50.0)'
    )
    parser.add_argument(
        '--max-active',
        type=int,
        default=1000,
        help='Maximum active keywords (default: 1000)'
    )
    parser.add_argument(
        '--target-day',
        type=str,
        default=None,
        help='Target day for optimization in format YYYY-MM-DD (e.g., 2024-11-04). If not provided, uses all data.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='clean_data',
        help='Directory containing embeddings and models (default: clean_data)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bid Optimization with Linear Programming")
    print("=" * 70)
    print(f"Embedding method: {args.embedding_method}")
    print("=" * 70)
    
    try:
        # Load embeddings data
        embeddings_file = Path(args.data_dir) / f'unique_keyword_embeddings_{args.embedding_method}.csv'
        keyword_df = pd.read_csv(str(embeddings_file))
        print(f"Loaded {len(keyword_df)} keywords from {embeddings_file}")
        
        # Load models
        lnr_conv, lnr_clicks = load_models(args.embedding_method, args.alg_conv, args.alg_clicks, args.models_dir)
        
        # Extract weights
        weights_dict = extract_weights(
            lnr_conv, lnr_clicks, 
            embedding_method=args.embedding_method, 
            n_embeddings=50
        )
        
        # Create feature matrix (includes all features and keyword values for target day)
        training_data_file = Path(args.data_dir) / f'ad_opt_data_{args.embedding_method}.csv'
        X, kw_idx_list, region_list, match_list = create_feature_matrix(
            keyword_df, 
            embedding_method=args.embedding_method,
            target_day=args.target_day,
            training_data_path=str(training_data_file),
            weights_dict=weights_dict
        )
        
        print(f"Feature matrix has {X.shape[1]} total features")

        # Optimize using embedded LR constraints
        model, b, y = optimize_bids_embedded(
            X,
            lnr_conv,
            lnr_clicks,
            budget=args.budget,
            max_bid=args.max_bid,
            max_active=args.max_active
        )
        
        # Extract solution
        if model.status == 2:  # OPTIMAL
            output_dir = Path('opt_results')
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f'optimized_bids_{args.embedding_method}.csv'
            
            bids_df = extract_solution(model, b, y, keyword_df, kw_idx_list, region_list, match_list, X=X, weights_dict=weights_dict)
            
            # Save results
            bids_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(f"\nTop 10 bids:")
            print(bids_df.head(10).to_string(index=False))
        else:
            print(f"Optimization failed with status {model.status}")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓ Bid optimization completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
