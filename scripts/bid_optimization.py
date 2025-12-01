"""
Bid Optimization with Linear Programming
==========================================
Optimizes keyword bids using Gurobi linear programming solver.
Maximizes profit by setting optimal bids for keywords across regions and match types.

Requires:
- Gurobi solver and license: https://www.gurobi.com/
- gurobipy: pip install gurobipy
- Pre-trained models (lr_conversion.json, lr_clicks.json)

Usage:
    python bid_optimization.py --budget 68096.51 --max-bid 100
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Check for required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not found. Install with: pip install gurobipy")
    print("Note: Gurobi requires a valid license.")
    sys.exit(1)

try:
    import iai
except ImportError:
    print("ERROR: IAI library not found. Install with: pip install iai")
    sys.exit(1)


def load_embeddings(embedding_file='raw_data/unique_keyword_embeddings.csv'):
    """Load keyword embeddings."""
    print(f"Loading keyword embeddings from {embedding_file}...")
    keyword_df = pd.read_csv(embedding_file)
    print(f"  Loaded {len(keyword_df)} keywords")
    return keyword_df


def load_models(conversion_model='models/lr_conversion.json', clicks_model='models/lr_clicks.json'):
    """Load pre-trained prediction models."""
    print(f"Loading models...")
    
    lnr_conv = iai.read_json(conversion_model)
    lnr_clicks = iai.read_json(clicks_model)
    
    print(f"  Loaded conversion model from {conversion_model}")
    print(f"  Loaded clicks model from {clicks_model}")
    
    return lnr_conv, lnr_clicks


def extract_weights(lnr_conv, lnr_clicks, n_tfidf=50):
    """Extract weights from trained models."""
    print(f"Extracting model weights...")
    
    # Conversion weights
    weights_conv = iai.get_prediction_weights(lnr_conv)[0]
    conv_tfidf_weights = np.zeros(n_tfidf)
    conv_cpc_weight = 0.0
    conv_const = 0.0
    
    for k, v in weights_conv.items():
        if str(k).startswith('tfidf_'):
            i = int(str(k).split('_')[1])
            conv_tfidf_weights[i] = v
        elif str(k) == 'Avg. CPC':
            conv_cpc_weight = v
        elif str(k) == '(Intercept)':
            conv_const = v
    
    # Clicks weights
    weights_clicks = iai.get_prediction_weights(lnr_clicks)[0]
    clicks_tfidf_weights = np.zeros(n_tfidf)
    clicks_cpc_weight = 0.0
    clicks_const = 0.0
    
    for k, v in weights_clicks.items():
        if str(k).startswith('tfidf_'):
            i = int(str(k).split('_')[1])
            clicks_tfidf_weights[i] = v
        elif str(k) == 'Avg. CPC':
            clicks_cpc_weight = v
        elif str(k) == '(Intercept)':
            clicks_const = v
    
    print(f"  Conversion weights: const={conv_const:.4f}, cpc_weight={conv_cpc_weight:.4f}")
    print(f"  Clicks weights: const={clicks_const:.4f}, cpc_weight={clicks_cpc_weight:.4f}")
    
    return (conv_const, conv_cpc_weight, conv_tfidf_weights), \
           (clicks_const, clicks_cpc_weight, clicks_tfidf_weights)


def create_feature_matrix(keyword_df, n_months=12, regions=None, match_types=None):
    """Create feature matrix for all keyword-month-region-match combinations."""
    if regions is None:
        regions = ["USA", "Region_A", "Region_B", "Region_C"]
    if match_types is None:
        match_types = ["broad match", "exact match", "phrase match"]
    
    num_keywords = len(keyword_df)
    n_combos = num_keywords * n_months * len(regions) * len(match_types)
    
    print(f"Creating feature matrix...")
    print(f"  Keywords: {num_keywords}, Months: {n_months}, Regions: {len(regions)}, Matches: {len(match_types)}")
    print(f"  Total combinations: {n_combos}")
    
    # Build combinations
    data = []
    keyword_idx_list = []
    month_list = []
    region_list = []
    match_list = []
    
    for kw_idx in range(num_keywords):
        for month in range(1, n_months + 1):
            for region in regions:
                for match_type in match_types:
                    keyword_idx_list.append(kw_idx)
                    month_list.append(month)
                    region_list.append(region)
                    match_list.append(match_type)
    
    # Create DataFrame with features
    result = pd.DataFrame({
        'month': month_list,
        'region': region_list,
        'match': match_list,
    })
    
    # Add embeddings
    embedding_cols = [col for col in keyword_df.columns if col.startswith('tfidf_')]
    for col in embedding_cols:
        result[col] = keyword_df.iloc[keyword_idx_list][col].reset_index(drop=True).values
    
    # Reset index for consistency
    result.reset_index(drop=True, inplace=True)
    
    return result, keyword_idx_list, month_list, region_list, match_list


def optimize_bids(X, conv_weights, clicks_weights, budget=68096.51, max_bid=100, n_active=14229):
    """Solve bid optimization problem using Gurobi."""
    print(f"\nSolving bid optimization problem...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Max bid: ${max_bid:.2f}")
    print(f"  Max active keywords: {n_active}")
    
    n = len(X)
    conv_const, conv_cpc_weight, conv_tfidf_weights = conv_weights
    clicks_const, clicks_cpc_weight, clicks_tfidf_weights = clicks_weights
    
    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', 1)  # Show solver output
    
    # Decision variables
    b = model.addMVar(shape=n, lb=0, name='bid')  # bid amounts
    z = model.addMVar(shape=n, vtype=GRB.BINARY, name='active')  # binary active indicator
    
    # Feature matrix as numpy array
    X_arr = X.values
    
    # Objective: Maximize profit
    # profit = conversion_value - clicks * bid
    conv_predictions = X_arr @ conv_tfidf_weights + conv_cpc_weight * b + conv_const
    clicks_predictions = X_arr @ clicks_tfidf_weights + clicks_cpc_weight * b + clicks_const
    
    profit = gp.quicksum(
        conv_predictions[i] - clicks_predictions[i] * b[i] 
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


def extract_solution(model, b, z, keyword_df, keyword_idx_list, month_list, region_list, match_list):
    """Extract non-zero bids from solution."""
    b_vals = b.X
    z_vals = z.X
    
    # Find active keywords
    active_idx = np.where(z_vals > 0.5)[0]  # Binary threshold
    
    print(f"\nSolution Summary:")
    print(f"  Active keywords: {len(active_idx)}")
    print(f"  Total spend: ${np.sum(b_vals):,.2f}")
    print(f"  Predicted profit: ${model.objVal:,.2f}")
    
    # Build result DataFrame
    bids_df = pd.DataFrame({
        'bid': b_vals[active_idx],
        'keyword': [keyword_df.iloc[keyword_idx_list[i]]['Keyword'] for i in active_idx],
        'month': [month_list[i] for i in active_idx],
        'region': [region_list[i] for i in active_idx],
        'match': [match_list[i] for i in active_idx],
    })
    
    # Sort by bid (descending)
    bids_df = bids_df.sort_values('bid', ascending=False).reset_index(drop=True)
    
    return bids_df


def main():
    parser = argparse.ArgumentParser(
        description="Optimize keyword bids using linear programming."
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=68096.51,
        help='Total budget for bids (default: 68096.51)'
    )
    parser.add_argument(
        '--max-bid',
        type=float,
        default=100.0,
        help='Maximum individual bid (default: 100.0)'
    )
    parser.add_argument(
        '--max-active',
        type=int,
        default=14229,
        help='Maximum active keywords (default: 14229)'
    )
    parser.add_argument(
        '--embedding-file',
        type=str,
        default='unique_keyword_embeddings.csv',
        help='Keyword embeddings file'
    )
    parser.add_argument(
        '--conv-model',
        type=str,
        default='models/lr_conversion.json',
        help='Path to conversion model'
    )
    parser.add_argument(
        '--clicks-model',
        type=str,
        default='models/lr_clicks.json',
        help='Path to clicks model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='optimized_bids.csv',
        help='Output CSV file for optimized bids'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bid Optimization with Linear Programming")
    print("=" * 70)
    
    try:
        # Load data and models
        keyword_df = load_embeddings(args.embedding_file)
        lnr_conv, lnr_clicks = load_models(args.conv_model, args.clicks_model)
        conv_weights, clicks_weights = extract_weights(lnr_conv, lnr_clicks)
        
        # Create feature matrix
        X, kw_idx_list, month_list, region_list, match_list = create_feature_matrix(keyword_df)
        
        # Extract embedding columns for weight calculation
        embedding_cols = [col for col in X.columns if col.startswith('tfidf_')]
        X_embeddings = X[embedding_cols]
        
        # Update weights to only include embeddings
        conv_const, conv_cpc_weight, conv_tfidf_weights = conv_weights
        clicks_const, clicks_cpc_weight, clicks_tfidf_weights = clicks_weights
        
        conv_weights_updated = (conv_const, conv_cpc_weight, conv_tfidf_weights)
        clicks_weights_updated = (clicks_const, clicks_cpc_weight, clicks_tfidf_weights)
        
        # Optimize
        model, b, z = optimize_bids(
            X_embeddings,
            conv_weights_updated,
            clicks_weights_updated,
            budget=args.budget,
            max_bid=args.max_bid,
            n_active=args.max_active
        )
        
        # Extract solution
        if model.status == 2:  # OPTIMAL
            bids_df = extract_solution(model, b, z, keyword_df, kw_idx_list, month_list, region_list, match_list)
            
            # Save results
            bids_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
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
