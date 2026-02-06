"""
Preprocessing Pipeline - Run ONCE before training
==================================================
Separates preprocessing from training for faster iteration.

Output:
- data/processed/train.parquet
- data/processed/val.parquet  
- data/processed/test.parquet
- data/processed/preprocessor.pkl (encoders + scalers)
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Feature configuration - UPDATED: seq/ack as NUMERICAL (not categorical)
# This reduces model parameters from 17M to ~200K!
CATEGORICAL_FEATURES = [
    'ip_version', 'ip_proto', 'ip_tos', 'tcp_flags', 'tcp_dataofs'
]

NUMERICAL_FEATURES = [
    # Original numerical
    'ip_len', 'ip_ttl', 'ip_chksum', 'tcp_window', 'udp_len', 'tcp_urg',
    'ip_ihl', 'ip_id', 'ip_flags', 'ip_frag', 'udp_chksum',
    # Split integers as numerical (they're continuous-ish, not discrete categories)
    'tcp_seq_high', 'tcp_seq_low', 'tcp_ack_high', 'tcp_ack_low'
]


def preprocess(df: pd.DataFrame, fit: bool = True, preprocessor: dict = None):
    """
    Preprocess dataframe.
    
    Args:
        df: Input dataframe
        fit: If True, fit encoders/scalers. If False, use provided preprocessor.
        preprocessor: Pre-fitted encoders/scalers (required if fit=False)
    
    Returns:
        Preprocessed dataframe, preprocessor dict
    """
    df = df.copy()
    
    if fit:
        preprocessor = {
            'label_encoders': {},
            'scaler': StandardScaler(),
            'cat_cardinalities': {}
        }
    
    # Fix column naming
    if 'tcp_dataoff' in df.columns and 'tcp_dataofs' not in df.columns:
        df['tcp_dataofs'] = df['tcp_dataoff']
    
    # Replace -1 sentinel with 0
    for col in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace(-1, 0).fillna(0)
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                preprocessor['label_encoders'][col] = le
                preprocessor['cat_cardinalities'][col] = len(le.classes_)
            else:
                le = preprocessor['label_encoders'][col]
                # Handle unseen values
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
    
    # Scale numerical features
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if fit:
        df[num_cols] = preprocessor['scaler'].fit_transform(df[num_cols])
    else:
        df[num_cols] = preprocessor['scaler'].transform(df[num_cols])
    
    return df, preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Total samples: {len(df):,}")
    
    # Split by flow_id to prevent leakage
    logger.info("Splitting by flow_id (no leakage)...")
    unique_flows = df['flow_id'].unique()
    logger.info(f"Unique flows: {len(unique_flows):,}")
    
    # Train/temp split
    train_flows, temp_flows = train_test_split(
        unique_flows, 
        test_size=args.val_size + args.test_size,
        random_state=args.seed
    )
    
    # Val/test split
    val_flows, test_flows = train_test_split(
        temp_flows,
        test_size=args.test_size / (args.val_size + args.test_size),
        random_state=args.seed
    )
    
    train_df = df[df['flow_id'].isin(train_flows)]
    val_df = df[df['flow_id'].isin(val_flows)]
    test_df = df[df['flow_id'].isin(test_flows)]
    
    logger.info(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Preprocess
    logger.info("Preprocessing train set (fitting encoders/scalers)...")
    train_df, preprocessor = preprocess(train_df, fit=True)
    
    logger.info("Preprocessing val/test sets (using fitted encoders)...")
    val_df, _ = preprocess(val_df, fit=False, preprocessor=preprocessor)
    test_df, _ = preprocess(test_df, fit=False, preprocessor=preprocessor)
    
    # Save preprocessor
    preprocessor_path = args.output_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            **preprocessor,
            'cat_features': CATEGORICAL_FEATURES,
            'num_features': NUMERICAL_FEATURES
        }, f)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Save splits
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = args.output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info(f"Saved {name} to {path}")
    
    # Summary
    logger.info("-" * 40)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("-" * 40)
    logger.info(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    logger.info(f"Numerical features: {len(NUMERICAL_FEATURES)}")
    logger.info(f"Cardinalities: {preprocessor['cat_cardinalities']}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
