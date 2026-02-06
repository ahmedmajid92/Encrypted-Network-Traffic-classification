"""
Phase 3: Training Pipeline for Rotary FT-Transformer
=====================================================
Rigorous training with GroupKFold cross-validation to prevent flow leakage.

PhD Research: Encrypted Traffic Classification
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Feature configuration
CATEGORICAL_FEATURES = [
    'ip_version', 'ip_proto', 'ip_tos', 'tcp_flags', 'tcp_dataofs',
    'tcp_seq_high', 'tcp_seq_low', 'tcp_ack_high', 'tcp_ack_low'
]

NUMERICAL_FEATURES = [
    'ip_len', 'ip_ttl', 'ip_chksum', 'tcp_window', 'udp_len', 'tcp_urg',
    'ip_ihl', 'ip_id', 'ip_flags', 'ip_frag', 'udp_chksum'
]

# Note: tcp_dataofs is the column name in our dataset (scapy uses dataofs)


class TrafficDataset:
    """Preprocessor and dataset handler for traffic classification."""
    
    def __init__(self, df: pd.DataFrame, cat_features: list, num_features: list):
        self.df = df.copy()
        self.cat_features = cat_features
        self.num_features = num_features
        
        # Encoders and scalers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cat_cardinalities = {}
        
    def preprocess(self):
        """Apply preprocessing: handle missing values, encode categoricals, scale numericals."""
        logger.info("Preprocessing data...")
        
        # Replace -1 sentinel values with 0
        for col in self.cat_features + self.num_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(-1, 0)
                self.df[col] = self.df[col].fillna(0)
        
        # Encode categorical features
        for col in self.cat_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.cat_cardinalities[col] = len(le.classes_)
            else:
                logger.warning(f"Categorical feature '{col}' not found, using placeholder")
                self.df[col] = 0
                self.cat_cardinalities[col] = 1
        
        # Scale numerical features
        num_cols_present = [c for c in self.num_features if c in self.df.columns]
        if num_cols_present:
            self.df[num_cols_present] = self.scaler.fit_transform(self.df[num_cols_present])
        
        # Fill missing numerical columns
        for col in self.num_features:
            if col not in self.df.columns:
                logger.warning(f"Numerical feature '{col}' not found, using zeros")
                self.df[col] = 0.0
        
        logger.info(f"  Categorical: {len(self.cat_features)} features")
        logger.info(f"  Numerical: {len(self.num_features)} features")
        logger.info(f"  Cardinalities: {self.cat_cardinalities}")
        
    def get_tensors(self, indices: np.ndarray):
        """Get PyTorch tensors for given indices."""
        subset = self.df.iloc[indices]
        
        x_cat = torch.tensor(subset[self.cat_features].values, dtype=torch.long)
        x_num = torch.tensor(subset[self.num_features].values, dtype=torch.float32)
        y = torch.tensor(subset['label_binary'].values, dtype=torch.long)
        
        return x_cat, x_num, y


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    
    for x_cat, x_num, y in dataloader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y)
    
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[float, float]:
    """Evaluate model, return accuracy and macro F1."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            logits = model(x_cat, x_num)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1


def measure_inference_time(
    model: nn.Module,
    x_cat_sample: torch.Tensor,
    x_num_sample: torch.Tensor,
    num_runs: int = 1000
) -> float:
    """Measure average inference time in microseconds per packet on CPU."""
    model.eval()
    model.cpu()
    x_cat_sample = x_cat_sample.cpu()
    x_num_sample = x_num_sample.cpu()
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_cat_sample[:1], x_num_sample[:1])
    
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x_cat_sample[:1], x_num_sample[:1])
    elapsed = time.perf_counter() - start
    
    return (elapsed / num_runs) * 1e6  # Convert to microseconds


def main():
    parser = argparse.ArgumentParser(description="Train Rotary FT-Transformer")
    parser.add_argument("--data", type=Path, default=Path("data/processed/dataset.parquet"))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--save-dir", type=Path, default=Path("models/checkpoints"))
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 3: Rotary FT-Transformer Training")
    logger.info("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        logger.info(f"Limited to {len(df):,} samples")
    
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Class distribution: {df['label_binary'].value_counts().to_dict()}")
    
    # Fix column naming (tcp_dataofs vs tcp_dataoff)
    if 'tcp_dataoff' in df.columns and 'tcp_dataofs' not in df.columns:
        df['tcp_dataofs'] = df['tcp_dataoff']
    
    # Preprocess
    dataset = TrafficDataset(df, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    dataset.preprocess()
    
    # GroupKFold by flow_id
    logger.info(f"Setting up {args.n_folds}-fold GroupKFold by flow_id...")
    groups = df['flow_id'].values
    gkf = GroupKFold(n_splits=args.n_folds)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label_binary'], groups)):
        logger.info("-" * 40)
        logger.info(f"Fold {fold + 1}/{args.n_folds}")
        logger.info(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        
        # Get tensors
        x_cat_train, x_num_train, y_train = dataset.get_tensors(train_idx)
        x_cat_val, x_num_val, y_val = dataset.get_tensors(val_idx)
        
        # DataLoaders
        train_ds = TensorDataset(x_cat_train, x_num_train, y_train)
        val_ds = TensorDataset(x_cat_val, x_num_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model_config = {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.embed_dim * 4,
            "dropout": 0.1
        }
        
        model = create_model(
            cat_cardinalities=dataset.cat_cardinalities,
            num_features=NUMERICAL_FEATURES,
            num_classes=2,
            config=model_config
        ).to(device)
        
        if fold == 0:
            logger.info(f"  Model parameters: {model.count_parameters():,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            scheduler.step()
            
            logger.info(
                f"  Epoch {epoch+1:2d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )
            
            if val_f1 > best_f1:
                best_f1 = val_f1
        
        fold_results.append({
            "fold": fold + 1,
            "best_f1": best_f1,
            "final_acc": val_acc
        })
        
        # Measure inference time (only on first fold)
        if fold == 0:
            inference_time = measure_inference_time(model, x_cat_val, x_num_val)
            logger.info(f"  Inference time: {inference_time:.2f} μs/packet (CPU)")
    
    # Summary
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    
    f1_scores = [r['best_f1'] for r in fold_results]
    acc_scores = [r['final_acc'] for r in fold_results]
    
    logger.info(f"Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    logger.info(f"Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    
    # Save final model
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_dir / "rotary_ft_transformer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cat_cardinalities": dataset.cat_cardinalities,
        "num_features": NUMERICAL_FEATURES,
        "model_config": model_config,
        "cv_results": fold_results
    }, save_path)
    logger.info(f"Model saved to {save_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
