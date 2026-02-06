"""
Phase 3: Training Pipeline for Rotary FT-Transformer (FAST VERSION)
====================================================================
Uses pre-processed data splits for fast iteration.

Run preprocess.py first:
    python src/preprocess.py

Then train:
    python src/train.py --epochs 10 --batch-size 8192
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from models import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_dir: Path):
    """Load pre-processed train/val/test splits and preprocessor."""
    
    preprocessor_path = data_dir / "preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            "Run 'python src/preprocess.py' first!"
        )
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    
    return train_df, val_df, preprocessor


def df_to_tensors(df: pd.DataFrame, cat_features: list, num_features: list):
    """Convert dataframe to PyTorch tensors."""
    x_cat = torch.tensor(df[cat_features].values, dtype=torch.long)
    x_num = torch.tensor(df[num_features].values, dtype=torch.float32)
    y = torch.tensor(df['label_binary'].values, dtype=torch.long)
    return x_cat, x_num, y


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler  # Mixed precision
) -> float:
    """Train for one epoch with AMP, return average loss."""
    model.train()
    total_loss = 0.0
    
    for x_cat, x_num, y in dataloader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
    
    return (elapsed / num_runs) * 1e6


def main():
    parser = argparse.ArgumentParser(description="Train Rotary FT-Transformer (Fast)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--save-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 3: Rotary FT-Transformer Training (FAST)")
    logger.info("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load pre-processed data
    logger.info("Loading pre-processed data...")
    try:
        train_df, val_df, preprocessor = load_preprocessed_data(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    cat_features = preprocessor['cat_features']
    num_features = preprocessor['num_features']
    cat_cardinalities = preprocessor['cat_cardinalities']
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    logger.info(f"Class dist (train): {train_df['label_binary'].value_counts().to_dict()}")
    
    # Convert to tensors
    logger.info("Converting to tensors...")
    x_cat_train, x_num_train, y_train = df_to_tensors(train_df, cat_features, num_features)
    x_cat_val, x_num_val, y_val = df_to_tensors(val_df, cat_features, num_features)
    
    # Free memory
    del train_df, val_df
    
    # DataLoaders with pinned memory for faster GPU transfer
    train_ds = TensorDataset(x_cat_train, x_num_train, y_train)
    val_ds = TensorDataset(x_cat_val, x_num_val, y_val)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Batch size: {args.batch_size} | Train batches: {len(train_loader)}")
    
    # Create model
    model_config = {
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_dim": args.embed_dim * 4,
        "dropout": 0.1
    }
    
    model = create_model(
        cat_cardinalities=cat_cardinalities,
        num_features=num_features,
        num_classes=2,
        config=model_config
    ).to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    # Training loop
    logger.info("-" * 40)
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save best model
            args.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = args.save_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "cat_cardinalities": cat_cardinalities,
                "num_features": num_features,
                "model_config": model_config,
                "best_f1": best_f1,
                "epoch": epoch + 1
            }, save_path)
    
    # Final results
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Val F1: {best_f1:.4f}")
    
    # Measure inference time
    inference_time = measure_inference_time(model, x_cat_val, x_num_val)
    logger.info(f"Inference time: {inference_time:.2f} μs/packet (CPU)")
    
    logger.info(f"Best model saved to {args.save_dir / 'best_model.pt'}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
