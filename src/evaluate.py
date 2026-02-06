"""
Model Evaluation Script
=======================
Evaluates the trained Rotary FT-Transformer on the test set.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/checkpoints/best_model.pt
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from models import create_model


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = create_model(
        cat_cardinalities=checkpoint['cat_cardinalities'],
        num_features=checkpoint['num_features'],
        num_classes=2,
        config=checkpoint['model_config']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def df_to_tensors(df: pd.DataFrame, cat_features: list, num_features: list):
    """Convert dataframe to PyTorch tensors."""
    x_cat = torch.tensor(df[cat_features].values, dtype=torch.long)
    x_num = torch.tensor(df[num_features].values, dtype=torch.float32)
    y = torch.tensor(df['label_binary'].values, dtype=torch.long)
    return x_cat, x_num, y


def evaluate_model(model, dataloader, device):
    """Run inference and collect predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            
            logits = model(x_cat, x_num)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (VPN)
            all_labels.extend(y.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def measure_inference_time(model, x_cat, x_num, device, num_runs=1000):
    """Measure inference time per packet."""
    model.eval()
    
    # GPU inference
    x_cat_gpu = x_cat[:1].to(device)
    x_num_gpu = x_num[:1].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_cat_gpu, x_num_gpu)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x_cat_gpu, x_num_gpu)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    gpu_time = (time.perf_counter() - start) / num_runs * 1e6
    
    # CPU inference
    model_cpu = model.cpu()
    x_cat_cpu = x_cat[:1].cpu()
    x_num_cpu = x_num[:1].cpu()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model_cpu(x_cat_cpu, x_num_cpu)
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_cpu(x_cat_cpu, x_num_cpu)
    
    cpu_time = (time.perf_counter() - start) / num_runs * 1e6
    
    # Move model back to original device
    model.to(device)
    
    return gpu_time, cpu_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", type=Path, default=Path("models/checkpoints/best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    model, checkpoint = load_model(args.model, device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Best Val F1 (training): {checkpoint.get('best_f1', 'N/A')}")
    
    # Load preprocessor
    with open(args.data_dir / "preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    
    cat_features = preprocessor['cat_features']
    num_features = preprocessor['num_features']
    
    # Load test data
    print(f"\n📊 Loading test data...")
    test_df = pd.read_parquet(args.data_dir / "test.parquet")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Class distribution: {test_df['label_binary'].value_counts().to_dict()}")
    
    # Convert to tensors
    x_cat, x_num, y_true = df_to_tensors(test_df, cat_features, num_features)
    
    # DataLoader
    test_ds = TensorDataset(x_cat, x_num, y_true)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print(f"\n🔍 Running evaluation...")
    y_pred, y_probs, y_true = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("📈 EVALUATION RESULTS")
    print("=" * 70)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {accuracy:>10.4f}")
    print(f"{'Precision (macro)':<25} {precision:>10.4f}")
    print(f"{'Recall (macro)':<25} {recall:>10.4f}")
    print(f"{'F1-Score (macro)':<25} {f1_macro:>10.4f}")
    print(f"{'F1-Score (weighted)':<25} {f1_weighted:>10.4f}")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    print(f"                 Predicted")
    print(f"                 NonVPN   VPN")
    print(f"Actual NonVPN    {cm[0,0]:>6,}  {cm[0,1]:>6,}")
    print(f"       VPN       {cm[1,0]:>6,}  {cm[1,1]:>6,}")
    
    # Per-class metrics
    print("\n📋 Classification Report:")
    print("-" * 40)
    print(classification_report(
        y_true, y_pred, 
        target_names=['NonVPN', 'VPN'],
        digits=4
    ))
    
    # Inference time
    print("⏱️  Inference Time:")
    print("-" * 40)
    gpu_time, cpu_time = measure_inference_time(model, x_cat, x_num, device)
    print(f"   GPU: {gpu_time:.2f} μs/packet")
    print(f"   CPU: {cpu_time:.2f} μs/packet")
    print(f"   Throughput (GPU): {1e6/gpu_time:,.0f} packets/sec")
    print(f"   Throughput (CPU): {1e6/cpu_time:,.0f} packets/sec")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'inference_time_gpu_us': gpu_time,
        'inference_time_cpu_us': cpu_time,
        'test_samples': len(test_df)
    }
    
    results_path = args.data_dir / "evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
