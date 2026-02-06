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


def measure_throughput(model, x_cat, x_num, device, batch_sizes=[1, 64, 256, 1024, 4096]):
    """
    Measure BATCHED throughput at various batch sizes.
    Returns throughput in packets/sec for thesis-quality benchmarks.
    """
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        # Prepare batch
        actual_batch = min(batch_size, len(x_cat))
        x_cat_batch = x_cat[:actual_batch].to(device)
        x_num_batch = x_num[:actual_batch].to(device)
        
        # Warmup (important for GPU)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x_cat_batch, x_num_batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark multiple runs
        num_runs = max(100, 10000 // batch_size)  # More runs for small batches
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x_cat_batch, x_num_batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        total_packets = num_runs * actual_batch
        throughput = total_packets / elapsed
        latency_per_packet = (elapsed / total_packets) * 1e6  # microseconds
        
        results[batch_size] = {
            'throughput': throughput,
            'latency_us': latency_per_packet,
            'total_packets': total_packets,
            'elapsed_sec': elapsed
        }
    
    return results


def measure_cpu_throughput(model, x_cat, x_num, batch_size=1024):
    """Measure CPU throughput for deployment scenarios."""
    model_cpu = model.cpu()
    model_cpu.eval()
    
    actual_batch = min(batch_size, len(x_cat))
    x_cat_batch = x_cat[:actual_batch].cpu()
    x_num_batch = x_num[:actual_batch].cpu()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(x_cat_batch, x_num_batch)
    
    # Benchmark
    num_runs = 100
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_cpu(x_cat_batch, x_num_batch)
    
    elapsed = time.perf_counter() - start
    total_packets = num_runs * actual_batch
    throughput = total_packets / elapsed
    
    return throughput, (elapsed / total_packets) * 1e6


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
    
    
    # BATCHED Throughput (Thesis-quality benchmarks)
    print("\n🚀 BATCHED THROUGHPUT (GPU):")
    print("-" * 50)
    print(f"{'Batch Size':<12} {'Throughput':>18} {'Latency':>15}")
    print(f"{'':12} {'(packets/sec)':>18} {'(μs/packet)':>15}")
    print("-" * 50)
    
    throughput_results = measure_throughput(model, x_cat, x_num, device)
    for batch_size, metrics in throughput_results.items():
        print(f"{batch_size:<12} {metrics['throughput']:>18,.0f} {metrics['latency_us']:>15.2f}")
    
    # Best throughput for thesis
    best_batch = max(throughput_results.keys())
    best_throughput = throughput_results[best_batch]['throughput']
    print("-" * 50)
    print(f"⭐ Peak Throughput: {best_throughput:,.0f} packets/sec (batch={best_batch})")
    
    # CPU throughput
    print("\n🖥️  CPU THROUGHPUT (batch=1024):")
    print("-" * 40)
    model.to(device)  # Reset model to GPU first
    cpu_throughput, cpu_latency = measure_cpu_throughput(model, x_cat, x_num, batch_size=1024)
    print(f"   Throughput: {cpu_throughput:,.0f} packets/sec")
    print(f"   Latency: {cpu_latency:.2f} μs/packet")
    model.to(device)  # Move back to GPU
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'throughput_results': {k: v for k, v in throughput_results.items()},
        'peak_throughput_gpu': best_throughput,
        'peak_batch_size': best_batch,
        'cpu_throughput': cpu_throughput,
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
