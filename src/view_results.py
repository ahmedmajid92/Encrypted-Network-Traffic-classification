"""Quick script to view evaluation results."""
import pickle

with open('data/processed/evaluation_results.pkl', 'rb') as f:
    r = pickle.load(f)

print("=" * 50)
print("EVALUATION RESULTS SUMMARY")
print("=" * 50)
print(f"Accuracy:     {r['accuracy']:.4f}")
print(f"F1 Macro:     {r['f1_macro']:.4f}")
print(f"F1 Weighted:  {r['f1_weighted']:.4f}")
print()
print("THROUGHPUT BY BATCH SIZE:")
print("-" * 40)
for batch, metrics in r['throughput_results'].items():
    print(f"  Batch {batch:>5}: {metrics['throughput']:>12,.0f} packets/sec")
print("-" * 40)
print(f"Peak GPU:   {r['peak_throughput_gpu']:,.0f} packets/sec (batch={r['peak_batch_size']})")
print(f"CPU (1024): {r['cpu_throughput']:,.0f} packets/sec")
