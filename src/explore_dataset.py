"""
Dataset Explorer - Display examples and statistics from the extracted Parquet dataset
"""

import pandas as pd
from pathlib import Path

# Configuration
DATASET_PATH = Path("data/processed/dataset.parquet")


def main():
    print("=" * 70)
    print("📊 Dataset Explorer - ISCX VPN-nonVPN Feature Dataset")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📁 Loading: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    
    # Basic stats
    print("\n" + "-" * 40)
    print("📈 BASIC STATISTICS")
    print("-" * 40)
    print(f"Total packets:     {len(df):,}")
    print(f"Total columns:     {len(df.columns)}")
    print(f"Memory usage:      {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Unique flows:      {df['flow_id'].nunique():,}")
    
    # Label distribution
    print("\n" + "-" * 40)
    print("🏷️  LABEL DISTRIBUTION")
    print("-" * 40)
    print(f"VPN packets:       {(df['label_binary'] == 1).sum():,} ({(df['label_binary'] == 1).mean()*100:.1f}%)")
    print(f"NonVPN packets:    {(df['label_binary'] == 0).sum():,} ({(df['label_binary'] == 0).mean()*100:.1f}%)")
    
    # App distribution
    print("\n" + "-" * 40)
    print("📱 APPLICATION DISTRIBUTION")
    print("-" * 40)
    app_counts = df['label_app'].value_counts()
    for app, count in app_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"{app:15s} {count:>10,} ({pct:5.1f}%) {bar}")
    
    # Protocol distribution
    print("\n" + "-" * 40)
    print("🔌 PROTOCOL DISTRIBUTION")
    print("-" * 40)
    tcp_count = (df['tcp_flags'] != -1).sum()
    udp_count = (df['udp_len'] != -1).sum()
    print(f"TCP packets:       {tcp_count:,} ({tcp_count/len(df)*100:.1f}%)")
    print(f"UDP packets:       {udp_count:,} ({udp_count/len(df)*100:.1f}%)")
    
    # Feature ranges
    print("\n" + "-" * 40)
    print("📐 FEATURE VALUE RANGES")
    print("-" * 40)
    feature_cols = ['ip_len', 'ip_ttl', 'tcp_window', 'tcp_seq_high', 'tcp_seq_low']
    for col in feature_cols:
        if col in df.columns:
            valid = df[df[col] != -1][col]
            if len(valid) > 0:
                print(f"{col:15s} min={valid.min():>8,}  max={valid.max():>10,}  mean={valid.mean():>10,.1f}")
    
    # Sample rows
    print("\n" + "-" * 40)
    print("🔍 SAMPLE TCP PACKET (VPN)")
    print("-" * 40)
    tcp_vpn = df[(df['label_binary'] == 1) & (df['tcp_flags'] != -1)].head(1)
    if len(tcp_vpn) > 0:
        for col in df.columns:
            print(f"  {col:18s}: {tcp_vpn[col].values[0]}")
    
    print("\n" + "-" * 40)
    print("🔍 SAMPLE UDP PACKET (NonVPN)")
    print("-" * 40)
    udp_nonvpn = df[(df['label_binary'] == 0) & (df['udp_len'] != -1)].head(1)
    if len(udp_nonvpn) > 0:
        for col in df.columns:
            print(f"  {col:18s}: {udp_nonvpn[col].values[0]}")
    
    # Flow ID examples
    print("\n" + "-" * 40)
    print("🌊 SAMPLE FLOW IDs")
    print("-" * 40)
    for i, flow_id in enumerate(df['flow_id'].unique()[:5]):
        flow_size = (df['flow_id'] == flow_id).sum()
        print(f"  {i+1}. {flow_id} ({flow_size:,} packets)")
    
    print("\n" + "=" * 70)
    print("✅ Dataset exploration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
