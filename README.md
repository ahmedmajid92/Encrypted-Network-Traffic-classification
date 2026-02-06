# NetworkTC: Encrypted Traffic Classification with Rotary FT-Transformer

A PhD research project implementing a novel **Rotary Feature-Tokenizer Transformer** for encrypted network traffic classification using only packet header features.

## 🎯 Research Contribution

**Novel Hypothesis:** Treating packet headers as a sequence of tabular features enriched with **Rotary Position Embeddings (RoPE)** to capture protocol hierarchy, where position represents protocol depth (IP → TCP/UDP → derived features).

### Key Innovations

- **Tabular-First Approach:** Move away from sequence modeling (BERT) to FT-Transformers
- **Protocol Hierarchy Encoding:** RoPE encodes IP → Transport → Application layer structure
- **Split-Integer Features:** 32-bit TCP seq/ack numbers split into 16-bit pairs for better learning
- **Flow-based Validation:** GroupKFold by flow_id prevents data leakage

---

## 📊 Dataset

Uses the **ISCX VPN-nonVPN** dataset with 140 PCAP files (~20GB):

- **VPN traffic:** 31 files (Skype, Hangouts, FTPS, etc.)
- **NonVPN traffic:** 109 files (Facebook, YouTube, Netflix, etc.)

### Extracted Features (24 columns)

| Category       | Features                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| **Metadata**   | `flow_id` (for splitting), `label_binary`, `label_app`                                                        |
| **IP Header**  | `ip_version`, `ip_ihl`, `ip_tos`, `ip_len`, `ip_id`, `ip_flags`, `ip_frag`, `ip_ttl`, `ip_proto`, `ip_chksum` |
| **TCP Header** | `tcp_dataofs`, `tcp_flags`, `tcp_window`, `tcp_chksum`, `tcp_urg`, `tcp_seq_high/low`, `tcp_ack_high/low`     |
| **UDP Header** | `udp_len`, `udp_chksum`                                                                                       |

---

## 🏗️ Model Architecture

```
RotaryFTTransformer (~207K parameters)
├── FeatureTokenizer
│   ├── 9 Categorical Embeddings (ip_version, tcp_flags, seq/ack splits, etc.)
│   └── 11 Numerical Projections (ip_len, ttl, window, etc.)
├── CLS Token (learnable)
├── 3× TransformerEncoderLayer
│   ├── RotaryMultiHeadAttention (4 heads + RoPE)
│   └── ReGLU FFN (proven better for tabular data)
└── Classification Head
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/NetworkTC.git
cd NetworkTC

# Install dependencies
pip install -r requirements.txt
```

### Phase 1: Dataset Extraction

Extract features from PCAP files:

```bash
# Place PCAP files in data/raw/VPN/ and data/raw/NonVPN/
python src/make_dataset.py --output data/processed/dataset.parquet
```

### Phase 2: Explore Dataset

View dataset statistics and samples:

```bash
python src/explore_dataset.py
```

### Phase 3: Train Model

Train with 5-fold GroupKFold cross-validation:

```bash
# Quick test
python src/train.py --max-samples 50000 --epochs 3

# Full training
python src/train.py --epochs 10 --batch-size 1024
```

---

## 📁 Project Structure

```
NetworkTC/
├── data/
│   ├── raw/                    # PCAP files (git-ignored)
│   │   ├── VPN/
│   │   └── NonVPN/
│   └── processed/              # Parquet dataset (git-ignored)
├── models/
│   └── checkpoints/            # Saved models (git-ignored)
├── src/
│   ├── make_dataset.py         # PCAP → Parquet extraction
│   ├── explore_dataset.py      # Dataset statistics viewer
│   ├── train.py                # Training pipeline
│   └── models/
│       ├── __init__.py
│       ├── components.py       # FeatureTokenizer, RoPE, ReGLU
│       └── transformer.py      # RotaryFTTransformer
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Training Configuration

| Parameter      | Default | Description               |
| -------------- | ------- | ------------------------- |
| `--epochs`     | 10      | Training epochs           |
| `--batch-size` | 1024    | Batch size                |
| `--lr`         | 1e-3    | Learning rate             |
| `--n-folds`    | 5       | GroupKFold splits         |
| `--embed-dim`  | 64      | Token embedding dimension |
| `--num-heads`  | 4       | Attention heads           |
| `--num-layers` | 3       | Transformer layers        |

---

## 📈 Metrics

The training pipeline logs:

- **Training Loss** (per epoch)
- **Validation Accuracy**
- **Macro F1-Score** (handles class imbalance)
- **Inference Time** (μs/packet on CPU)

---

## 🔬 Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

---

## 📄 License

This project is part of PhD research. Please cite appropriately if used.

---

## 👤 Author

PhD Candidate - Encrypted Traffic Classification Research
