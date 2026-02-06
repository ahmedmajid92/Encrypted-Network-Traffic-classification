"""
Phase 2: Tabular Feature Extraction Pipeline
=============================================
Parses ISCX VPN-nonVPN PCAP files and extracts protocol header features
for FT-Transformer training. Outputs a structured Parquet file.

Author: NetworkTC PhD Project
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed
from scapy.all import IP, TCP, UDP, PcapReader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Feature column order
FEATURE_COLUMNS = [
    # Metadata
    "flow_id", "label_binary", "label_app",
    # IP Header (10 fields)
    "ip_version", "ip_ihl", "ip_tos", "ip_len", "ip_id",
    "ip_flags", "ip_frag", "ip_ttl", "ip_proto", "ip_chksum",
    # TCP Header (9 fields including split seq/ack)
    "tcp_dataoff", "tcp_flags", "tcp_window", "tcp_chksum", "tcp_urg",
    "tcp_seq_high", "tcp_seq_low", "tcp_ack_high", "tcp_ack_low",
    # UDP Header (2 fields)
    "udp_len", "udp_chksum"
]


def parse_app_from_filename(filename: str, is_vpn: bool) -> str:
    """
    Extract application name from PCAP filename.
    
    VPN files: vpn_skype_audio1.pcap -> 'skype'
    NonVPN files: facebook_audio2a.pcap -> 'facebook'
                  AIMchat1.pcapng -> 'aim'
    """
    stem = Path(filename).stem.lower()
    
    if is_vpn:
        # VPN format: vpn_<app>_<variant>.pcap
        parts = stem.split("_")
        if len(parts) > 1:
            return parts[1]
        return "unknown"
    else:
        # NonVPN format: <app>_<variant>.pcap or <app><variant>.pcap
        parts = stem.split("_")
        # Remove trailing digits (e.g., 'aimchat1' -> 'aimchat')
        app = parts[0].rstrip("0123456789")
        # Handle cases like 'aimchat' -> 'aim', 'facebookchat' -> 'facebook'
        # Common suffixes to strip
        suffixes = ["chat", "audio", "video", "file", "down", "up"]
        for suffix in suffixes:
            if app.endswith(suffix) and len(app) > len(suffix):
                app = app[:-len(suffix)]
                break
        return app if app else "unknown"


def extract_packet_features(pkt, label_binary: int, label_app: str) -> Optional[dict]:
    """
    Extract header features from a single packet.
    Returns None if packet is not TCP/UDP over IP.
    """
    # Must have IP layer
    if not pkt.haslayer(IP):
        return None
    
    ip = pkt[IP]
    
    # Determine transport protocol
    is_tcp = pkt.haslayer(TCP)
    is_udp = pkt.haslayer(UDP)
    
    # Drop non-TCP/UDP packets (e.g., ICMP)
    if not is_tcp and not is_udp:
        return None
    
    # Build flow_id
    if is_tcp:
        transport = pkt[TCP]
        proto_str = "TCP"
        src_port = transport.sport
        dst_port = transport.dport
    else:
        transport = pkt[UDP]
        proto_str = "UDP"
        src_port = transport.sport
        dst_port = transport.dport
    
    flow_id = f"{ip.src}-{ip.dst}-{src_port}-{dst_port}-{proto_str}"
    
    # Extract IP header fields
    features = {
        "flow_id": flow_id,
        "label_binary": label_binary,
        "label_app": label_app,
        # IP Header
        "ip_version": ip.version,
        "ip_ihl": ip.ihl,
        "ip_tos": ip.tos,
        "ip_len": ip.len,
        "ip_id": ip.id,
        "ip_flags": int(ip.flags),  # Convert FlagValue to int
        "ip_frag": ip.frag,
        "ip_ttl": ip.ttl,
        "ip_proto": ip.proto,
        "ip_chksum": ip.chksum if ip.chksum is not None else -1,
    }
    
    # Extract TCP or UDP fields
    if is_tcp:
        tcp = pkt[TCP]
        seq = tcp.seq if tcp.seq is not None else 0
        ack = tcp.ack if tcp.ack is not None else 0
        
        features.update({
            "tcp_dataoff": tcp.dataofs if tcp.dataofs is not None else -1,
            "tcp_flags": int(tcp.flags),  # Bitmask
            "tcp_window": tcp.window if tcp.window is not None else -1,
            "tcp_chksum": tcp.chksum if tcp.chksum is not None else -1,
            "tcp_urg": tcp.urgptr if tcp.urgptr is not None else -1,
            # Split 32-bit seq/ack into two 16-bit values
            "tcp_seq_high": seq // 65536,
            "tcp_seq_low": seq % 65536,
            "tcp_ack_high": ack // 65536,
            "tcp_ack_low": ack % 65536,
            # UDP fields set to -1
            "udp_len": -1,
            "udp_chksum": -1,
        })
    else:
        udp = pkt[UDP]
        features.update({
            # TCP fields set to -1
            "tcp_dataoff": -1,
            "tcp_flags": -1,
            "tcp_window": -1,
            "tcp_chksum": -1,
            "tcp_urg": -1,
            "tcp_seq_high": -1,
            "tcp_seq_low": -1,
            "tcp_ack_high": -1,
            "tcp_ack_low": -1,
            # UDP Header
            "udp_len": udp.len if udp.len is not None else -1,
            "udp_chksum": udp.chksum if udp.chksum is not None else -1,
        })
    
    return features


def process_single_pcap(pcap_path: Path, label_binary: int, label_app: str) -> pd.DataFrame:
    """
    Process a single PCAP file using streaming to avoid memory issues.
    Returns a DataFrame of extracted features.
    """
    records = []
    
    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                features = extract_packet_features(pkt, label_binary, label_app)
                if features is not None:
                    records.append(features)
    except Exception as e:
        logger.error(f"Error processing {pcap_path.name}: {e}")
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    
    if not records:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


def collect_pcap_files(data_dir: Path, max_files: Optional[int] = None) -> list[tuple[Path, int, str]]:
    """
    Collect all PCAP files with their labels.
    Returns list of (path, binary_label, app_label) tuples.
    """
    files = []
    
    # VPN files (label=1)
    vpn_dir = data_dir / "VPN"
    if vpn_dir.exists():
        for pcap in vpn_dir.glob("*"):
            if pcap.suffix.lower() in [".pcap", ".pcapng"]:
                app = parse_app_from_filename(pcap.name, is_vpn=True)
                files.append((pcap, 1, app))
    
    # NonVPN files (label=0)
    nonvpn_dir = data_dir / "NonVPN"
    if nonvpn_dir.exists():
        for pcap in nonvpn_dir.glob("*"):
            if pcap.suffix.lower() in [".pcap", ".pcapng"]:
                app = parse_app_from_filename(pcap.name, is_vpn=False)
                files.append((pcap, 0, app))
    
    logger.info(f"Found {len(files)} PCAP files total")
    
    if max_files:
        files = files[:max_files]
        logger.info(f"Limited to {max_files} files for testing")
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Extract tabular features from ISCX VPN-nonVPN PCAP dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input directory containing VPN/ and NonVPN/ folders"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/dataset.parquet"),
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 2: Tabular Feature Extraction")
    logger.info("=" * 60)
    
    # Collect files
    pcap_files = collect_pcap_files(args.input, args.max_files)
    
    if not pcap_files:
        logger.error(f"No PCAP files found in {args.input}")
        return
    
    # Process files in parallel with progress bar
    logger.info(f"Processing with {args.n_jobs} parallel jobs...")
    
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_single_pcap)(path, label, app)
        for path, label, app in tqdm(pcap_files, desc="Processing PCAPs")
    )
    
    # Concatenate all results
    logger.info("Concatenating results...")
    df = pd.concat(results, ignore_index=True)
    
    # Summary statistics
    logger.info("-" * 40)
    logger.info(f"Total packets extracted: {len(df):,}")
    logger.info(f"VPN packets: {(df['label_binary'] == 1).sum():,}")
    logger.info(f"NonVPN packets: {(df['label_binary'] == 0).sum():,}")
    logger.info(f"Unique apps: {df['label_app'].nunique()}")
    logger.info(f"Unique flows: {df['flow_id'].nunique():,}")
    logger.info("-" * 40)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    logger.info(f"Saving to {args.output}...")
    df.to_parquet(args.output, index=False, engine="pyarrow")
    
    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")
    logger.info("Done!")


if __name__ == "__main__":
    main()
