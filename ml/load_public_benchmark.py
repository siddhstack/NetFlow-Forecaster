"""Load and convert CICIDS2017 public benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def convert_cicids2017(df: pd.DataFrame, bin_minutes: int = 1) -> pd.DataFrame:
    """Convert CICIDS2017 CSV to project schema.
    
    Column mapping:
    - timestamp: from 'Timestamp' column, binned to fixed intervals
    - traffic_mbps: sum of 'Flow Bytes/s' across flows per bin
    - latency_ms: mean 'Flow Duration' per bin
    - packet_loss_pct: proxy from retransmission/flag-based fields
    
    Note: packet_loss_pct is a DERIVED PROXY, not directly measured.
    See docstring for details.
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("Input must be a non-empty DataFrame")
    
    # Parse timestamp
    if 'Timestamp' in df.columns:
        df['parsed_time'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        raise ValueError("Missing 'Timestamp' column")
    
    # Sort by timestamp
    df = df.sort_values('parsed_time').reset_index(drop=True)
    
    # Bin by time
    bin_col = df['parsed_time'].dt.floor(f"{bin_minutes}min")
    
    # Aggregate per bin
    result = []
    for ts, group in df.groupby(bin_col):
        traffic_bytes_s = group['Flow Bytes/s'].sum() if 'Flow Bytes/s' in group.columns else 0
        traffic_mbps = traffic_bytes_s / 1_000_000.0
        
        latency_ms = 0.0
        if 'Flow Duration' in group.columns:
            latency_ms = group['Flow Duration'].mean()
        
        # Packet loss proxy: ratio of flows with retransmit/PSH/RST flags
        packet_loss_pct = 0.0
        if 'PSH Flag Count' in group.columns or 'RST Flag Count' in group.columns:
            retransmit_flows = (
                (group['PSH Flag Count'] > 0).sum() + 
                (group['RST Flag Count'] > 0).sum()
            )
            packet_loss_pct = 100.0 * retransmit_flows / max(len(group), 1)
        
        result.append({
            'timestamp': ts,
            'traffic_mbps': float(traffic_mbps),
            'latency_ms': float(latency_ms),
            'packet_loss_pct': float(packet_loss_pct),
        })
    
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to load.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--local-csv", type=Path, default=None, help="Local CSV fallback if Kaggle unavailable.")
    args = parser.parse_args()
    
    df = None
    
    # Try kagglehub
    if args.local_csv is None:
        try:
            import kagglehub
            try:
                ds = kagglehub.dataset_download("cicdataset/cicids2017")
                csv_file = Path(ds) / "MachineLearningCSV/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file, nrows=args.samples)
            except Exception:
                try:
                    ds = kagglehub.dataset_download("dhoogla/cicids2017")
                    csv_files = list(Path(ds).glob("**/*.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0], nrows=args.samples)
                except Exception:
                    pass
        except ImportError:
            pass
    
    # Fallback to local CSV
    if df is None and args.local_csv:
        if args.local_csv.exists():
            df = pd.read_csv(args.local_csv, nrows=args.samples)
    
    if df is None or len(df) == 0:
        raise FileNotFoundError(
            "Could not load CICIDS2017. Provide --local-csv with a manually downloaded copy, "
            "or ensure kagglehub is configured and the dataset is accessible."
        )
    
    # Convert to project schema
    telemetry = convert_cicids2017(df)
    
    # Sample if needed
    if len(telemetry) > args.samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(telemetry), size=args.samples, replace=False)
        telemetry = telemetry.iloc[sorted(indices)].reset_index(drop=True)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(args.output, index=False)
    print(f"Loaded {len(telemetry)} samples from CICIDS2017 -> {args.output}")


if __name__ == "__main__":
    main()
