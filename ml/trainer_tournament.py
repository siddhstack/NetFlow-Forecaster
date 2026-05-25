"""Candidate selection helpers for automated telemetry benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from telemetry_profile import TelemetryProfile, profile_telemetry


@dataclass
class Candidate:
    id: str
    trainer: str
    args: list[str]


def candidates_for_profile(profile: TelemetryProfile) -> list[Candidate]:
    seq = str(profile.recommended_sequence_length)
    lookback = str(profile.recommended_lookback)
    q = str(profile.recommended_spike_quantile)
    epochs = str(profile.recommended_epochs)
    base = [
        Candidate("hybrid_default", "enhanced", ["--epochs", epochs, "--sequence-length", seq, "--spike-quantile", q]),
        Candidate(
            "hybrid_aggressive",
            "enhanced",
            ["--epochs", str(profile.recommended_epochs + 20), "--sequence-length", seq, "--spike-quantile", "0.85", "--spike-weight", "8", "--focal-gamma", "1.0"],
        ),
        Candidate("gb_spike", "gb", ["--lookback", lookback, "--spike-oversample", "4"]),
        Candidate("hybrid_short_seq", "enhanced", ["--epochs", epochs, "--sequence-length", "48", "--spike-quantile", q]),
        Candidate("hybrid_gb_heavy", "enhanced", ["--epochs", epochs, "--sequence-length", seq, "--gb-weight", "0.85", "--lstm-weight", "0.15"]),
        Candidate("gb_spike_deep", "gb", ["--lookback", "36", "--spike-oversample", "6"]),
        Candidate(
            "hybrid_low_quantile",
            "enhanced",
            [
                "--epochs", str(profile.recommended_epochs + 20),
                "--sequence-length", seq,
                "--spike-quantile", "0.82",
                "--spike-weight", "10",
                "--spike-lift-near", "1.0,0.9,0.95",
                "--spike-lift-factors", "1.05,1.01,1.05",
            ],
        ),
        # R2 recovery candidate: short lookback (24h) forces the model to learn
        # from recent spike context rather than long-range daily patterns.
        # Combined with high spike_weight=8.0 this specifically targets the
        # latency and packet_loss spike recall gap.
        Candidate(
            "hybrid_r2_recovery",
            "enhanced",
            [
                "--epochs", str(profile.recommended_epochs + 20),
                "--sequence-length", "24",
                "--spike-quantile", "0.85",
                "--spike-weight", "8.0",
                "--focal-gamma", "0.5",
                "--feature-spike-multipliers", "1.0,1.8,2.5",
            ],
        ),
    ]
    preferred = profile.recommended_trainer
    if preferred == "gb_only":
        order = ["gb_spike_deep", "gb_spike", "hybrid_r2_recovery", "hybrid_short_seq", "hybrid_default", "hybrid_aggressive", "hybrid_gb_heavy", "hybrid_low_quantile"]
    elif preferred == "hybrid_aggressive":
        order = ["hybrid_aggressive", "hybrid_r2_recovery", "hybrid_low_quantile", "hybrid_gb_heavy", "hybrid_default", "gb_spike", "gb_spike_deep", "hybrid_short_seq"]
    else:
        order = ["hybrid_default", "hybrid_r2_recovery", "hybrid_aggressive", "hybrid_gb_heavy", "gb_spike", "gb_spike_deep", "hybrid_short_seq", "hybrid_low_quantile"]
    lookup = {candidate.id: candidate for candidate in base}
    return [lookup[item] for item in order]


def next_candidate(profile: TelemetryProfile, attempts: list[dict]) -> Candidate:
    try:
        from experience_store import load_policy
        from meta_policy import rank_candidates

        ranked = rank_candidates(profile, load_policy(), attempts)
        if ranked:
            return ranked[0]
    except Exception:
        pass

    candidates = candidates_for_profile(profile)
    if not attempts:
        return candidates[0]
    last = attempts[-1]
    gates = last.get("summary", {}).get("gates_passed", {})
    per_feature = {row["metric"]: row for row in last.get("summary", {}).get("per_feature", [])}
    tried = [attempt["candidate"]["id"] for attempt in attempts]
    preferred = None
    quality = float(last.get("summary", {}).get("overall", {}).get("normalized_quality_pct", 0.0))
    if not gates.get("traffic_spike_f1_ge_0_50", True):
        preferred = "hybrid_aggressive"
    if not gates.get("beats_persistence_each_feature_mae", True):
        traffic = per_feature.get("traffic_mbps", {})
        preferred = "hybrid_gb_heavy" if float(traffic.get("mae_improvement_pct", 0.0)) < 0 else "gb_spike"
    if not gates.get("quality_ge_90", True) and quality < 80:
        # If below 80% and not hitting traffic spike gate, try the recovery
        # candidate before falling all the way back to gb_spike.
        if not gates.get("traffic_spike_f1_ge_0_50", True):
            preferred = "hybrid_aggressive"
        else:
            preferred = "hybrid_r2_recovery"
    if preferred:
        for candidate in candidates:
            if candidate.id == preferred and candidate.id not in tried:
                return candidate
    for candidate in candidates:
        if candidate.id not in tried:
            return candidate
    return candidates[len(attempts) % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    profile = profile_telemetry(Path(args.data))
    payload = {"profile": asdict(profile), "candidates": [asdict(candidate) for candidate in candidates_for_profile(profile)]}
    text = json.dumps(payload, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
