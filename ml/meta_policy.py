"""Rank benchmark candidates using persisted experience and UCB1 exploration."""

from __future__ import annotations

import math

import numpy as np

from experience_store import fingerprint
from telemetry_profile import TelemetryProfile
from trainer_tournament import Candidate, candidates_for_profile


def _normalize_quality(raw_quality: float) -> float:
    value = float(raw_quality)
    if value > 1.0:
        return value / 100.0
    return max(0.0, min(1.0, value))


def select_candidate_config(experience_store, dataset_profile: TelemetryProfile) -> Candidate:
    """Select the next candidate using explicit UCB1 exploration/exploitation."""
    candidates = experience_store.get_applicable_candidates(dataset_profile)
    total_tournament_runs = max(1, int(experience_store.get_total_runs()))

    best_candidate = None
    max_ucb_score = -float("inf")

    for candidate in candidates:
        selection_count = int(getattr(candidate, "selection_count", 0))
        if selection_count == 0:
            return candidate

        exploitation = _normalize_quality(getattr(candidate, "mean_quality_score", 0.0))
        exploration = np.sqrt(2.0 * np.log(total_tournament_runs) / selection_count)
        ucb_score = exploitation + exploration

        if ucb_score > max_ucb_score:
            best_candidate = candidate
            max_ucb_score = ucb_score

    if best_candidate is not None:
        return best_candidate
    return candidates[0] if candidates else None


def rank_candidates(profile: TelemetryProfile, policy: dict, attempts: list) -> list[Candidate]:
    candidates = candidates_for_profile(profile)
    tried = {attempt["candidate"]["id"] for attempt in attempts if "candidate" in attempt}
    if not policy:
        return [candidate for candidate in candidates if candidate.id not in tried]
    fp = fingerprint(profile)
    global_scores = policy.get("global_candidate_scores", {})
    attempts_by_candidate = policy.get("candidate_attempts", {})
    fp_policy = policy.get("by_fingerprint", {}).get(fp, {})
    vol_policy = policy.get("by_volatility", {}).get(profile.volatility, {})
    total = max(1, sum(int(row.get("count", 1)) for row in policy.get("by_fingerprint", {}).values()))
    base_order = {candidate.id: idx for idx, candidate in enumerate(candidates)}

    ranked = []
    for candidate in candidates:
        if candidate.id in tried:
            continue
        historical = _normalize_quality(global_scores.get(candidate.id, 50.0))
        if fp_policy.get("best_candidate") == candidate.id:
            historical = max(historical, _normalize_quality(fp_policy.get("mean_quality", historical)))
        if vol_policy.get("best_candidate") == candidate.id:
            historical = max(historical, _normalize_quality(vol_policy.get("mean_quality", historical)))
        count = int(attempts_by_candidate.get(candidate.id, 0))
        selection_count = max(1, count)
        exploration = math.sqrt(2.0 * math.log(total + 1.0) / selection_count)
        rule_priority = 100.0 - base_order[candidate.id] * 10.0
        score = 0.6 * historical + 0.2 * (rule_priority / 100.0) + 0.2 * exploration
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]
