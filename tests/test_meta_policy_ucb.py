import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.meta_policy import select_candidate_config
from tests.test_experience_store import sample_profile


class FakeCandidate:
    def __init__(self, candidate_id, quality_score, selection_count=0):
        self.id = candidate_id
        self.mean_quality_score = quality_score
        self.selection_count = selection_count


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates

    def get_applicable_candidates(self, dataset_profile):
        return self.candidates

    def get_total_runs(self):
        return 10


def test_select_candidate_config_prefers_exploration_when_untried():
    unseen = FakeCandidate("new_candidate", 0.0, 0)
    known = FakeCandidate("known_candidate", 0.95, 10)
    store = FakeStore([unseen, known])
    candidate = select_candidate_config(store, sample_profile())
    assert candidate.id == "new_candidate"


def test_select_candidate_config_uses_ucb_score_for_tried_candidates():
    unseen = FakeCandidate("new_candidate", 0.0, 0)
    known = FakeCandidate("known_candidate", 0.95, 10)
    store = FakeStore([known, unseen])
    candidate = select_candidate_config(store, sample_profile())
    assert candidate.id == "new_candidate"
