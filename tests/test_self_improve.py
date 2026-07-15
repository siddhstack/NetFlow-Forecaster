import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.self_improve import verify_and_deploy_ensemble


class FakeModel:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, validation_data):
        return self.metrics


def test_verify_and_deploy_ensemble_falls_back_to_baseline():
    deployed = verify_and_deploy_ensemble(
        FakeModel({"weighted_loss": 0.40}),
        FakeModel({"weighted_loss": 0.20}),
        validation_data={"samples": 100},
    )
    assert deployed is not None
    assert deployed.metrics["weighted_loss"] == 0.20


def test_verify_and_deploy_ensemble_keeps_stronger_model():
    deployed = verify_and_deploy_ensemble(
        FakeModel({"weighted_loss": 0.15}),
        FakeModel({"weighted_loss": 0.20}),
        validation_data={"samples": 100},
    )
    assert deployed.metrics["weighted_loss"] == 0.15
