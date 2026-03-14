from __future__ import annotations

from cannabis_maturity.maturity_assessor import MaturityAssessor
from cannabis_maturity.models import MaturityStage, TrichomeType


def test_early_stage() -> None:
    dist = {TrichomeType.CLEAR: 80, TrichomeType.CLOUDY: 20, TrichomeType.AMBER: 0}
    stage, _ = MaturityAssessor.assess(dist, 0.5, 0.5)
    assert stage == MaturityStage.EARLY


def test_developing_stage() -> None:
    dist = {TrichomeType.CLEAR: 50, TrichomeType.CLOUDY: 50, TrichomeType.AMBER: 0}
    stage, _ = MaturityAssessor.assess(dist, 0.5, 0.5)
    assert stage == MaturityStage.DEVELOPING


def test_peak_stage() -> None:
    dist = {TrichomeType.CLOUDY: 60, TrichomeType.AMBER: 10, TrichomeType.CLEAR: 30}
    stage, _ = MaturityAssessor.assess(dist, 0.5, 0.5)
    assert stage == MaturityStage.PEAK


def test_mature_stage() -> None:
    dist = {TrichomeType.AMBER: 20, TrichomeType.CLOUDY: 60, TrichomeType.CLEAR: 20}
    stage, _ = MaturityAssessor.assess(dist, 0.5, 0.5)
    assert stage == MaturityStage.MATURE


def test_late_stage() -> None:
    dist = {TrichomeType.AMBER: 40, TrichomeType.CLOUDY: 40, TrichomeType.CLEAR: 20}
    stage, _ = MaturityAssessor.assess(dist, 0.5, 0.5)
    assert stage == MaturityStage.LATE


def test_empty_distribution() -> None:
    stage, msg = MaturityAssessor.assess({}, 0.0, 0.0)
    assert stage == MaturityStage.DEVELOPING
    assert msg == "Monitor trichomes closely."
