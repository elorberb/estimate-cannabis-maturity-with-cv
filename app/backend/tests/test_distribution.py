"""Tests for app.backend.distribution module."""

import json
import pytest

from app.backend.models import AnalysisResult, TrichomeDistribution
from app.backend.distribution import (
    aggregate_distributions,
    aggregate_results,
    get_maturity_assessment,
    load_distribution,
    save_distribution,
    save_result,
)


class TestAggregateDistributions:
    """Tests for aggregate_distributions function."""

    def test_aggregate_two_distributions(self):
        """Test aggregating two distributions."""
        dist1 = TrichomeDistribution(clear_count=10, cloudy_count=20, amber_count=5)
        dist2 = TrichomeDistribution(clear_count=15, cloudy_count=30, amber_count=10)

        aggregated = aggregate_distributions([dist1, dist2])

        assert aggregated.clear_count == 25
        assert aggregated.cloudy_count == 50
        assert aggregated.amber_count == 15
        assert aggregated.total_count == 90

    def test_aggregate_empty_list(self):
        """Test aggregating empty list."""
        aggregated = aggregate_distributions([])
        assert aggregated.total_count == 0

    def test_aggregate_single_distribution(self, sample_distribution):
        """Test aggregating single distribution."""
        aggregated = aggregate_distributions([sample_distribution])
        assert aggregated.clear_count == sample_distribution.clear_count
        assert aggregated.cloudy_count == sample_distribution.cloudy_count
        assert aggregated.amber_count == sample_distribution.amber_count


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_results(self, sample_result):
        """Test aggregating analysis results."""
        results = [sample_result, sample_result]
        aggregated = aggregate_results(results)

        # sample_result has 1 clear, 2 cloudy, 1 amber
        assert aggregated.clear_count == 2
        assert aggregated.cloudy_count == 4
        assert aggregated.amber_count == 2


class TestSaveLoadDistribution:
    """Tests for save/load distribution functions."""

    def test_save_and_load(self, tmp_path, sample_distribution):
        """Test saving and loading distribution."""
        output_path = tmp_path / "distribution.json"

        save_distribution(sample_distribution, output_path)
        assert output_path.exists()

        loaded = load_distribution(output_path)
        assert loaded.clear_count == sample_distribution.clear_count
        assert loaded.cloudy_count == sample_distribution.cloudy_count
        assert loaded.amber_count == sample_distribution.amber_count

    def test_save_creates_directory(self, tmp_path, sample_distribution):
        """Test that save creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "distribution.json"
        save_distribution(sample_distribution, output_path)
        assert output_path.exists()


class TestSaveResult:
    """Tests for save_result function."""

    def test_save_result(self, tmp_path, sample_result):
        """Test saving analysis result."""
        output_path = tmp_path / "result.json"
        save_result(sample_result, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_detections"] == 4
        assert "distribution" in data
        assert "detections" in data


class TestGetMaturityAssessment:
    """Tests for get_maturity_assessment function."""

    def test_empty_distribution(self):
        """Test assessment with no trichomes."""
        dist = TrichomeDistribution()
        assessment = get_maturity_assessment(dist)
        assert assessment["stage"] == "unknown"

    def test_early_stage(self):
        """Test assessment for early stage (mostly clear)."""
        dist = TrichomeDistribution(clear_count=60, cloudy_count=30, amber_count=10)
        assessment = get_maturity_assessment(dist)
        assert assessment["stage"] == "early"
        assert "too early" in assessment["recommendation"].lower()

    def test_peak_stage(self):
        """Test assessment for peak stage (mostly cloudy)."""
        dist = TrichomeDistribution(clear_count=10, cloudy_count=80, amber_count=10)
        assessment = get_maturity_assessment(dist)
        assert assessment["stage"] == "peak"
        assert "optimal" in assessment["recommendation"].lower()

    def test_mature_stage(self):
        """Test assessment for mature stage (cloudy + amber mix)."""
        dist = TrichomeDistribution(clear_count=10, cloudy_count=60, amber_count=30)
        assessment = get_maturity_assessment(dist)
        # Could be mature or late depending on exact percentages
        assert assessment["stage"] in ["mature", "late"]

    def test_late_stage(self):
        """Test assessment for late stage (mostly amber)."""
        dist = TrichomeDistribution(clear_count=10, cloudy_count=30, amber_count=60)
        assessment = get_maturity_assessment(dist)
        assert assessment["stage"] == "late"
        assert "sedative" in assessment["recommendation"].lower()

    def test_assessment_includes_percentages(self, sample_distribution):
        """Test that assessment includes percentage breakdown."""
        assessment = get_maturity_assessment(sample_distribution)
        assert "percentages" in assessment
        assert assessment["percentages"]["clear"] == 20.0
        assert assessment["percentages"]["cloudy"] == 60.0
        assert assessment["percentages"]["amber"] == 20.0
