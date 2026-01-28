"""Distribution calculation and aggregation utilities."""

import json
import logging
from pathlib import Path
from typing import Union

from app.backend.models import AnalysisResult, TrichomeDistribution, TrichomeType

logger = logging.getLogger(__name__)


def aggregate_distributions(
    distributions: list[TrichomeDistribution],
) -> TrichomeDistribution:
    """
    Aggregate multiple distributions into a single combined distribution.

    Args:
        distributions: List of TrichomeDistribution objects

    Returns:
        Combined TrichomeDistribution
    """
    total_clear = sum(d.clear_count for d in distributions)
    total_cloudy = sum(d.cloudy_count for d in distributions)
    total_amber = sum(d.amber_count for d in distributions)

    return TrichomeDistribution(
        clear_count=total_clear,
        cloudy_count=total_cloudy,
        amber_count=total_amber,
    )


def aggregate_results(results: list[AnalysisResult]) -> TrichomeDistribution:
    """
    Aggregate distributions from multiple analysis results.

    Args:
        results: List of AnalysisResult objects

    Returns:
        Combined TrichomeDistribution
    """
    distributions = [r.distribution for r in results if r.distribution is not None]
    return aggregate_distributions(distributions)


def save_distribution(
    distribution: TrichomeDistribution,
    output_path: Union[str, Path],
) -> None:
    """
    Save distribution to a JSON file.

    Args:
        distribution: TrichomeDistribution to save
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = distribution.to_dict()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Distribution saved to {output_path}")


def load_distribution(input_path: Union[str, Path]) -> TrichomeDistribution:
    """
    Load distribution from a JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        TrichomeDistribution object
    """
    with open(input_path) as f:
        data = json.load(f)

    counts = data.get("counts", data.get("class_distribution", {}))
    return TrichomeDistribution(
        clear_count=counts.get("clear", counts.get("1", 0)),
        cloudy_count=counts.get("cloudy", counts.get("2", 0)),
        amber_count=counts.get("amber", counts.get("3", 0)),
    )


def save_result(result: AnalysisResult, output_path: Union[str, Path]) -> None:
    """
    Save complete analysis result to a JSON file.

    Args:
        result: AnalysisResult to save
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Result saved to {output_path}")


def get_maturity_assessment(distribution: TrichomeDistribution) -> dict:
    """
    Assess cannabis maturity based on trichome distribution.

    General guidelines:
    - Mostly clear: Too early for harvest
    - Mostly cloudy: Peak THC potency
    - Cloudy + amber mix: Peak potency with more body effects
    - Mostly amber: Past peak, more sedative effects

    Args:
        distribution: TrichomeDistribution to assess

    Returns:
        Dictionary with maturity assessment
    """
    if distribution.total_count == 0:
        return {
            "stage": "unknown",
            "description": "No trichomes detected",
            "recommendation": "Unable to assess - ensure image quality",
        }

    clear_pct = distribution.clear_ratio * 100
    cloudy_pct = distribution.cloudy_ratio * 100
    amber_pct = distribution.amber_ratio * 100

    # Determine maturity stage
    if clear_pct >= 50:
        stage = "early"
        description = "Trichomes are predominantly clear"
        recommendation = "Too early for harvest. Wait for more trichomes to turn cloudy."
    elif clear_pct >= 30 and cloudy_pct >= 30:
        stage = "developing"
        description = "Mix of clear and cloudy trichomes"
        recommendation = "Approaching harvest window. Monitor closely."
    elif cloudy_pct >= 60 and amber_pct < 15:
        stage = "peak"
        description = "Predominantly cloudy trichomes with minimal amber"
        recommendation = "Optimal harvest window for maximum THC potency."
    elif cloudy_pct >= 40 and amber_pct >= 15 and amber_pct < 30:
        stage = "mature"
        description = "Good mix of cloudy and amber trichomes"
        recommendation = "Ideal harvest window for balanced effects (THC + CBN)."
    elif amber_pct >= 30:
        stage = "late"
        description = "High proportion of amber trichomes"
        recommendation = "Past peak potency. Harvest soon for more sedative effects."
    else:
        stage = "transitioning"
        description = "Mixed trichome stages"
        recommendation = "Continue monitoring trichome development."

    return {
        "stage": stage,
        "description": description,
        "recommendation": recommendation,
        "percentages": {
            "clear": round(clear_pct, 1),
            "cloudy": round(cloudy_pct, 1),
            "amber": round(amber_pct, 1),
        },
    }
