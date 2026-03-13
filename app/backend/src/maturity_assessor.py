from __future__ import annotations

from src.models import MaturityStage, TrichomeType


class MaturityAssessor:
    @staticmethod
    def assess(
        distribution: dict[TrichomeType, int],
        avg_green_ratio: float,
        avg_orange_ratio: float,
    ) -> tuple[MaturityStage, str]:
        total = sum(distribution.values())
        if total == 0:
            return MaturityStage.DEVELOPING, "Monitor trichomes closely."

        clear_pct = distribution.get(TrichomeType.CLEAR, 0) / total * 100
        cloudy_pct = distribution.get(TrichomeType.CLOUDY, 0) / total * 100
        amber_pct = distribution.get(TrichomeType.AMBER, 0) / total * 100

        if clear_pct > 70:
            return MaturityStage.EARLY, "Plant is in early development. Continue growing."
        if 40 <= clear_pct <= 70:
            return MaturityStage.DEVELOPING, "Plant is developing. Monitor trichomes."
        if cloudy_pct > 50 and amber_pct < 15:
            return MaturityStage.PEAK, "Plant is at peak maturity. Harvest soon."
        if 15 <= amber_pct <= 30:
            return MaturityStage.MATURE, "Plant is mature. Optimal harvest window."
        if amber_pct > 30:
            return MaturityStage.LATE, "Plant is past peak. Harvest immediately."
        return MaturityStage.DEVELOPING, "Monitor trichomes closely."
