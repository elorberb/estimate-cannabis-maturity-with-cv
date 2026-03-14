from __future__ import annotations

from cannabis_maturity.models import MaturityStage, TrichomeType


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

        stigma_late = avg_orange_ratio > 0.7
        stigma_early = avg_green_ratio > 0.7

        if clear_pct > 70 or (clear_pct > 50 and stigma_early):
            return MaturityStage.EARLY, "Plant is in early development. Continue growing."
        if 40 <= clear_pct <= 70:
            return MaturityStage.DEVELOPING, "Plant is developing. Monitor trichomes."
        if cloudy_pct > 50 and amber_pct < 15 and not stigma_late:
            return MaturityStage.PEAK, "Plant is at peak maturity. Harvest soon."
        if 15 <= amber_pct <= 30 or (cloudy_pct > 50 and stigma_late):
            return MaturityStage.MATURE, "Plant is mature. Optimal harvest window."
        if amber_pct > 30 or stigma_late:
            return MaturityStage.LATE, "Plant is past peak. Harvest immediately."
        return MaturityStage.DEVELOPING, "Monitor trichomes closely."
