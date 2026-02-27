from collections import Counter


class Metrics:
    @staticmethod
    def compute_intersection(box1, box2) -> float:
        x1 = max(box1.minx, box2.minx)
        y1 = max(box1.miny, box2.miny)
        x2 = min(box1.maxx, box2.maxx)
        y2 = min(box1.maxy, box2.maxy)
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def compute_box_area(box) -> float:
        return (box.maxx - box.minx) * (box.maxy - box.miny)

    @staticmethod
    def compute_iou(box1, box2) -> float:
        intersection = Metrics.compute_intersection(box1, box2)
        area1 = Metrics.compute_box_area(box1)
        area2 = Metrics.compute_box_area(box2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    @staticmethod
    def compute_class_distribution(predictions: list) -> Counter:
        counts: Counter = Counter()
        for pred in predictions:
            counts[pred.category.id] += 1
        return counts

    @staticmethod
    def normalize_distribution(counts: Counter) -> dict[int, float]:
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}
