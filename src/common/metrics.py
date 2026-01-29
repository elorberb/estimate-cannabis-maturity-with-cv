from collections import Counter


def compute_intersection(box1, box2):
    x1 = max(box1.minx, box2.minx)
    y1 = max(box1.miny, box2.miny)
    x2 = min(box1.maxx, box2.maxx)
    y2 = min(box1.maxy, box2.maxy)
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_box_area(box):
    return (box.maxx - box.minx) * (box.maxy - box.miny)


def compute_iou(box1, box2):
    intersection = compute_intersection(box1, box2)
    area1 = compute_box_area(box1)
    area2 = compute_box_area(box2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def compute_class_distribution(predictions):
    counts = Counter()
    for pred in predictions:
        counts[pred.category.id] += 1
    return counts


def normalize_distribution(counts):
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}
