"""Classification metrics with zero division defined as zero."""


def classification_metrics(actual, predicted, classes):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Expected equally sized, nonempty label lists")
    per_class = {}
    for label, name in enumerate(classes):
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        precision = tp / max(1, predicted.count(label))
        recall = tp / max(1, actual.count(label))
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
            "macro_precision": sum(m["precision"] for m in per_class.values()) / len(classes),
            "macro_recall": sum(m["recall"] for m in per_class.values()) / len(classes),
            "macro_f1": sum(m["f1"] for m in per_class.values()) / len(classes),
            "per_class": per_class}
