import numpy as np
from datasets import load_metric


class BertBaseMultilingualMetrics:
    def __init__(self, label_names, return_entity_level_metrics=False) -> None:
        self.label_names = label_names
        self.return_entity_level_metrics = return_entity_level_metrics

    def compute_metrics(self, p):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [
                self.label_names[p]
                for (p, token_label) in zip(prediction, label)
                if token_label != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                self.label_names[token_label]
                for (p, token_label) in zip(prediction, label)
                if token_label != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        if self.return_entity_level_metrics:
            for k in results.keys():
                if k not in flattened_results.keys():
                    flattened_results[k + "_f1"] = results[k]["f1"]

        return flattened_results
