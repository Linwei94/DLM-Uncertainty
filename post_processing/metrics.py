import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

class Metrics:
    def __init__(self, grades, confidence_scores):
        """
        grades: list of str — 'A' (correct), 'B' (incorrect), 'C' (not attempted), or None
        confidence_scores: list or np.array of floats, possibly containing NaN or None
        """
        self.grades = np.array(grades, dtype=object)
        self.confidence_scores = np.array(confidence_scores, dtype=float)

        # Filter out invalid entries
        valid_mask = self._get_valid_mask()
        self.grades = self.grades[valid_mask]
        self.confidence_scores = self.confidence_scores[valid_mask]

        self.acc_incl = self.compute_accuracy(include_not_attempted=True)
        self.acc_excl = self.compute_accuracy(include_not_attempted=False)
        self.roc_auc_incl = self.compute_roc_auc(include_not_attempted=True)
        self.roc_auc_excl = self.compute_roc_auc(include_not_attempted=False)
        self.ECE_incl = self.compute_ECE(include_not_attempted=True)
        self.ECE_excl = self.compute_ECE(include_not_attempted=False)

    def _get_valid_mask(self):
        """Filter out None or NaN values in both grades and confidence."""
        valid_grade_mask = np.array([g in ("A", "B", "C") for g in self.grades])
        valid_conf_mask = ~np.isnan(self.confidence_scores)
        return valid_grade_mask & valid_conf_mask

    def compute_accuracy(self, include_not_attempted=True):
        mask = np.ones(len(self.grades), dtype=bool)
        if not include_not_attempted:
            mask = self.grades != "C"
        y = self.grades[mask]
        if len(y) == 0:
            return np.nan
        return np.mean(y == "A")

    def compute_roc_auc(self, include_not_attempted=True):
        mask = np.ones(len(self.grades), dtype=bool)
        if not include_not_attempted:
            mask = self.grades != "C"
        y = self.grades[mask]
        conf = self.confidence_scores[mask]
        if len(y) == 0 or len(np.unique(y)) < 2:
            return np.nan
        y_true = (y == "A").astype(int)
        try:
            return roc_auc_score(y_true, conf)
        except ValueError:
            return np.nan

    def compute_ECE(self, include_not_attempted=True, n_bins=10):
        mask = np.ones(len(self.grades), dtype=bool)
        if not include_not_attempted:
            mask = self.grades != "C"
        y = self.grades[mask]
        conf = self.confidence_scores[mask]
        if len(y) == 0:
            return np.nan
        y_true = (y == "A").astype(int)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if np.any(bin_mask):
                acc_bin = np.mean(y_true[bin_mask])
                conf_bin = np.mean(conf[bin_mask])
                weight = np.sum(bin_mask) / len(y_true)
                ece += weight * abs(acc_bin - conf_bin)
        return ece

    def save(self, path):
        results = {
            "accuracy_including_C": self.acc_incl,
            "accuracy_excluding_C": self.acc_excl,
            "roc_auc_including_C": self.roc_auc_incl,
            "roc_auc_excluding_C": self.roc_auc_excl,
            "ECE_including_C": self.ECE_incl,
            "ECE_excluding_C": self.ECE_excl,
            "n_samples": len(self.grades)
        }
        pd.DataFrame(results, index=[0]).to_csv(path, index=False)