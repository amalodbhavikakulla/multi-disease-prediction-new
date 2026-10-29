"""app/services/results_service.py - reads the stored evaluation results (store D5) written by ml/train_models.py."""
import json
import os


class ResultsService:
    """Read access to the stored evaluation results."""
    def __init__(self, path):
        """Read results.json if it exists. Input: path (str)."""
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, key):
        """Input: disease key. Output: its stored results, or None."""
        return self.data.get(key)

    def comparison_rows(self, key):
        """List of dicts, one per classifier, in the order trained."""
        entry = self.data[key]
        rows = []
        for name, m in entry['models'].items():
            rows.append({'model': name, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'],
                         'f1': m['f1'], 'roc_auc': m['roc_auc'], 'cv': '%.3f ± %.3f' % (m['cv_auc_mean'], m['cv_auc_std']),
                         'best': name == entry['best']})
        return rows
