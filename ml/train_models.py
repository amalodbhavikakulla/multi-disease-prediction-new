"""
ml/train_models.py  -  Module M4 (training and comparison) and M6 (feature importance).

Run:  python -m ml.train_models
Trains five classifiers per disease on a stratified 80/20 split (random_state = 42), computes the
evaluation metrics, selects the best model by ROC-AUC (ties: F1-Score), saves the best pipeline to
ml/models/ and all results (metrics, ROC points, confusion matrix, importances) to ml/results/results.json.
"""
import json
import os
import warnings
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ml.preprocessing import SPEC, build_preprocessor, load_dataset

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), 'data')
MODEL_DIR = os.path.join(BASE, 'models')
RESULT_FILE = os.path.join(BASE, 'results', 'results.json')
TITLES = {'diabetes': 'Diabetes', 'heart': 'Heart Disease', 'liver': 'Liver Disease'}


def classifiers():
    """Return the five classifiers with the settings of Section 5.2 (a fresh set on every call)."""
    return {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
            'Decision Tree': DecisionTreeClassifier(random_state=SEED),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=SEED),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)}


def evaluate(pipe, X_test, y_test):
    """Metrics at the 0.50 threshold plus ROC points. Returns (metrics dict, probabilities)."""
    p = pipe.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    fpr, tpr, _ = roc_curve(y_test, p)
    m = dict(accuracy=accuracy_score(y_test, yhat), precision=precision_score(y_test, yhat),
             recall=recall_score(y_test, yhat), f1=f1_score(y_test, yhat), roc_auc=roc_auc_score(y_test, p),
             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
             roc=[[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)])
    return m, p


def feature_importance(key, X_train, y_train):
    """Random Forest importances, adding one-hot columns back to their original feature."""
    rf = Pipeline([('pre', build_preprocessor(key)), ('clf', RandomForestClassifier(n_estimators=100, random_state=SEED))])
    rf.fit(X_train, y_train)
    names = rf.named_steps['pre'].get_feature_names_out()
    scores = {}
    for name, value in zip(names, rf.named_steps['clf'].feature_importances_):
        base = name.split('__', 1)[1]
        for c in SPEC[key]['cat']:
            if base.startswith(c + '_'):
                base = c
        scores[base] = scores.get(base, 0.0) + float(value)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def main():
    """Train, compare and save the models of all three diseases; write results.json."""
    warnings.filterwarnings('ignore')
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    results = {}
    for key in ('diabetes', 'heart', 'liver'):
        X, y = load_dataset(key, DATA_DIR)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        entry = {'title': TITLES[key], 'n_total': len(X), 'n_train': len(X_tr), 'n_test': len(X_te),
                 'class_counts': {str(k): int(v) for k, v in y.value_counts().sort_index().items()}, 'models': {}}
        best = None
        for name, clf in classifiers().items():
            pipe = Pipeline([('pre', build_preprocessor(key)), ('clf', clf)]).fit(X_tr, y_tr)
            m, p = evaluate(pipe, X_te, y_te)
            scores = cross_val_score(Pipeline([('pre', build_preprocessor(key)), ('clf', classifiers()[name])]),
                                     X_tr, y_tr, cv=cv, scoring='roc_auc')
            m['cv_auc_mean'], m['cv_auc_std'] = float(scores.mean()), float(scores.std())
            entry['models'][name] = m
            if best is None or (m['roc_auc'], m['f1']) > (best[1]['roc_auc'], best[1]['f1']):
                best = (name, m, pipe, p)
        entry['best'] = best[0]
        entry['threshold_table'] = {str(t): {'precision': float(precision_score(y_te, best[3] >= t, zero_division=0)),
                                             'recall': float(recall_score(y_te, best[3] >= t))} for t in (.3, .4, .5, .6, .7)}
        entry['importance'] = feature_importance(key, X_tr, y_tr)
        joblib.dump(best[2], os.path.join(MODEL_DIR, key + '_best_pipeline.pkl'))
        results[key] = entry
        print(key, 'best model:', best[0], 'ROC-AUC = %.3f' % best[1]['roc_auc'])
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
