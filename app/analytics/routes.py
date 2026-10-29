"""app/analytics/routes.py - Modules M6 and M7: EDA, model comparison, feature importance and dashboard.
All pages are restricted to Healthcare Professionals."""
from flask import Blueprint, abort, current_app, render_template, request
from sqlalchemy import func
from app import db
from app.diseases.config import DISEASES
from app.models import PredictionRecord
from app.security import professional_required
from ml.preprocessing import load_dataset

bp = Blueprint('analytics', __name__)
_eda_cache = {}


def _disease_arg(default='diabetes'):
    """Read the 'disease' query parameter. Output: a valid disease key; 404 if unknown."""
    key = request.args.get('disease', default)
    if key not in DISEASES:
        abort(404)
    return key


@bp.route('/eda')
@professional_required
def eda_index():
    """List the datasets available for exploratory analysis."""
    return render_template('eda_index.html', diseases=DISEASES)


@bp.route('/eda/<key>')
@professional_required
def eda(key):
    """Class distribution, class-wise means and correlation with the target for one disease."""
    if key not in DISEASES:
        abort(404)
    if key not in _eda_cache:
        X, y = load_dataset(key, current_app.config['DATA_DIR'])
        X = X.copy()
        if 'gender' in X:
            X['gender'] = (X['gender'] == 'Male').astype(int)
        corr = X.corrwith(y).round(3)
        rows = [{'feature': c, 'mean0': round(float(X.loc[y == 0, c].mean()), 2), 'mean1': round(
            float(X.loc[y == 1, c].mean()), 2), 'corr': float(corr[c])} for c in X.columns]
        _eda_cache[key] = {
            'counts': {
                str(k): int(v) for k,
                v in y.value_counts().sort_index().items()},
            'rows': rows,
            'n': len(y)}
    return render_template('eda.html', key=key, cfg=DISEASES[key], data=_eda_cache[key])


@bp.route('/analytics/comparison')
@professional_required
def comparison():
    """Show the five-model comparison table and chart for one disease."""
    key = _disease_arg()
    rs = current_app.results_service
    if rs.get(key) is None:
        abort(404)
    return render_template('comparison.html', key=key, diseases=DISEASES, entry=rs.get(key), rows=rs.comparison_rows(key))


@bp.route('/analytics/importance')
@professional_required
def importance():
    """Show the Random Forest feature-importance chart for one disease."""
    key = _disease_arg()
    entry = current_app.results_service.get(key)
    if entry is None:
        abort(404)
    top = list(entry['importance'].items())[:10]
    return render_template('importance.html', key=key, diseases=DISEASES, entry=entry, top=top)


@bp.route('/analytics/dashboard')
@professional_required
def dashboard():
    """Collect the data for the four dashboard charts and show the dashboard."""
    rs = current_app.results_service
    classes, cm, roc = {}, {}, {}
    for key, cfg in DISEASES.items():
        entry = rs.get(key)
        if entry:
            classes[cfg['title']] = entry['class_counts']
            best = entry['models'][entry['best']]
            cm[cfg['title']] = [best['tn'], best['fp'], best['fn'], best['tp']]
            roc[cfg['title']] = best['roc']
    counts = dict(db.session.query(PredictionRecord.disease, func.count(PredictionRecord.prediction_id))
                  .group_by(PredictionRecord.disease).all())
    return render_template('dashboard.html', classes=classes, cm=cm, roc=roc, counts=counts)
