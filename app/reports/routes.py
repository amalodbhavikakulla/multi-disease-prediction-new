"""app/reports/routes.py - Module M8: CSV reports."""
import csv
import io
from flask import Blueprint, Response, current_app, render_template
from flask_login import current_user, login_required
from app.diseases.config import DISEASES
from app.models import PredictionRecord
from app.security import professional_required

bp = Blueprint('reports', __name__)


def _csv_response(header, rows, filename):
    """Build a CSV download. Input: header, rows, filename. Output: Flask Response."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return Response(buf.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=' + filename})


@bp.route('/reports')
@login_required
def index():
    """Show the list of reports the user may download."""
    return render_template('reports.html')


@bp.route('/reports/prediction.csv')
@login_required
def prediction_report():
    """A General User receives only their own records; a Healthcare Professional receives all records."""
    query = PredictionRecord.query.order_by(PredictionRecord.created_at)
    if not current_user.is_professional():
        query = query.filter_by(user_id=current_user.user_id)
    rows = [(r.prediction_id, r.disease, r.inputs, r.label, round(r.probability, 4), r.model_name,
             r.created_at.strftime('%Y-%m-%d %H:%M:%S')) for r in query]
    return _csv_response(['prediction_id', 'disease', 'inputs', 'label', 'probability', 'model_name', 'created_at'],
                         rows, 'prediction_report.csv')


@bp.route('/reports/comparison.csv')
@professional_required
def comparison_report():
    """CSV of the model-comparison results (Healthcare Professional only)."""
    rs, rows = current_app.results_service, []
    for key, cfg in DISEASES.items():
        if rs.get(key):
            for r in rs.comparison_rows(key):
                rows.append((cfg['title'], r['model'], round(r['accuracy'], 4), round(r['precision'], 4), round(r['recall'], 4),
                             round(r['f1'], 4), round(r['roc_auc'], 4), r['cv'], 'yes' if r['best'] else ''))
    return _csv_response(['disease', 'model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_roc_auc', 'selected'],
                         rows, 'model_comparison_report.csv')


@bp.route('/reports/importance.csv')
@professional_required
def importance_report():
    """CSV of the feature-importance ranking (Healthcare Professional only)."""
    rs, rows = current_app.results_service, []
    for key, cfg in DISEASES.items():
        if rs.get(key):
            for rank, (feature, score) in enumerate(rs.get(key)['importance'].items(), 1):
                rows.append((cfg['title'], rank, feature, round(score, 4)))
    return _csv_response(['disease', 'rank', 'feature', 'importance'], rows, 'feature_importance_report.csv')
