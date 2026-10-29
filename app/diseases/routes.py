"""app/diseases/routes.py - Module M5/M9: one blueprint per disease, built by the same factory,
so a new disease only needs a new DISEASES entry and a saved model."""
import json
from flask import Blueprint, current_app, render_template, request
from sqlalchemy.exc import SQLAlchemyError
from flask_login import current_user, login_required
from app import db
from app.diseases.config import DISEASES
from app.diseases.validators import validate
from app.models import PredictionRecord


def make_blueprint(key):
    """Build the blueprint of one disease. Input: key (str). Output: Blueprint with the route /<key>."""
    cfg = DISEASES[key]
    bp = Blueprint(key, __name__)

    @bp.route('/' + key, methods=['GET', 'POST'], endpoint='predict')
    @login_required
    def predict():
        """GET: show the form. POST: validate, predict, save the record and show the result."""
        service = current_app.model_service
        if not service.available(key):
            return render_template('error.html', code=503, message='The model for this disease is not available.'), 503
        if request.method == 'GET':
            return render_template('form.html', key=key, cfg=cfg, values={}, errors={})
        values, errors, cautions = validate(cfg, request.form)
        if errors:
            return render_template('form.html', key=key, cfg=cfg, values=request.form, errors=errors), 400
        probability = service.predict(key, values)
        label, band, message = service.interpret(probability)
        record = PredictionRecord(user_id=current_user.user_id, disease=cfg['title'], inputs=json.dumps(values),
                                  label=label, probability=probability, model_name=service.model_name(key))
        saved = True
        try:
            db.session.add(record)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()          # keep the session usable; the user still gets the result
            current_app.logger.exception('Could not save the prediction record')
            saved = False
        return render_template('result.html', key=key, cfg=cfg, label=label, probability=probability, band=band,
                               message=message, cautions=cautions, model_name=record.model_name, saved=saved)

    return bp


def make_blueprints():
    """Build the blueprints of all diseases in DISEASES."""
    return [make_blueprint(k) for k in DISEASES]
