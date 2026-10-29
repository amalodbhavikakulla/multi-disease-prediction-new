"""app/main.py - home page."""
from flask import Blueprint, render_template
from app.diseases.config import DISEASES

bp = Blueprint('main', __name__)


@bp.route('/')
def home():
    """Show the home page with the three disease cards."""
    return render_template('home.html', diseases=DISEASES)
