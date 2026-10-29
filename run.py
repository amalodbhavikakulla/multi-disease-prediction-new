"""run.py - entry point. Local:  python run.py     Cloud:  gunicorn run:app
The server listens on this computer only (127.0.0.1) unless HOST is set, e.g. HOST=0.0.0.0 to serve a LAN."""
import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 5000)), debug=False)
