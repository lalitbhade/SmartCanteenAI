# SmartCanteen AI

Python Flask + Random Forest project for AI-based canteen food demand prediction.

## Run
1. Install Python 3.10+.
2. `python -m venv venv`
3. Windows: `venv\\Scripts\\activate`
4. `pip install -r requirements.txt`
5. `python train_model.py`
6. `python app.py`
7. Open `http://127.0.0.1:5000`

## Important hosting note
InfinityFree generally does not run Flask/Python server applications. Use a Python-capable host for this Flask app. If InfinityFree is compulsory, host the frontend/PHP/MySQL there and keep this Python API on a Python-capable service.

The included sales dataset is synthetic demo data. Replace it with real canteen data for actual deployment/competition claims.
