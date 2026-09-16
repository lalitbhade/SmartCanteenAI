from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np, os, joblib, math
from datetime import date, timedelta

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "sales.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURES = ["day_of_week", "is_weekend", "is_holiday", "is_exam_day", "students_present", "price", "lag_1", "rolling_7"]


def load_data():
    return pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else pd.DataFrame()


def model_file(food):
    return os.path.join(MODEL_DIR, food.lower().replace(" ", "_") + ".joblib")


def get_food_history(food, limit=10):
    df = load_data()
    if df.empty:
        return []
    part = df[df["food_item"].str.lower() == food.lower()].copy()
    if part.empty:
        return []
    part = part.sort_values("date", ascending=False).head(limit)
    return part[["date", "quantity_sold", "students_present", "price", "is_holiday", "is_exam_day"]].to_dict("records")


def build_prediction_inputs(food, students, price, holiday, exam, target):
    df = load_data()
    part = df[df.food_item.str.lower() == food.lower()].copy()
    if part.empty:
        return None
    part["date"] = pd.to_datetime(part["date"])
    part = part.sort_values("date")
    lag = float(part.iloc[-1].quantity_sold)
    rolling = float(part.quantity_sold.tail(7).mean())
    d = pd.to_datetime(target)
    values = {
        "day_of_week": int(d.dayofweek),
        "is_weekend": int(d.dayofweek >= 5),
        "is_holiday": int(holiday),
        "is_exam_day": int(exam),
        "students_present": int(students),
        "price": float(price),
        "lag_1": lag,
        "rolling_7": rolling,
    }
    return values


def predict_food(food, students, price, holiday, exam, target):
    inputs = build_prediction_inputs(food, students, price, holiday, exam, target)
    if inputs is None:
        return None
    X = pd.DataFrame([inputs])[FEATURES]
    path = model_file(food)
    if os.path.exists(path):
        model = joblib.load(path)
        pred = float(model.predict(X)[0])
        model_name = "Random Forest Regression"
        model_status = "Trained model loaded"
        trees = getattr(model, "n_estimators", None)
    else:
        pred = float(inputs["rolling_7"])
        model_name = "7-day rolling average fallback"
        model_status = "Model file not found"
        trees = None
    pred = max(0, round(pred))
    buffer = math.ceil(pred * 0.08)
    rec = pred + buffer
    return {
        "food_item": food,
        "predicted_demand": pred,
        "recommended_preparation": rec,
        "buffer": buffer,
        "buffer_percent": 8,
        "features": inputs,
        "model": model_name,
        "model_status": model_status,
        "trees": trees,
    }


@app.route("/")
def index():
    df = load_data()
    foods = sorted(df.food_item.unique().tolist()) if not df.empty else []
    return render_template("index.html", foods=foods)


@app.post("/api/predict")
def api_predict():
    x = request.get_json(silent=True) or {}
    food = x.get("food_item", "")
    try:
        r = predict_food(
            food,
            int(x.get("students_present", 500)),
            float(x.get("price", 20)),
            int(x.get("is_holiday", 0)),
            int(x.get("is_exam_day", 0)),
            x.get("target_date", str(date.today() + timedelta(days=1))),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return (jsonify(r), 200) if r else (jsonify({"error": "Food item not found."}), 404)


@app.get("/api/summary")
def summary():
    df = load_data()
    if df.empty:
        return jsonify({"foods": [], "sales": [], "total_sales": 0, "records": 0})
    recent = df.sort_values("date").groupby("food_item").tail(7)
    s = recent.groupby("food_item").quantity_sold.sum().sort_values(ascending=False)
    return jsonify({
        "foods": s.index.tolist(),
        "sales": s.values.tolist(),
        "total_sales": int(df.quantity_sold.sum()),
        "records": int(len(df)),
    })


@app.get("/api/historical")
def historical():
    df = load_data()
    if df.empty:
        return jsonify({"rows": []})
    df = df.sort_values("date", ascending=False).head(20).copy()
    return jsonify({"rows": df.to_dict("records")})


@app.get("/api/features/<food>")
def features(food):
    df = load_data()
    if df.empty:
        return jsonify({"error": "No historical data."}), 404
    part = df[df.food_item.str.lower() == food.lower()].copy()
    if part.empty:
        return jsonify({"error": "Food item not found."}), 404
    part["date"] = pd.to_datetime(part["date"])
    part = part.sort_values("date")
    last = part.iloc[-1]
    return jsonify({
        "food_item": food,
        "features": {
            "day_of_week": int(last.date.dayofweek),
            "is_weekend": int(last.date.dayofweek >= 5),
            "is_holiday": int(last.is_holiday),
            "is_exam_day": int(last.is_exam_day),
            "students_present": int(last.students_present),
            "price": float(last.price),
            "lag_1": float(last.quantity_sold),
            "rolling_7": float(part.quantity_sold.tail(7).mean()),
        },
    })


@app.get("/api/model/<food>")
def model_info(food):
    path = model_file(food)
    if not os.path.exists(path):
        return jsonify({"food_item": food, "available": False})
    model = joblib.load(path)
    return jsonify({
        "food_item": food,
        "available": True,
        "algorithm": "Random Forest Regression",
        "estimators": getattr(model, "n_estimators", None),
        "features": FEATURES,
    })


@app.get("/api/history/<food>")
def history(food):
    return jsonify({"food_item": food, "rows": get_food_history(food)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
