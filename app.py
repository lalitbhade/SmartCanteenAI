from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np, os, joblib, math
from datetime import date, timedelta

app = Flask(__name__)
DATA_FILE = "data/sales.csv"
MODEL_DIR = "models"
FEATURES = ["day_of_week","is_weekend","is_holiday","is_exam_day","students_present","price","lag_1","rolling_7"]

def load_data():
    return pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else pd.DataFrame()

def model_file(food):
    return os.path.join(MODEL_DIR, food.lower().replace(" ","_") + ".joblib")

def predict_food(food, students, price, holiday, exam, target):
    df=load_data()
    part=df[df.food_item.str.lower()==food.lower()].copy()
    if part.empty: return None
    part["date"]=pd.to_datetime(part["date"]); part=part.sort_values("date")
    lag=float(part.iloc[-1].quantity_sold)
    rolling=float(part.quantity_sold.tail(7).mean())
    d=pd.to_datetime(target)
    X=pd.DataFrame([{"day_of_week":d.dayofweek,"is_weekend":int(d.dayofweek>=5),
        "is_holiday":holiday,"is_exam_day":exam,"students_present":students,
        "price":price,"lag_1":lag,"rolling_7":rolling}])[FEATURES]
    path=model_file(food)
    pred=float(joblib.load(path).predict(X)[0]) if os.path.exists(path) else rolling
    pred=max(0,round(pred)); rec=math.ceil(pred*1.08)
    return {"food_item":food,"predicted_demand":pred,"recommended_preparation":rec,"buffer":rec-pred}

@app.route("/")
def index():
    df=load_data()
    foods=sorted(df.food_item.unique().tolist()) if not df.empty else []
    return render_template("index.html",foods=foods)

@app.post("/api/predict")
def api_predict():
    x=request.get_json()
    r=predict_food(x["food_item"],int(x.get("students_present",500)),float(x.get("price",20)),
                   int(x.get("is_holiday",0)),int(x.get("is_exam_day",0)),
                   x.get("target_date",str(date.today()+timedelta(days=1))))
    return (jsonify(r),200) if r else (jsonify({"error":"Food item not found."}),404)

@app.get("/api/summary")
def summary():
    df=load_data()
    if df.empty: return jsonify({"foods":[],"sales":[],"total_sales":0})
    recent=df.sort_values("date").groupby("food_item").tail(7)
    s=recent.groupby("food_item").quantity_sold.sum().sort_values(ascending=False)
    return jsonify({"foods":s.index.tolist(),"sales":s.values.tolist(),"total_sales":int(df.quantity_sold.sum())})

if __name__=="__main__": app.run(debug=True)
