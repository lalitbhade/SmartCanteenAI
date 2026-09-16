import pandas as pd, joblib, os
from sklearn.ensemble import RandomForestRegressor
DATA="data/sales.csv"; OUT="models"
features=["day_of_week","is_weekend","is_holiday","is_exam_day","students_present","price","lag_1","rolling_7"]
df=pd.read_csv(DATA); df["date"]=pd.to_datetime(df.date); df=df.sort_values(["food_item","date"])
df["day_of_week"]=df.date.dt.dayofweek; df["is_weekend"]=(df.day_of_week>=5).astype(int)
df["lag_1"]=df.groupby("food_item").quantity_sold.shift(1)
df["rolling_7"]=df.groupby("food_item").quantity_sold.transform(lambda s:s.shift(1).rolling(7,min_periods=1).mean())
df["lag_1"]=df.lag_1.fillna(df.quantity_sold.median()); df["rolling_7"]=df.rolling_7.fillna(df.quantity_sold.median())
os.makedirs(OUT,exist_ok=True)
for food in df.food_item.unique():
    p=df[df.food_item==food]
    m=RandomForestRegressor(n_estimators=250,random_state=42,min_samples_leaf=2)
    m.fit(p[features],p.quantity_sold)
    joblib.dump(m,f"{OUT}/{food.lower().replace(' ','_')}.joblib")
print("All models trained.")
