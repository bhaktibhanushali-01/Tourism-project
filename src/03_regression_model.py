
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_master.csv"))

#  feature engineering 
attr_avg_rating = df.groupby("AttractionId")["Rating"].transform("mean")
df["AttractionAvgRatingHist"] = attr_avg_rating
attr_popularity = df.groupby("AttractionId")["TransactionId"].transform("count")
df["AttractionPopularity"] = attr_popularity

feature_cols = [
    "VisitYear", "VisitMonth", "VisitMode", "AttractionTypeId",
    "AttractionCityId", "AttractionCountryId", "AttractionPopularity",
]
target_col = "Rating"

X = df[feature_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
}

results = {}
best_model, best_name, best_r2 = None, None, -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"{name:20s}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    if r2 > best_r2:
        best_model, best_name, best_r2 = model, name, r2

print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

attr_lookup = df.groupby("AttractionId").agg(
    AttractionPopularity=("TransactionId", "count"),
    AttractionAvgRatingHist=("Rating", "mean"),
).reset_index()

joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "feature_cols": feature_cols,
    "metrics": results[best_name],
}, os.path.join(MODEL_DIR, "regression_model.pkl"))

attr_lookup.to_csv(os.path.join(DATA_DIR, "attraction_popularity_lookup.csv"), index=False)

print(f"\nSaved best regression model -> models/regression_model.pkl")
