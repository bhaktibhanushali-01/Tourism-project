
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_master.csv"))

attr_popularity = df.groupby("AttractionId")["TransactionId"].transform("count")
df["AttractionPopularity"] = attr_popularity

feature_cols = [
    "VisitYear", "VisitMonth", "Rating", "AttractionTypeId",
    "AttractionCityId", "AttractionCountryId", "AttractionPopularity",
]
target_col = "VisitMode"  # numeric id 1-5

X = df[feature_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42,
                                            class_weight="balanced", n_jobs=-1),
}
if HAS_LGBM:
    candidates["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=8, random_state=42, verbose=-1)

results = {}
best_model, best_name, best_acc = None, None, -1

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name:20s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    if acc > best_acc:
        best_model, best_name, best_acc = model, name, acc

print(f"\nBest model: {best_name} (Accuracy={best_acc:.4f})")
print("\nClassification report (best model):")
print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

mode_lookup = pd.read_csv(os.path.join(DATA_DIR, "clean_mode.csv"))

joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "feature_cols": feature_cols,
    "metrics": results[best_name],
    "mode_lookup": mode_lookup,
}, os.path.join(MODEL_DIR, "classification_model.pkl"))

print(f"\nSaved best classification model -> models/classification_model.pkl")
