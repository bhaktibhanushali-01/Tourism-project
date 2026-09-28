# Tourism Experience Analytics
### Classification, Prediction, and Recommendation System

An end-to-end data science project analyzing tourist visit data to power three
capabilities: Rating prediction (regression), visit-mode prediction
(classification), and attraction recommendations — all served through a Streamlit web application.



## 📁 Project Structure

```
tourism_project/
├── data/                          # raw + cleaned datasets
│   ├── *.xlsx                     # original source files (8 tables)
│   ├── cleaned_master.csv         # merged, cleaned analytical dataset
│   ├── clean_*.csv                # cleaned lookup tables
│   └── attraction_popularity_lookup.csv
├── src/                           # pipeline scripts (run in order)
│   ├── 01_data_cleaning.py        # merge + clean all 8 raw tables
│   ├── 02_eda.py                  # generates charts to docs/figures/
│   ├── 03_regression_model.py     # trains & saves rating prediction model
│   ├── 04_classification_model.py # trains & saves visit-mode model
│   └── 05_recommendation_engine.py# builds collaborative + content-based recommenders
├── models/                        # trained model artifacts (.pkl)
├── docs/
│   ├── figures/                   # EDA chart PNGs
│   └── Project_Report.md          # full write-up: approach, findings, insights
├── app/
│   └── app.py                     # Streamlit application (multi-page)
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Run the full pipeline (optional — models are already trained)

```bash
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_regression_model.py
python src/04_classification_model.py
python src/05_recommendation_engine.py
```

## 🖥️ Launch the app

```bash
streamlit run app/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## 📊 App pages

| Page | What it does |
|---|---|
| 🏠 Overview | Key metrics & headline charts |
| 📊 Explore the Data | Interactive EDA — attractions, geography, time trends, ratings |
| ⭐ Predict Rating | Regression model — estimate a visit's likely rating |
| 🧳 Predict Visit Mode | Classification model — Business/Couples/Family/Friends/Solo |
| 🎯 Get Recommendations | Collaborative filtering (by user), content-based (by attraction), and popularity fallback |
| ℹ️ About | Project write-up and known data limitations |

## ⚠️ Known data limitation

The project brief describes a **User** lookup table (`UserId` → Continent /
Region / Country / City) that was **not included** in the files provided for
this build — only `Transaction`, `Item`, `Type`, `Mode`, `City`, `Country`,
`Region`, and `Continent` were supplied. As a result:

- Models are trained on **visit context + attraction features only** — no
  user demographic features. This is the main reason regression R² and
  classification accuracy are moderate rather than high.
- If the User table becomes available, add `ContinentId/RegionId/CountryId/
  CityId` (of the user, not the attraction) to `feature_cols` in
  `03_regression_model.py` and `04_classification_model.py` and retrain —
  expect a meaningful accuracy improvement.
- The recommendation engine is **unaffected** by this gap: collaborative
  filtering only needs the user-item rating matrix, which is complete.

## 🧰 Tech Stack

Python · pandas · scikit-learn · LightGBM · Streamlit · seaborn/matplotlib · joblib
