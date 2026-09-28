
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Paths & page config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

# Cached loaders

@st.cache_data
def load_master():
    return pd.read_csv(os.path.join(DATA_DIR, "cleaned_master.csv"))


@st.cache_data
def load_lookups():
    return {
        "mode": pd.read_csv(os.path.join(DATA_DIR, "clean_mode.csv")),
        "type": pd.read_csv(os.path.join(DATA_DIR, "clean_type.csv")),
        "city": pd.read_csv(os.path.join(DATA_DIR, "clean_city.csv")),
        "item": pd.read_csv(os.path.join(DATA_DIR, "clean_item.csv")),
        "popularity": pd.read_csv(os.path.join(DATA_DIR, "attraction_popularity_lookup.csv")),
        "geo": pd.read_csv(os.path.join(DATA_DIR, "corrected_attraction_geo.csv")),
    }


@st.cache_resource
def load_regression_model():
    return joblib.load(os.path.join(MODEL_DIR, "regression_model.pkl"))


@st.cache_resource
def load_classification_model():
    return joblib.load(os.path.join(MODEL_DIR, "classification_model.pkl"))


@st.cache_resource
def load_recommendation_artifacts():
    return joblib.load(os.path.join(MODEL_DIR, "recommendation_artifacts.pkl"))


df = load_master()
lookups = load_lookups()


# Sidebar navigation

st.sidebar.title("🧭 Tourism Analytics")
st.sidebar.caption("Classification · Prediction · Recommendation")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Explore the Data", "⭐ Predict Rating",
     "🧳 Predict Visit Mode", "🎯 Get Recommendations", "ℹ️ About this Project"],
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Dataset**: 52,930 transactions · 33,530 users · 30 attractions "
    "across Bali & Java, Indonesia (2013–2022)."
)


# PAGE: Overview

if page == "🏠 Overview":
    st.title("Tourism Experience Analytics")
    st.subheader("Classification, Prediction & Recommendation System")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Users", f"{df.UserId.nunique():,}")
    c3.metric("Attractions Covered", f"{df.AttractionId.nunique()}")
    c4.metric("Average Rating", f"{df.Rating.mean():.2f} / 5")

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Ratings Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.countplot(data=df, x="Rating", hue="Rating", palette="viridis", legend=False, ax=ax)
        st.pyplot(fig)

    with colB:
        st.markdown("#### Visits by Mode")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        order = df.VisitModeName.value_counts().index
        sns.countplot(data=df, y="VisitModeName", order=order, hue="VisitModeName",
                      palette="mako", legend=False, ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### What can you do here?")
    st.markdown(
        """
        - **📊 Explore the Data** — interactive charts on attractions, cities, seasonality, and ratings.
        - **⭐ Predict Rating** — estimate the rating a visit is likely to receive, given the context.
        - **🧳 Predict Visit Mode** — predict whether a visit looks like Business / Family / Couples / Friends / Solo.
        - **🎯 Get Recommendations** — collaborative-filtering and content-based attraction suggestions.
        """
    )


# PAGE: Explore the Data

elif page == "📊 Explore the Data":
    st.title("📊 Explore the Data")

    tab1, tab2, tab3, tab4 = st.tabs(["Attractions", "Geography", "Time Trends", "Ratings & Modes"])

    with tab1:
        st.markdown("#### Top Attractions by Visit Count")
        top_n = st.slider("Number of attractions to show", 5, 30, 10, key="attr_slider")
        top_attr = df.Attraction.value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
        sns.barplot(x=top_attr.values, y=top_attr.index, hue=top_attr.index,
                    palette="crest", legend=False, ax=ax)
        ax.set_xlabel("Number of Visits")
        st.pyplot(fig)

        st.markdown("#### Average Rating by Attraction Type")
        avg_type = df.groupby("AttractionType").Rating.mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=avg_type.values, y=avg_type.index, hue=avg_type.index,
                    palette="flare", legend=False, ax=ax)
        ax.set_xlabel("Average Rating")
        st.pyplot(fig)

    with tab2:
        st.markdown("#### Top Cities by Visit Count")
        top_city = df.AttractionCity.value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_city.values, y=top_city.index, hue=top_city.index,
                    palette="mako", legend=False, ax=ax)
        ax.set_xlabel("Number of Visits")
        st.pyplot(fig)

        st.markdown("#### Visits by Country")
        country_counts = df.AttractionCountry.value_counts()
        st.bar_chart(country_counts)

    with tab3:
        st.markdown("#### Visits per Year")
        year_counts = df.groupby("VisitYear").size()
        st.line_chart(year_counts)

        st.markdown("#### Visits by Season")
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.countplot(data=df, x="VisitSeason", order=season_order, hue="VisitSeason",
                      palette="crest", legend=False, ax=ax)
        st.pyplot(fig)

    with tab4:
        st.markdown("#### Rating Distribution by Visit Mode")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=df, x="VisitModeName", y="Rating", hue="VisitModeName",
                    palette="viridis", legend=False, ax=ax)
        st.pyplot(fig)

        st.markdown("#### Correlation Heatmap (numeric features)")
        num_df = df[["VisitYear", "VisitMonth", "VisitMode", "Rating"]]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

# PAGE: Predict Rating (Regression)

elif page == "⭐ Predict Rating":
    st.title("⭐ Predict Attraction Rating")
    st.caption("Estimate the rating a visit is likely to receive based on visit context.")

    reg_bundle = load_regression_model()
    model = reg_bundle["model"]
    feature_cols = reg_bundle["feature_cols"]
    metrics = reg_bundle["metrics"]

    with st.expander("Model performance (held-out test set)"):
        st.write(f"**Model:** {reg_bundle['model_name']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R2']:.3f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
        m3.metric("MAE", f"{metrics['MAE']:.3f}")
        st.caption(
            "Note: no user-demographic table was available for this dataset, so the model "
            "relies on visit context + attraction features only. R² is modest as a result — "
            "adding user profile data would meaningfully improve this."
        )

    item = lookups["item"]
    type_df = lookups["type"]
    mode_df = lookups["mode"]
    pop_lookup = lookups["popularity"]

    col1, col2 = st.columns(2)
    with col1:
        attraction_name = st.selectbox("Attraction", item.Attraction.sort_values().unique())
        visit_mode_name = st.selectbox("Visit Mode", mode_df[mode_df.VisitModeId != 0].VisitMode)
        visit_year = st.number_input("Visit Year", min_value=2013, max_value=2030, value=2023)
    with col2:
        visit_month = st.slider("Visit Month", 1, 12, 6)

    if st.button("Predict Rating", type="primary"):
        attr_row = item[item.Attraction == attraction_name].iloc[0]
        attraction_id = attr_row.AttractionId
        mode_id = int(mode_df[mode_df.VisitMode == visit_mode_name].VisitModeId.iloc[0])
        pop_row = pop_lookup[pop_lookup.AttractionId == attraction_id]
        popularity = int(pop_row.AttractionPopularity.iloc[0]) if len(pop_row) else 1

        X_new = pd.DataFrame([{
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "VisitMode": mode_id,
            "AttractionTypeId": attr_row.AttractionTypeId,
            "AttractionCityId": attr_row.AttractionCityId,
            "AttractionCountryId": lookups["city"][lookups["city"].CityId == attr_row.AttractionCityId].CountryId.iloc[0],
            "AttractionPopularity": popularity,
        }])[feature_cols]

        pred = model.predict(X_new)[0]
        pred_clamped = float(np.clip(pred, 1, 5))

        st.success(f"**Predicted Rating: {pred_clamped:.2f} / 5**")
        st.progress(pred_clamped / 5)

        hist_avg = pop_row.AttractionAvgRatingHist.iloc[0] if len(pop_row) else df.Rating.mean()
        st.caption(f"Historical average rating for {attraction_name}: {hist_avg:.2f} / 5")

# ------------------------------------------------------------------
# PAGE: Predict Visit Mode (Classification)
# ------------------------------------------------------------------
elif page == "🧳 Predict Visit Mode":
    st.title("🧳 Predict Visit Mode")
    st.caption("Predict whether a visit looks like Business, Couples, Family, Friends, or Solo travel.")

    clf_bundle = load_classification_model()
    model = clf_bundle["model"]
    feature_cols = clf_bundle["feature_cols"]
    metrics = clf_bundle["metrics"]
    mode_lookup = clf_bundle["mode_lookup"]

    with st.expander("Model performance (held-out test set)"):
        st.write(f"**Model:** {clf_bundle['model_name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['Precision']:.3f}")
        m3.metric("Recall", f"{metrics['Recall']:.3f}")
        m4.metric("F1-score", f"{metrics['F1']:.3f}")

    item = lookups["item"]
    pop_lookup = lookups["popularity"]

    col1, col2 = st.columns(2)
    with col1:
        attraction_name = st.selectbox("Attraction", item.Attraction.sort_values().unique(), key="clf_attr")
        rating = st.slider("Rating given", 1, 5, 4)
        visit_year = st.number_input("Visit Year", min_value=2013, max_value=2030, value=2023, key="clf_year")
    with col2:
        visit_month = st.slider("Visit Month", 1, 12, 6, key="clf_month")

    if st.button("Predict Visit Mode", type="primary"):
        attr_row = item[item.Attraction == attraction_name].iloc[0]
        attraction_id = attr_row.AttractionId
        pop_row = pop_lookup[pop_lookup.AttractionId == attraction_id]
        popularity = int(pop_row.AttractionPopularity.iloc[0]) if len(pop_row) else 1

        X_new = pd.DataFrame([{
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "Rating": rating,
            "AttractionTypeId": attr_row.AttractionTypeId,
            "AttractionCityId": attr_row.AttractionCityId,
            "AttractionCountryId": lookups["city"][lookups["city"].CityId == attr_row.AttractionCityId].CountryId.iloc[0],
            "AttractionPopularity": popularity,
        }])[feature_cols]

        pred_id = model.predict(X_new)[0]
        pred_label = mode_lookup[mode_lookup.VisitModeId == pred_id].VisitMode.iloc[0]

        st.success(f"**Predicted Visit Mode: {pred_label}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            classes = model.classes_
            labels = [mode_lookup[mode_lookup.VisitModeId == c].VisitMode.iloc[0] for c in classes]
            proba_df = pd.DataFrame({"Visit Mode": labels, "Probability": proba}).sort_values(
                "Probability", ascending=False)
            st.markdown("#### Prediction Probabilities")
            st.bar_chart(proba_df.set_index("Visit Mode"))

# PAGE: Recommendations

elif page == "🎯 Get Recommendations":
    st.title("🎯 Get Attraction Recommendations")

    rec = load_recommendation_artifacts()
    item_sim_df = rec["item_sim_df"]
    content_sim_df = rec["content_sim_df"]
    attr_meta = rec["attr_meta"]
    popularity = rec["popularity"]
    user_item = rec["user_item"]

    rec_tab1, rec_tab2, rec_tab3 = st.tabs(
        ["👤 By Existing User", "🏛️ Similar to an Attraction", "🔥 Trending / Popular"])

    with rec_tab1:
        st.markdown("Recommend attractions for a **known user** using collaborative filtering "
                     "(based on similar users' rating patterns).")
        user_ids = user_item.index.tolist()
        selected_user = st.selectbox("Select a User ID", user_ids[:2000],
                                      help="Showing a sample of user IDs from the dataset.")
        top_n = st.slider("Number of recommendations", 3, 10, 5, key="user_topn")

        if st.button("Recommend for this User", type="primary"):
            user_ratings = user_item.loc[selected_user].dropna()
            if user_ratings.empty:
                st.warning("This user has no rating history — showing popular attractions instead.")
                st.dataframe(popularity[["Attraction", "AttractionType", "AttractionCity",
                                          "AvgRating", "NumVisits"]].head(top_n), hide_index=True)
            else:
                visited = user_ratings.index.tolist()
                scores = pd.Series(dtype=float)
                for aid, r in user_ratings.items():
                    sims = item_sim_df[aid].drop(index=visited, errors="ignore")
                    scores = scores.add(sims * r, fill_value=0)
                scores = scores.sort_values(ascending=False).head(top_n)
                result = attr_meta.loc[scores.index].copy()
                result["MatchScore"] = scores.values
                st.markdown(f"#### Recommended for User `{selected_user}`")
                st.dataframe(result.reset_index()[["Attraction", "AttractionType", "AttractionCity", "MatchScore"]],
                             hide_index=True)
                st.caption(f"Based on this user's {len(visited)} previously rated attraction(s).")

    with rec_tab2:
        st.markdown("Find attractions **similar to one you already like** — works for new/anonymous users too.")
        chosen_attraction = st.selectbox("I liked...", attr_meta.Attraction.sort_values().unique(), key="content_attr")
        method = st.radio("Similarity method", ["Content-Based (type & location)", "Collaborative (co-rated by users)"],
                           horizontal=True)
        top_n2 = st.slider("Number of recommendations", 3, 10, 5, key="content_topn")

        if st.button("Find Similar Attractions", type="primary"):
            attraction_id = attr_meta[attr_meta.Attraction == chosen_attraction].index[0]
            sim_df = content_sim_df if method.startswith("Content") else item_sim_df
            scores = sim_df[attraction_id].drop(index=attraction_id).sort_values(ascending=False).head(top_n2)
            result = attr_meta.loc[scores.index].copy()
            result["SimilarityScore"] = scores.values
            st.dataframe(result.reset_index()[["Attraction", "AttractionType", "AttractionCity", "SimilarityScore"]],
                         hide_index=True)

    with rec_tab3:
        st.markdown("Overall most popular & highly-rated attractions (cold-start fallback).")
        top_n3 = st.slider("Show top", 3, 30, 10, key="pop_topn")
        st.dataframe(
            popularity[["Attraction", "AttractionType", "AttractionCity", "AvgRating", "NumVisits"]].head(top_n3),
            hide_index=True,
        )

# PAGE: About

elif page == "ℹ️ About this Project":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
### Tourism Experience Analytics: Classification, Prediction, and Recommendation System

**Domain:** Tourism

**Objective:** Analyze tourist visit data to (1) predict the rating a user might give
an attraction, (2) classify the likely mode of a visit, and (3) recommend attractions
users are likely to enjoy.

**Pipeline followed:**
1. Data Cleaning — merged 8 source tables (Transaction, Item, City, Country, Region,
   Continent, Type, Mode), handled placeholder `-` values, validated ranges.
2. Exploratory Data Analysis — rating distributions, attraction popularity, seasonality,
   geographic patterns.
3. Feature Engineering — attraction popularity, visit season, categorical encoding.
4. Model Training — Regression (Rating), Classification (Visit Mode), Recommendation
   (collaborative + content-based filtering).
5. Deployment — this Streamlit application.

**⚠️ Known data limitation:** the project brief describes a *User* table (with each
user's Continent / Region / Country / City) that was **not included** in the files
provided for this build. As a result, models are trained on visit-context and
attraction features only — no user demographic features. This caps regression/
classification accuracy below what's achievable with full user profiles; if that
table becomes available, re-run `src/03_regression_model.py` and
`src/04_classification_model.py` after adding the demographic columns to the
feature list for a meaningful accuracy boost.

**🔧 Data quality fix applied:** `Item.xlsx`'s `AttractionCityId` (values 1, 2, 3)
does **not** reference the same id space as `City.xlsx`'s `CityId` — joining them
directly (as the raw schema suggests) silently resolves to the wrong cities/countries
(e.g. Cameroon/Chad instead of Indonesia). This was caught by cross-checking against
the `AttractionAddress` text field, which confirmed the true geography is **Bali,
Malang, and Yogyakarta — all Indonesia**. The cleaning script overrides the
human-readable geography with this corrected mapping (see `corrected_attraction_geo.csv`
and the comment block in `01_data_cleaning.py`); the raw numeric ids are left
untouched everywhere they're used as model features.

**Tech stack:** Python, pandas, scikit-learn, LightGBM, Streamlit, seaborn/matplotlib.
        """
    )
