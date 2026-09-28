
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_master.csv"))

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close()

# 1. Rating distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Rating", hue="Rating", palette="viridis", legend=False)
plt.title("Distribution of Ratings")
savefig("01_rating_distribution.png")

# 2. Visit mode distribution
plt.figure(figsize=(6, 4))
order = df["VisitModeName"].value_counts().index
sns.countplot(data=df, y="VisitModeName", order=order, hue="VisitModeName",
              palette="mako", legend=False)
plt.title("Visits by Mode")
savefig("02_visit_mode_distribution.png")

# 3. Top attractions by number of visits
plt.figure(figsize=(7, 5))
top_attr = df["Attraction"].value_counts().head(10)
sns.barplot(x=top_attr.values, y=top_attr.index, hue=top_attr.index,
            palette="crest", legend=False)
plt.title("Top 10 Most Visited Attractions")
plt.xlabel("Number of Visits")
savefig("03_top_attractions.png")

# 4. Average rating by attraction type
plt.figure(figsize=(7, 6))
avg_rating_type = df.groupby("AttractionType")["Rating"].mean().sort_values()
sns.barplot(x=avg_rating_type.values, y=avg_rating_type.index, hue=avg_rating_type.index,
            palette="flare", legend=False)
plt.title("Average Rating by Attraction Type")
plt.xlabel("Average Rating")
savefig("04_avg_rating_by_type.png")

# 5. Visits over years
plt.figure(figsize=(7, 4))
year_counts = df.groupby("VisitYear").size()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")
plt.title("Number of Visits per Year")
plt.ylabel("Visits")
savefig("05_visits_by_year.png")

# 6. Seasonality
plt.figure(figsize=(6, 4))
season_order = ["Winter", "Spring", "Summer", "Autumn"]
sns.countplot(data=df, x="VisitSeason", order=season_order, hue="VisitSeason",
              palette="crest", legend=False)
plt.title("Visits by Season")
savefig("06_visits_by_season.png")

# 7. Rating by visit mode (box)
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="VisitModeName", y="Rating", hue="VisitModeName",
            palette="viridis", legend=False)
plt.title("Rating Distribution by Visit Mode")
savefig("07_rating_by_mode.png")

# 8. City-level popularity (top 10)
plt.figure(figsize=(7, 5))
top_city = df["AttractionCity"].value_counts().head(10)
sns.barplot(x=top_city.values, y=top_city.index, hue=top_city.index,
            palette="mako", legend=False)
plt.title("Top 10 Cities by Visit Count")
plt.xlabel("Number of Visits")
savefig("08_top_cities.png")

# 9. Correlation heatmap on numeric features
plt.figure(figsize=(5, 4))
num_df = df[["VisitYear", "VisitMonth", "VisitMode", "Rating"]]
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
savefig("09_correlation_heatmap.png")

#  printed insights 
print("KEY INSIGHTS")
print("-" * 60)
print(f"Total transactions: {len(df):,}")
print(f"Unique users: {df.UserId.nunique():,}")
print(f"Unique attractions: {df.AttractionId.nunique()}")
print(f"Date range: {df.VisitYear.min()}-{df.VisitYear.max()}")
print(f"Average rating overall: {df.Rating.mean():.2f}")
print(f"\nMost common visit mode: {df.VisitModeName.value_counts().idxmax()}")
print(f"\nTop attraction: {df.Attraction.value_counts().idxmax()}")
print(f"\nAvg rating by mode:\n{df.groupby('VisitModeName').Rating.mean().sort_values(ascending=False)}")
print(f"\nAvg rating by type (top 5):\n{avg_rating_type.sort_values(ascending=False).head()}")
print(f"\nCharts saved to {FIG_DIR}")
