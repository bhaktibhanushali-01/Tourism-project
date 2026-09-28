
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_master.csv"))

# 1. Collaborative filtering (item-based, user-item matrix) 
user_item = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean")
item_user = user_item.fillna(0).T  # attractions x users

item_similarity = cosine_similarity(item_user)
item_sim_df = pd.DataFrame(item_similarity, index=item_user.index, columns=item_user.index)

# 2. Content-based filtering 
attr_meta = df[["AttractionId", "Attraction", "AttractionType", "AttractionTypeId",
                 "AttractionCity", "AttractionCityId"]].drop_duplicates("AttractionId").set_index("AttractionId")

content_features = pd.get_dummies(attr_meta[["AttractionTypeId", "AttractionCityId"]].astype(str))
content_similarity = cosine_similarity(content_features)
content_sim_df = pd.DataFrame(content_similarity, index=attr_meta.index, columns=attr_meta.index)

#  Popularity fallback (for cold-start users with zero history) 
popularity = df.groupby("AttractionId").agg(
    NumVisits=("TransactionId", "count"),
    AvgRating=("Rating", "mean"),
).reset_index()
popularity["Score"] = popularity["AvgRating"] * np.log1p(popularity["NumVisits"])
popularity = popularity.merge(attr_meta.reset_index(), on="AttractionId").sort_values("Score", ascending=False)

joblib.dump({
    "item_sim_df": item_sim_df,
    "content_sim_df": content_sim_df,
    "attr_meta": attr_meta,
    "popularity": popularity,
    "user_item": user_item,
}, os.path.join(MODEL_DIR, "recommendation_artifacts.pkl"))

print("Saved recommendation artifacts -> models/recommendation_artifacts.pkl")
print(f"Item-item similarity matrix shape: {item_sim_df.shape}")
print(f"Content similarity matrix shape: {content_sim_df.shape}")


# quick sanity check 
def recommend_similar_attractions(attraction_id, top_n=5, method="collaborative"):
    sim_df = item_sim_df if method == "collaborative" else content_sim_df
    if attraction_id not in sim_df.index:
        return pd.DataFrame()
    scores = sim_df[attraction_id].drop(index=attraction_id).sort_values(ascending=False).head(top_n)
    result = attr_meta.loc[scores.index].copy()
    result["SimilarityScore"] = scores.values
    return result.reset_index()


sample_id = attr_meta.index[0]
print(f"\nSample: attractions similar to '{attr_meta.loc[sample_id, 'Attraction']}' (collaborative):")
print(recommend_similar_attractions(sample_id, method="collaborative")[["Attraction", "SimilarityScore"]])
print(f"\nSample: attractions similar to '{attr_meta.loc[sample_id, 'Attraction']}' (content-based):")
print(recommend_similar_attractions(sample_id, method="content")[["Attraction", "SimilarityScore"]])
