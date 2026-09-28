
import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = RAW_DIR

def load_raw():
    trans = pd.read_excel(os.path.join(RAW_DIR, "Transaction.xlsx"))
    item = pd.read_excel(os.path.join(RAW_DIR, "Item.xlsx"))
    city = pd.read_excel(os.path.join(RAW_DIR, "City.xlsx"))
    country = pd.read_excel(os.path.join(RAW_DIR, "Country.xlsx"))
    region = pd.read_excel(os.path.join(RAW_DIR, "Region.xlsx"))
    continent = pd.read_excel(os.path.join(RAW_DIR, "Continent.xlsx"))
    mode = pd.read_excel(os.path.join(RAW_DIR, "Mode.xlsx"))
    atype = pd.read_excel(os.path.join(RAW_DIR, "Type.xlsx"))
    return trans, item, city, country, region, continent, mode, atype


def clean_lookup_tables(city, country, region, continent):
    """Replace the '-' sentinel (id 0 rows) with proper 'Unknown' labels
    and strip whitespace from text columns."""
    for df, col in [(city, "CityName"), (country, "Country"),
                     (region, "Region"), (continent, "Continent")]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col] == "-", col] = "Unknown"
        df[col] = df[col].fillna("Unknown")
    
    city["CityName"] = city["CityName"].fillna("Unknown")
    return city, country, region, continent


def clean_transactions(trans):
    before = len(trans)
    trans = trans.drop_duplicates(subset="TransactionId")
    trans = trans.dropna()
    
    trans = trans[(trans["Rating"].between(1, 5)) &
                   (trans["VisitMonth"].between(1, 12)) &
                   (trans["VisitYear"].between(2000, 2026))]
    after = len(trans)
    print(f"Transactions: {before} -> {after} after cleaning ({before - after} dropped)")
    return trans.reset_index(drop=True)


def build_master(trans, item, city, country, region, continent, mode, atype):
    df = trans.merge(mode.rename(columns={"VisitModeId": "VisitMode"}),
                      on="VisitMode", how="left")
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(atype, on="AttractionTypeId", how="left")
    df = df.merge(city.rename(columns={"CityId": "AttractionCityId",
                                        "CityName": "AttractionCity",
                                        "CountryId": "AttractionCountryId"}),
                   on="AttractionCityId", how="left")
    df = df.merge(country.rename(columns={"CountryId": "AttractionCountryId",
                                           "Country": "AttractionCountry",
                                           "RegionId": "AttractionRegionId"}),
                   on="AttractionCountryId", how="left")
    df = df.merge(region.rename(columns={"RegionId": "AttractionRegionId",
                                          "Region": "AttractionRegion",
                                          "ContinentId": "AttractionContinentId"}),
                   on="AttractionRegionId", how="left")
    df = df.merge(continent.rename(columns={"ContinentId": "AttractionContinentId",
                                             "Continent": "AttractionContinent"}),
                   on="AttractionContinentId", how="left")

    df["VisitDate"] = pd.to_datetime(dict(year=df.VisitYear, month=df.VisitMonth, day=1))
    df["VisitSeason"] = df["VisitMonth"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                               3: "Spring", 4: "Spring", 5: "Spring",
                                               6: "Summer", 7: "Summer", 8: "Summer",
                                               9: "Autumn", 10: "Autumn", 11: "Autumn"})

    keep_cols = [
        "TransactionId", "UserId", "VisitYear", "VisitMonth", "VisitSeason", "VisitDate",
        "VisitMode", "AttractionId", "Attraction", "AttractionAddress",
        "AttractionTypeId", "AttractionType",
        "AttractionCityId", "AttractionCity", "AttractionCountryId", "AttractionCountry",
        "AttractionRegionId", "AttractionRegion", "AttractionContinentId", "AttractionContinent",
        "Rating",
    ]
    df = df[keep_cols].rename(columns={"VisitMode": "VisitModeLabel"})
    df = df.rename(columns={"VisitModeLabel": "VisitMode"})
    return df


def main():
    trans, item, city, country, region, continent, mode, atype = load_raw()
    city, country, region, continent = clean_lookup_tables(city, country, region, continent)
    trans = clean_transactions(trans)

    master = trans.merge(
        mode.rename(columns={"VisitModeId": "VisitMode", "VisitMode": "VisitModeName"}),
        on="VisitMode", how="left")
    master = master.merge(item, on="AttractionId", how="left")
    master = master.merge(atype, on="AttractionTypeId", how="left")
    master = master.merge(
        city.rename(columns={"CityId": "AttractionCityId", "CityName": "AttractionCity",
                              "CountryId": "AttractionCountryId"}),
        on="AttractionCityId", how="left")
    master = master.merge(
        country.rename(columns={"CountryId": "AttractionCountryId", "Country": "AttractionCountry",
                                 "RegionId": "AttractionRegionId"}),
        on="AttractionCountryId", how="left")
    master = master.merge(
        region.rename(columns={"RegionId": "AttractionRegionId", "Region": "AttractionRegion",
                                "ContinentId": "AttractionContinentId"}),
        on="AttractionRegionId", how="left")
    master = master.merge(
        continent.rename(columns={"ContinentId": "AttractionContinentId",
                                   "Continent": "AttractionContinent"}),
        on="AttractionContinentId", how="left")

    master["VisitDate"] = pd.to_datetime(dict(year=master.VisitYear, month=master.VisitMonth, day=1))
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    master["VisitSeason"] = master["VisitMonth"].map(season_map)

    corrected_geo = {
        1: {"AttractionCity": "Bali", "AttractionCountry": "Indonesia",
            "AttractionRegion": "South-East Asia", "AttractionContinent": "Asia"},
        2: {"AttractionCity": "Malang", "AttractionCountry": "Indonesia",
            "AttractionRegion": "South-East Asia", "AttractionContinent": "Asia"},
        3: {"AttractionCity": "Yogyakarta", "AttractionCountry": "Indonesia",
            "AttractionRegion": "South-East Asia", "AttractionContinent": "Asia"},
    }
    for col in ["AttractionCity", "AttractionCountry", "AttractionRegion", "AttractionContinent"]:
        master[col] = master["AttractionCityId"].map(lambda cid: corrected_geo.get(cid, {}).get(col))

    final_cols = [
        "TransactionId", "UserId", "VisitYear", "VisitMonth", "VisitSeason", "VisitDate",
        "VisitMode", "VisitModeName", "AttractionId", "Attraction", "AttractionAddress",
        "AttractionTypeId", "AttractionType",
        "AttractionCityId", "AttractionCity", "AttractionCountryId", "AttractionCountry",
        "AttractionRegionId", "AttractionRegion", "AttractionContinentId", "AttractionContinent",
        "Rating",
    ]
    master = master[final_cols]

    out_path = os.path.join(OUT_DIR, "cleaned_master.csv")
    master.to_csv(out_path, index=False)
    print(f"\nSaved cleaned master dataset -> {out_path}")
    print(f"Shape: {master.shape}")
    print(master.isnull().sum())

    city.to_csv(os.path.join(OUT_DIR, "clean_city.csv"), index=False)
    country.to_csv(os.path.join(OUT_DIR, "clean_country.csv"), index=False)
    region.to_csv(os.path.join(OUT_DIR, "clean_region.csv"), index=False)
    continent.to_csv(os.path.join(OUT_DIR, "clean_continent.csv"), index=False)
    item.to_csv(os.path.join(OUT_DIR, "clean_item.csv"), index=False)
    mode.to_csv(os.path.join(OUT_DIR, "clean_mode.csv"), index=False)
    atype.to_csv(os.path.join(OUT_DIR, "clean_type.csv"), index=False)

    geo_lookup = master[["AttractionCityId", "AttractionCity", "AttractionCountry",
                          "AttractionRegion", "AttractionContinent"]].drop_duplicates()
    geo_lookup.to_csv(os.path.join(OUT_DIR, "corrected_attraction_geo.csv"), index=False)


if __name__ == "__main__":
    main()
