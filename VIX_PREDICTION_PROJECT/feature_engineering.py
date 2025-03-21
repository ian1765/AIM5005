# feature_engineering.py
import pandas as pd
from preprocess_data import preprocess_data

def create_features():
    """Merge datasets and create lagged features for VIX prediction."""
    vix_df, fedfunds_daily, gdp_daily, unrate_daily = preprocess_data()

    # Merge datasets on DATE
    merged_df = vix_df.merge(fedfunds_daily, left_on="DATE", right_on="observation_date", how="inner")
    merged_df = merged_df.merge(gdp_daily, left_on="DATE", right_on="observation_date", how="inner")
    merged_df = merged_df.merge(unrate_daily, left_on="DATE", right_on="observation_date", how="inner")

    # Drop redundant date columns
    merged_df.drop(columns=["observation_date_x", "observation_date_y", "observation_date"], inplace=True)

    # Create target variable: Tomorrow's VIX
    merged_df["VIX_tomorrow"] = merged_df["VIX_Close"].shift(-1)

    # Create lagged features
    merged_df["VIX_lag1"] = merged_df["VIX_Close"].shift(1)
    merged_df["FEDFUNDS_lag1"] = merged_df["FEDFUNDS"].shift(1)
    merged_df["GDPC1_lag1"] = merged_df["GDPC1"].shift(1)
    merged_df["UNRATE_lag1"] = merged_df["UNRATE"].shift(1)

    # Drop last row due to NaN target value
    merged_df.dropna(inplace=True)

    return merged_df

if __name__ == "__main__":
    merged_df = create_features()
    print("Feature engineering complete!")
