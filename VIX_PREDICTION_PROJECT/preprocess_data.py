# preprocess_data.py
import pandas as pd
from load_data import load_data

def preprocess_data():
    """Preprocess data: Convert dates, resample economic indicators to daily frequency."""
    vix_df, fedfunds_df, gdp_df, unrate_df = load_data()

    # Convert date columns to datetime format
    vix_df["DATE"] = pd.to_datetime(vix_df["DATE"])
    fedfunds_df["observation_date"] = pd.to_datetime(fedfunds_df["observation_date"])
    gdp_df["observation_date"] = pd.to_datetime(gdp_df["observation_date"])
    unrate_df["observation_date"] = pd.to_datetime(unrate_df["observation_date"])

    # Resample economic indicators to daily frequency
    fedfunds_df.set_index("observation_date", inplace=True)
    gdp_df.set_index("observation_date", inplace=True)
    unrate_df.set_index("observation_date", inplace=True)

    fedfunds_daily = fedfunds_df.resample("D").ffill()
    gdp_daily = gdp_df.resample("D").interpolate()
    unrate_daily = unrate_df.resample("D").ffill()

    # Reset index for merging
    fedfunds_daily.reset_index(inplace=True)
    gdp_daily.reset_index(inplace=True)
    unrate_daily.reset_index(inplace=True)

    # Trim VIX data to only keep DATE and CLOSE (VIX)
    vix_df = vix_df[["DATE", "CLOSE"]]
    vix_df.rename(columns={"CLOSE": "VIX_Close"}, inplace=True)

    return vix_df, fedfunds_daily, gdp_daily, unrate_daily

if __name__ == "__main__":
    vix_df, fedfunds_daily, gdp_daily, unrate_daily = preprocess_data()
    print("Data preprocessing complete!")
