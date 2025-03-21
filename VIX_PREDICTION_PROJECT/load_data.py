# load_data.py
import pandas as pd

# File paths
data_path = "data/"
files = {
    "vix": "VIX_History_Past_2_Years.csv",
    "fedfunds": "FEDFUNDS.csv",
    "gdp": "GDPC1.csv",
    "unrate": "UNRATE.csv"
}

def load_data():
    """Loads all datasets into Pandas DataFrames"""
    vix_df = pd.read_csv(data_path + files["vix"])
    fedfunds_df = pd.read_csv(data_path + files["fedfunds"])
    gdp_df = pd.read_csv(data_path + files["gdp"])
    unrate_df = pd.read_csv(data_path + files["unrate"])
    return vix_df, fedfunds_df, gdp_df, unrate_df

if __name__ == "__main__":
    vix_df, fedfunds_df, gdp_df, unrate_df = load_data()
    print("Data loaded successfully!")
