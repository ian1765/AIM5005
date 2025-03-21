# train_models.py
from feature_engineering import create_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
import numpy as np

def train_models():
    """Train multiple regression, ridge regression, and PCR models."""
    df = create_features()
    features = ["VIX_lag1", "FEDFUNDS_lag1", "GDPC1_lag1", "UNRATE_lag1"]
    target = "VIX_tomorrow"

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)

    models = {
        "Multiple Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Principal Component Regression": LinearRegression()
    }

    results = {}
    for name, model in models.items():
        if name == "Principal Component Regression":
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        results[name] = rmse

    return results

if __name__ == "__main__":
    results = train_models()
    print("Model training complete!")
    print(results)
