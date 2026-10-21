import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

def prepare_lag_features(df: pd.DataFrame, target_col: str = 'consumption_kwh', lags: int = 3) -> pd.DataFrame:
    """Generates temporal lag features for multivariate time-series forecasting."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

def train_forecaster(data_path: str):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    featured_df = prepare_lag_features(df)

    feature_cols = ['temperature_c', 'humidity_pct', 'occupancy_rate', 'hour', 'lag_1', 'lag_2', 'lag_3']
    X = featured_df[feature_cols]
    y = featured_df['consumption_kwh']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"XGBoost Validation RMSE: {rmse:.3f} kWh | MAE: {mae:.3f} kWh")
    return model

if __name__ == "__main__":
    train_forecaster("data/energy_consumption.csv")
