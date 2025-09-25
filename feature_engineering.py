# feature_engineering.py

def add_time_features(df):
    """
    Adds basic time features (hour, dayofweek, month) extracted from the Timestamp column.
    """
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    return df
