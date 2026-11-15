import os
import json
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

class F1ParallelMLEngine:
    """
    Parallel predictive model engine and registry (XGBoost, LightGBM, Isolation Forest).
    """
    def __init__(self, registry_dir: str = "ml_engine/registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def load_model(self, model_name: str):
        model_path = os.path.join(self.registry_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        return None

    def run_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs Isolation Forest anomaly detection on lap counts / race metrics."""
        if df.empty or "laps" not in df.columns:
            return df
        
        model = IsolationForest(contamination=0.05, random_state=42)
        df["anomaly_score"] = model.fit_predict(df[["laps"]])
        return df

    def get_leaderboard(self) -> dict:
        lb_path = os.path.join(self.registry_dir, "model_leaderboard.json")
        if os.path.exists(lb_path):
            with open(lb_path, "r") as f:
                return json.load(f)
        return {"status": "No active models in registry"}