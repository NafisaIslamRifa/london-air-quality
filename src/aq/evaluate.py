"""Retrain a candidate on fresh data and promote it only if it beats production.

Run after: python -m aq.ingest && python -m aq.validate  (AQ_END_DATE set to a recent date)

The candidate trains on everything except the most recent HOLDOUT_DAYS days. Production and the
candidate are both scored on that same holdout; the candidate is registered and promoted only if
its MAE is lower by at least MIN_IMPROVEMENT. Either way the decision is logged to MLflow.
"""
import os

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from aq import config
from aq.features import FEATURES, TARGET, make_dataset
from aq.train import data_version

HOLDOUT_DAYS = 30
VAL_DAYS = 90
MIN_IMPROVEMENT = 0.01    # candidate must beat production MAE by at least 1%
MIN_TRAIN_ROWS = 300
MIN_HOLDOUT_ROWS = 10


def set_output(key: str, value: str):
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def say(text: str):
    print(text)
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = make_dataset(pd.read_parquet(config.DATA_PROCESSED / "daily.parquet"))
    cutoff = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    pool, holdout = df[df.index < cutoff], df[df.index >= cutoff]
    if len(pool) < MIN_TRAIN_ROWS or len(holdout) < MIN_HOLDOUT_ROWS:
        say(f"Not enough data to judge a candidate (pool={len(pool)}, holdout={len(holdout)}). "
            "Keeping the current production model.")
        set_output("promoted", "false")
        return
    val_start = pool.index.max() - pd.Timedelta(days=VAL_DAYS - 1)
    train, val = pool[pool.index < val_start], pool[pool.index >= val_start]

    candidate = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    candidate.fit(train[FEATURES], train[TARGET],
                  eval_set=[(val[FEATURES], val[TARGET])],
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    name = config.REGISTERED_MODEL_NAME
    client = MlflowClient()
    prod_version = client.get_model_version_by_alias(name, "production").version
    prod_model = mlflow.pyfunc.load_model(f"models:/{name}@production")

    X, y = holdout[FEATURES], holdout[TARGET]
    prod_pred, cand_pred = prod_model.predict(X), candidate.predict(X)
    prod_mae, cand_mae = mean_absolute_error(y, prod_pred), mean_absolute_error(y, cand_pred)
    promote = cand_mae <= prod_mae * (1 - MIN_IMPROVEMENT)
    reason = (f"candidate MAE {cand_mae:.2f} vs production {prod_mae:.2f} "
              f"({'at least' if promote else 'less than'} {MIN_IMPROVEMENT:.0%} better)")

    with mlflow.start_run(run_name="retrain-candidate"):
        mlflow.set_tags({
            "kind": "scheduled-retrain", "site": config.SITE, "data_version": data_version(),
            "gate": "promoted" if promote else "rejected", "reason": reason,
            "production_version_compared": str(prod_version)})
        mlflow.log_params({
            "model": "lightgbm", "best_iteration": candidate.best_iteration_,
            "train_rows": len(train), "holdout_rows": len(holdout),
            "holdout_days": HOLDOUT_DAYS, "data_end": str(df.index.max().date())})
        mlflow.log_metrics({
            "holdout_mae_candidate": cand_mae, "holdout_mae_production": prod_mae,
            "holdout_rmse_candidate": root_mean_squared_error(y, cand_pred),
            "holdout_rmse_production": root_mean_squared_error(y, prod_pred)})
        if promote:
            info = mlflow.lightgbm.log_model(candidate, name="model",
                                             input_example=val[FEATURES].head(5))

    if promote:
        mv = mlflow.register_model(info.model_uri, name)
        client.set_registered_model_alias(name, "production", mv.version)
        say(f"### Promoted\nNew version v{mv.version} is now `production`: {reason}.")
        set_output("new_version", str(mv.version))
    else:
        say(f"### Not promoted\nKeeping production v{prod_version}: {reason}.")
    set_output("promoted", "true" if promote else "false")


if __name__ == "__main__":
    main()