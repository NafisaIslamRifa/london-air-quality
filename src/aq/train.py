"""Train persistence baseline, Ridge and LightGBM; log to MLflow; register the best."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from mlflow import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aq import config
from aq.features import FEATURES, TARGET, make_dataset


def data_version() -> str:
    """md5 of daily.parquet from dvc.lock, so every run traces to an exact dataset."""
    try:
        lock = yaml.safe_load((config.ROOT / "dvc.lock").read_text())
        for out in lock["stages"]["validate"]["outs"]:
            if out["path"].endswith("daily.parquet"):
                return out["md5"]
    except (OSError, KeyError, TypeError):
        pass
    return "unknown"


def time_split(df):
    test_start = df.index.max() - pd.Timedelta(days=182)       # last ~6 months
    val_start = test_start - pd.Timedelta(days=183)            # previous ~6 months
    return (df[df.index < val_start],
            df[(df.index >= val_start) & (df.index < test_start)],
            df[df.index >= test_start])


def metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred))}


def plot_pred_vs_actual(y, pred, title):
    idx = y.index + pd.Timedelta(days=1)                       # date being forecast
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, y.values, label="actual", lw=1.3)
    ax.plot(idx, pred, label="predicted", lw=1.3)
    ax.set(title=title, ylabel="NO₂ (µg/m³)")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_importance(model):
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=FEATURES)
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax)
    ax.set(title="LightGBM feature importance (gain)")
    fig.tight_layout()
    plt.close(fig)
    return fig


def run(name, params, predict, model, log_model, data, version):
    """Log one run. Returns (results, model_uri)."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"data_version": version, "site": config.SITE,
                         "target": "next-day NO2"})
        mlflow.log_params(params)
        results = {}
        for split, (X, y) in data.items():
            results[split] = metrics(y, predict(X))
            mlflow.log_metrics({f"{split}_mae": results[split]["mae"],
                                f"{split}_rmse": results[split]["rmse"]})
        X_test, y_test = data["test"]
        mlflow.log_figure(plot_pred_vs_actual(y_test, predict(X_test), f"{name}: test set"),
                          "test_pred_vs_actual.png")
        model_uri = None
        if model is not None:
            if name == "lightgbm":
                mlflow.log_figure(plot_importance(model), "feature_importance.png")
            model_uri = log_model(model, name="model",
                                  input_example=data["val"][0].head(5)).model_uri
    return results, model_uri


def main():
    if config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        print("Tracking to", config.MLFLOW_TRACKING_URI)
    else:
        print("WARNING: no tracking URI set - logging to local ./mlruns")
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = make_dataset(pd.read_parquet(config.DATA_PROCESSED / "daily.parquet"))
    train, val, test = time_split(df)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")
    data = {s: (d[FEATURES], d[TARGET]) for s, d in
            {"train": train, "val": val, "test": test}.items()}
    version = data_version()
    results, uris = {}, {}

    # 1) persistence baseline: tomorrow = today
    results["persistence"], _ = run(
        "persistence", {"model": "yesterday's value carried forward"},
        lambda X: X["no2_lag0"].to_numpy(), None, None, data, version)

    # 2) Ridge (alpha chosen on validation MAE)
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    best_alpha = min([0.1, 1, 10, 100], key=lambda a: mean_absolute_error(
        y_va, make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_tr, y_tr).predict(X_va)))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(X_tr, y_tr)
    results["ridge"], uris["ridge"] = run(
        "ridge", {"model": "ridge", "alpha": best_alpha}, ridge.predict, ridge,
        mlflow.sklearn.log_model, data, version)

    # 3) LightGBM (early stopping on validation)
    lgbm = LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=15,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=42, verbose=-1)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
             callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    params = {"model": "lightgbm", "best_iteration": lgbm.best_iteration_,
              **{k: lgbm.get_params()[k] for k in
                 ["learning_rate", "num_leaves", "min_child_samples",
                  "subsample", "colsample_bytree"]}}
    results["lightgbm"], uris["lightgbm"] = run(
        "lightgbm", params, lgbm.predict, lgbm, mlflow.lightgbm.log_model, data, version)

    # compare + register the best trained model (chosen on validation MAE)
    print(f"\n{'model':<12}{'val MAE':>9}{'test MAE':>10}{'test RMSE':>11}")
    for n, r in results.items():
        print(f"{n:<12}{r['val']['mae']:>9.2f}{r['test']['mae']:>10.2f}{r['test']['rmse']:>11.2f}")
    gain = 1 - results["lightgbm"]["test"]["mae"] / results["persistence"]["test"]["mae"]
    print(f"\nLightGBM vs persistence, test MAE improvement: {gain:.1%}")

    best = min(uris, key=lambda n: results[n]["val"]["mae"])
    mv = mlflow.register_model(uris[best], config.REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(config.REGISTERED_MODEL_NAME, "production", mv.version)
    print(f"Registered {best} as {config.REGISTERED_MODEL_NAME} v{mv.version} (alias: production)")


if __name__ == "__main__":
    main()