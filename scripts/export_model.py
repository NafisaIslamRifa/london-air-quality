"""Download the production model into model_baked/ so the image has a fallback copy."""
import shutil

import mlflow
from mlflow import MlflowClient

from aq import config

OUT = config.ROOT / "model_baked"


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    name = config.REGISTERED_MODEL_NAME
    version = MlflowClient().get_model_version_by_alias(name, "production").version
    shutil.rmtree(OUT, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@production", dst_path=str(OUT))
    found = next(OUT.rglob("MLmodel"), None)
    if found is None:
        raise SystemExit("MLmodel file not found after download")
    if found.parent != OUT:
        for p in found.parent.iterdir():
            shutil.move(str(p), str(OUT / p.name))
    (OUT / "model_version.txt").write_text(str(version))
    print(f"Exported {name} v{version} to {OUT}")


if __name__ == "__main__":
    main()
