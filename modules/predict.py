import pandas as pd
import dill

import os
import glob
import logging

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
MODELS_DIR = os.path.join(path, 'data','models')
TEST_DIR = os.path.join(path, 'data','test')
PRED_DIR = os.path.join(path, 'data','predictions')


def get_latest_model_path(models_dir: str) -> str:
    candidates = glob.glob(os.path.join(models_dir, "*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model .pkl files found in: {models_dir}")
    return max(candidates, key=os.path.getmtime)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return dill.load(f)


def load_test_files(test_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(test_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No test JSON files found in: {test_dir}")
    return files


def predict() -> None:
    os.makedirs(PRED_DIR, exist_ok=True)

    # берём последний .pkl
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained model found in data/models")

    latest_model = max(model_files, key=os.path.getmtime)

    with open(latest_model, "rb") as f:
        model = dill.load(f)

    logging.info(f"Loaded model: {latest_model}")

    json_files = glob.glob(os.path.join(TEST_DIR, "*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON test files found")

    results = []

    for file_path in json_files:
        df = pd.read_json(file_path, typ="series").to_frame().T

        prediction = model.predict(df)[0]

        results.append({
            "file": os.path.basename(file_path),
            "prediction": prediction
        })

    result_df = pd.DataFrame(results)

    output_path = os.path.join(
        PRED_DIR,
        f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    result_df.to_csv(output_path, index=False)

    logging.info(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    predict()