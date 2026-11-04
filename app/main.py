# edit 3
import json
import logging
import os
import redis
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import redis
import sklearn
from fastapi import FastAPI
from pydantic import Field, create_model

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
GIT_SHA = os.getenv("GIT_SHA", "unknown")

logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("triage")

FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

PredictRequest = create_model(
    "PredictRequest", **{name: (float, Field(ge=0)) for name in FEATURES}
)

state = {}
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
rds = None
if REDIS_HOST:
    rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                      socket_connect_timeout=1, socket_timeout=1)
    logger.info("Redis counting enabled at %s:%s", REDIS_HOST, REDIS_PORT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(MODEL_PATH)
    if not path.exists():
        logger.critical("Model artifact not found at '%s'. Train the model first.", path)
        raise RuntimeError(f"Model artifact not found: {path}")
    state["model"] = joblib.load(path)
    state["malignant_idx"] = list(state["model"].classes_).index(0)
    card_path = path.with_name("model_card.json")
    state["card"] = json.loads(card_path.read_text()) if card_path.exists() else {}
    state["version"] = os.getenv("MODEL_VERSION") or state["card"].get("model_version", "unknown")
    logger.info("Model loaded from %s (version %s, threshold %.2f)",
                path, state["version"], DECISION_THRESHOLD)
    yield


app = FastAPI(title="Breast Cancer Triage Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    values = req.model_dump()
    x = np.array([[values[n] for n in FEATURES]], dtype=float)
    p = float(state["model"].predict_proba(x)[0, state["malignant_idx"]])
    label = "malignant" if p >= DECISION_THRESHOLD else "benign"
    logger.info("prediction label=%s malignant_probability=%.4f", label, p)
    if rds is not None:
        try:
            rds.incr("predictions")
        except Exception as e:
            logger.warning("Redis unavailable, not counting: %s", e)
    return {
        "label": label,
        "malignant_probability": round(p, 4),
        "threshold": DECISION_THRESHOLD,
        "model_version": state["version"],
    }


@app.get("/model/info")
def model_info():
    card = state["card"]
    return {
        "model_version": state["version"],
        "model_family": card.get("model_family", "unknown"),
        "scikit_learn_version": sklearn.__version__,
        "git_commit": GIT_SHA,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(FEATURES),
        "test_metrics": card.get("metrics", {}),
    }



@app.get("/stats")
def stats():
    if rds is None:
        return {"predictions_served": None, "redis_enabled": False}
    try:
        value = int(rds.get("predictions") or 0)
        return {"predictions_served": value, "redis_enabled": True}
    except Exception as e:
        logger.warning("Redis unavailable: %s", e)
        return {"predictions_served": None, "redis_enabled": True, "error": "redis unreachable"}
    return {"predictions_served": None, "redis_enabled": False}