# Breast Cancer Triage Service

A containerized ML prediction service for **Meridian Diagnostics Lab**. It classifies cell-nucleus measurements (scikit-learn `load_breast_cancer`) as likely **malignant** or likely **benign**. Malignant is the positive class.

> Educational project. **Not a medical device.**

- GitHub: https://github.com/ashley400/breast-cancer-triage-service
- Docker Hub: https://hub.docker.com/r/ayeshatariq0050/breast-cancer-triage-api

## Tech stack

Python 3.11, scikit-learn, FastAPI, Redis, DVC, Docker, Docker Compose v2.

## Repository layout

```
app/                    FastAPI service
training/               prepare_data.py, train.py (+ helper scripts)
data/raw/               breast_cancer.csv.dvc (the CSV itself is tracked by DVC)
dvc-remote/             local DVC remote (committed on purpose)
dvc.yaml, dvc.lock      DVC training pipeline
params.yaml             seed and hyperparameters (seed = 42)
Dockerfile              production multi-stage image
Dockerfile.naive        baseline image used for comparison
docker-compose.yml      api + redis stack (builds locally)
docker-compose.hub.yml  api + redis stack (uses the published image)
.env.example            documented environment variables
```

## 1. Clone and set up (Windows PowerShell)

```powershell
git clone https://github.com/ashley400/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -r requirements-train.txt -r requirements-serve.txt
```

Linux/macOS/WSL: activate with `source .venv/bin/activate`.

## 2. Restore the dataset with DVC

The CSV is not stored in Git. Only the pointer file `data/raw/breast_cancer.csv.dvc` is. The data lives in the local remote `dvc-remote/`.

```powershell
dvc pull
```

The restored file is dataset version 2 (539 rows).

To recreate the dataset from scratch instead:

```powershell
python training/prepare_data.py
```

## 3. Train the model (DVC pipeline)

```powershell
dvc repro
```

This reads `params.yaml` (seed 42) and writes `artifacts/model.joblib`, `artifacts/model_card.json` and `artifacts/metrics.json`. Running it again without changes skips the stage. Changing `params.yaml` reruns it.

Train directly without DVC:

```powershell
python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

The script exits with a non-zero status if the test-set malignant recall is below 0.95.

## 4. Run the API locally (no Docker)

```powershell
uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs

Create a sample payload and call the API:

```powershell
python training/make_sample.py
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample.json"
```

## 5. Build the production image

`dvc pull` must be run first, because the training stage of the image reads the CSV.

```powershell
$sha = git rev-parse --short HEAD
docker build --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=$sha -t breast-cancer-triage-api:1.0.0 .
```

Run it on its own (Redis is optional):

```powershell
docker run -d --name api -p 8000:8000 breast-cancer-triage-api:1.0.0
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model/info
docker rm -f api
```

Override configuration at run time without rebuilding:

```powershell
docker run -d -p 8001:8000 -e DECISION_THRESHOLD=0.9 -e LOG_LEVEL=DEBUG breast-cancer-triage-api:1.0.0
```

Build the baseline image for comparison:

```powershell
docker build -f Dockerfile.naive -t naive .
```

## 6. Start the Compose stack (api + Redis)

```powershell
copy .env.example .env
# edit .env: set DOCKERHUB_USER and GIT_SHA
docker compose up -d --build
docker compose ps
```

Send predictions and read the counter:

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample.json"
curl.exe http://localhost:8000/stats
```

Stop the stack and keep the counter (named volume `redis-data`):

```powershell
docker compose down
```

Stop the stack and delete the volume (counter resets to 0):

```powershell
docker compose down -v
```

## 7. Push to Docker Hub

```powershell
docker login
docker tag breast-cancer-triage-api:1.0.0 ayeshatariq0050/breast-cancer-triage-api:1.0.0
docker tag breast-cancer-triage-api:1.0.0 ayeshatariq0050/breast-cancer-triage-api:latest
docker push ayeshatariq0050/breast-cancer-triage-api:1.0.0
docker push ayeshatariq0050/breast-cancer-triage-api:latest
```

## 8. Run from Docker Hub only (clean room)

Create an empty folder containing only `docker-compose.hub.yml` and a `.env` file, then:

```powershell
docker compose -f docker-compose.hub.yml up -d
curl.exe http://localhost:8000/health
```

Or without Compose:

```powershell
docker run -d -p 8000:8000 ayeshatariq0050/breast-cancer-triage-api:1.0.0
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}`. Does not depend on Redis. |
| POST | `/predict` | Takes 30 numeric features (all >= 0) and returns label, malignant probability, threshold and model version. Returns 422 on invalid input. |
| GET | `/model/info` | Model version, family, scikit-learn version, git commit, threshold, feature count, test metrics. |
| GET | `/stats` | Number of predictions served, counted in Redis. |

Input fields are the 30 dataset feature names with spaces replaced by underscores (`mean_radius` to `worst_fractal_dimension`).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `artifacts/model.joblib` | Location of the model artifact |
| `DECISION_THRESHOLD` | `0.5` | Malignant probability cutoff |
| `LOG_LEVEL` | `INFO` | Logging verbosity (logs go to stdout) |
| `REDIS_HOST` | unset | Enables request counting when set |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_VERSION` | build arg | Set at build time, shown in `/model/info` |
| `GIT_SHA` | build arg | Set at build time, shown in `/model/info` |

Compose variables (see `.env.example`): `DOCKERHUB_USER`, `IMAGE_TAG`, `GIT_SHA`, `HOST_PORT`, `DECISION_THRESHOLD`, `LOG_LEVEL`.

## Reproducibility

- Seed 42 is used for the train/test split, cross-validation and all model initialization (`params.yaml`).
- Python dependencies are pinned. The scikit-learn version is identical in `requirements-train.txt` and `requirements-serve.txt`.
- `.env` files and the dataset CSV are never committed. The data is restored with `dvc pull`.