# 🏠 House Price Predictor – An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help you master the art of building and operationalizing machine learning pipelines.

## You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.

## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/
│   └── mlflow/             # Docker Compose setup for MLflow
├── models/                 # Trained models and preprocessors
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You’re here!
```

---

## 🛠️ Setting up Learning/Development Environment

To begin, ensure the following tools are installed on your system:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) or your preferred editor
- [UV – Python package and environment manager](https://github.com/astral-sh/uv)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) **or** [Podman Desktop](https://podman-desktop.io/)

---

## 🚀 Preparing Your Environment

1. **Fork this repo** on GitHub.

2. **Clone your forked copy:**

   ```bash
   # Replace xxxxxx with your GitHub username or org
   git clone https://github.com/xxxxxx/house-price-predictor.git
   cd house-price-predictor
   ```

3. **Setup Python Virtual Environment using UV:**

   ```bash
   uv venv --python python3.11
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

---

## 📊 Setup MLflow for Experiment Tracking

To track experiments and model runs:

```bash
cd deployment/mlflow
docker compose -f docker-compose.yaml up -d
docker compose ps
```

> 🐧 **Using Podman?** Use this instead:

```bash
podman compose -f docker-compose.yaml up -d
podman compose ps
```

Access the MLflow UI at [http://localhost:5555](http://localhost:5555)

---

## 📒 Using JupyterLab (Optional)

If you prefer an interactive experience, launch JupyterLab with:

```bash
uv python -m jupyterlab
# or
python -m jupyterlab
```

---

## 🔁 Model Workflow

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Feature Engineering

Apply transformations and generate features:

```bash
python src/features/engineer.py   --input data/processed/cleaned_house_data.csv   --output data/processed/featured_house_data.csv   --preprocessor models/trained/preprocessor.pkl
```

---

### 📈 Step 3: Modeling & Experimentation

Train your model and log everything to MLflow:

```bash
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/featured_house_data.csv   --models-dir models   --mlflow-tracking-uri http://localhost:5555
```

`configs/model_config.yaml` is written by the last cell of `notebooks/03_experimentation.ipynb`: the winning model,
its hyperparameters, the RFE features and the `quality_gate` thresholds (minimum R², maximum MAPE, beat the median
baseline) that the training script must enforce on the fixed holdout.

For the lab comparison, both models used the same 61 training rows, 16 holdout rows,
15 features and `random_state=42`. RandomForest used 50 trees. GradientBoosting used
the parameters in `configs/model_config.yaml`, including 200 estimators.

| Model | Holdout R² | MAE (USD) | RMSE (USD) |
| --- | ---: | ---: | ---: |
| RandomForest | 0.9635 | 22,867.08 | 41,234.17 |
| GradientBoosting | 0.9817 | 14,122.07 | 29,227.99 |

The production config keeps GradientBoosting because it had higher R² and lower
errors in this comparison. The holdout has 16 houses, so the result applies to this split.

For the API contract check, run `python -m pytest -q tests/test_api_contract.py`.
The test checks that the prediction schema rejects a request missing `sqft`.

---

## Building FastAPI and Streamlit

The code for both the apps are available in `src/api` and `streamlit_app` already. To build and launch these apps

- Add a `Dockerfile` in the root of the source code for building FastAPI
- Add `streamlit_app/Dockerfile` to package and build the Streamlit app
- Add `docker-compose.yaml` in the root path to launch both these apps. be sure to provide `API_URL=http://fastapi:8000` in the streamlit app's environment.

Once you have launched both the apps, you should be able to access streamlit web ui and make predictions.

You could also test predictions with FastAPI directly using

```
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "suburban",
  "year_built": 2000,
  "condition": fair
}'

```

Be sure to replace `http://localhost:8000/predict` with actual endpoint based on where its running.

---

## 🤝 Contributing

We welcome contributions, issues, and suggestions to make this project even better. Feel free to fork, explore, and raise PRs!

---

Happy Learning!  
— Team **School of DevOps**
