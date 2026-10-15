# Credit Ledger

### Explainable credit-risk assessment for consumer loan applications

Credit Ledger is a lightweight machine-learning application that estimates the probability that a consumer loan application will default. It combines a FastAPI prediction service with a focused browser interface for entering applicant, loan, and credit-bureau details.

The application returns a probability, applies a calibrated decision threshold, and presents the result as either **Low Risk** or **High Risk**.

> **Important:** This is a predictive demonstration and decision-support tool—not a lending decision, financial advice, or a substitute for a regulated credit-underwriting process.

## What it does

- Collects applicant, loan-request, and credit-history features through a responsive web form.
- Validates input values with Pydantic, including age and credit-history relationships.
- Loads a trained model and tuned classification threshold at application startup.
- Returns a default probability and binary risk classification through a JSON API.
- Displays the prediction with an animated probability gauge and decision-threshold marker.
- Serves the frontend and API from the same FastAPI application.
- Includes a Render deployment definition for a hosted Python web service.

## Product preview

The interface is designed as a compact underwriting desk:

1. Enter applicant information.
2. Add the requested loan details.
3. Provide prior credit-bureau information.
4. Select **Assess risk**.
5. Review the estimated default probability and model verdict.

## Technology stack

| Layer | Technology |
| --- | --- |
| API | FastAPI |
| Validation | Pydantic 2 |
| Data preparation | pandas |
| Machine learning | XGBoost / scikit-learn-compatible model interface |
| Model persistence | joblib (`.pkl`) |
| Frontend | HTML, CSS, vanilla JavaScript |
| Application server | Uvicorn |
| Deployment | Render |

## Machine-learning pipeline

The training workflow is documented in [`Credit_Risk.ipynb`](Credit_Risk.ipynb). It uses the credit-risk dataset stored in [`credit_risk_dataset.csv`](credit_risk_dataset.csv).

### Dataset profile

- **Rows:** 32,581 before validation
- **Input columns:** 11
- **Target:** `loan_status` (`0` = non-default, `1` = default)
- **Default observations:** 7,108 (`21.82%`)
- **Non-default observations:** 25,473 (`78.18%`)
- **Missing values:** `person_emp_length` (895), `loan_int_rate` (3,116)
- **Duplicate rows removed:** 165

The notebook removes duplicate records, ages outside 18–100, employment lengths above the applicant age or 60 years, and non-positive loan amounts. The cleaned data is split into training and test partitions using an 80/20 stratified split with `random_state=42`. The held-out test set contains 6,305 observations.

### Features

The model uses seven numerical features:

```text
person_age
person_income
person_emp_length
loan_amnt
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
```

and four categorical features:

```text
person_home_ownership
loan_intent
loan_grade
cb_person_default_on_file
```

### Preprocessing

Preprocessing is implemented with scikit-learn `Pipeline` and `ColumnTransformer` objects so that transformations are fitted only on the training data and reused consistently at inference time.

| Feature group | Logistic Regression pipeline | XGBoost pipeline |
| --- | --- | --- |
| Numerical columns | Median imputation → `StandardScaler` | Median imputation |
| Categorical columns | Constant imputation (`Missing`) → one-hot encoding | Constant imputation (`Missing`) → one-hot encoding |
| Unknown categories | Ignored by `OneHotEncoder(handle_unknown="ignore")` | Ignored by `OneHotEncoder(handle_unknown="ignore")` |

Because the target is imbalanced, the training workflow uses a positive-class weight of approximately `3.6312` for XGBoost (`negative_count / positive_count`) and `class_weight="balanced"` for the logistic-regression baseline.

### Model selection

The notebook evaluates a balanced logistic-regression baseline and an XGBoost classifier. XGBoost is tuned with `RandomizedSearchCV` using:

- 150 sampled hyperparameter configurations;
- five-fold `StratifiedKFold` cross-validation;
- `average_precision` as the search objective;
- parallel execution with `n_jobs=-1`.

The selected XGBoost configuration recorded in the notebook is:

```text
n_estimators:      366
max_depth:         5
learning_rate:     0.131669
subsample:         0.845304
colsample_bytree:  0.937426
min_child_weight:  7
gamma:             2.391359
```

### Cross-validation results

Mean scores across five stratified folds from the notebook:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.871 | 0.812 | 0.545 | 0.778 | 0.641 |
| XGBoost | **0.939** | **0.909** | **0.791** | **0.786** | **0.788** |

Held-out test-set classification results at the model’s default decision behavior:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.82 | 0.56 | 0.79 | 0.65 |
| Tuned XGBoost | **0.92** | **0.81** | **0.81** | **0.81** |

These numbers are reported from the checked-in notebook and should be treated as an experiment snapshot, not a guarantee of future performance.

## Project structure

```text
.
├── main.py                    # FastAPI application and /predict endpoint
├── credit_risk_model.pkl      # Trained model artifact
├── best_threshold.pkl         # Tuned probability threshold
├── credit_risk_dataset.csv    # Model-development dataset
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python runtime used by Render
├── render.yaml                # Render service configuration
└── static/
    ├── index.html             # Credit Ledger interface
    ├── script.js              # Form submission and result rendering
    └── style.css              # Application styling
```

## How predictions work

At startup, the service loads:

- `credit_risk_model.pkl` — the trained classifier
- `best_threshold.pkl` — the probability cutoff used to turn a probability into a binary prediction

For each request, the model produces a default probability. The service then evaluates:

```text
default_prediction = 1 if default_probability >= threshold else 0
```

The API labels prediction `1` as **High Risk** and prediction `0` as **Low Risk**.

### Calibration and thresholding

The notebook wraps the tuned XGBoost pipeline in `CalibratedClassifierCV` using sigmoid (Platt-style) calibration with `cv=5`, then saves that calibrated estimator to `credit_risk_model.pkl`. The probability returned by the API is therefore the calibrated model’s class-1 probability.

The notebook also searches the precision-recall curve for a probability threshold and saves it as `best_threshold.pkl`. The API applies that persisted threshold instead of assuming `0.5`, which makes the classification policy explicit and configurable.

For reproducible retraining, calculate the threshold on validation probabilities from the same final calibrated estimator that will be deployed. The checked-in notebook currently calculates the threshold before calibration, so the threshold should be revalidated whenever the model is retrained.

### Explainability

The notebook includes both global and local SHAP analysis:

- `shap.TreeExplainer` is applied to the fitted XGBoost classifier;
- the same fitted preprocessing transformer is used before calculating SHAP values;
- one-hot-expanded feature names are retained for readable explanations;
- a summary plot shows global feature influence;
- a waterfall plot explains the contribution of features for one applicant.

The current FastAPI endpoint returns risk probability and classification only. SHAP explanations are produced in the notebook and are not yet exposed through a production API route.

## Input features

The prediction endpoint accepts the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `person_age` | integer | Applicant age, 18–100 |
| `person_income` | number | Annual income |
| `person_home_ownership` | string | `RENT`, `MORTGAGE`, `OWN`, or `OTHER` |
| `person_emp_length` | number | Employment length in years, 0–60 |
| `loan_intent` | string | Loan purpose such as `PERSONAL`, `EDUCATION`, or `MEDICAL` |
| `loan_grade` | string | Loan grade from `A` through `G` |
| `loan_amnt` | number | Requested loan amount |
| `loan_int_rate` | number | Interest rate in percent |
| `loan_percent_income` | number | Loan-to-income ratio, represented as a decimal |
| `cb_person_default_on_file` | string | Prior default indicator: `Y` or `N` |
| `cb_person_cred_hist_length` | integer | Credit history length in years, 0–60 |

The API also rejects applications where employment length or credit-history length exceeds the applicant’s age.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/developerAS123/CreditRisk-Analyzer-via-SHAP.git
cd CreditRisk-Analyzer-via-SHAP
```

### 2. Create and activate a virtual environment

Python 3.12 is recommended because the pinned XGBoost version requires Python 3.12 or newer.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To execute the complete training and SHAP notebook locally, also install the notebook tooling and SHAP:

```bash
pip install jupyter shap
jupyter notebook Credit_Risk.ipynb
```

### 4. Start the application

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Interactive API documentation is available at:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## API usage

### `POST /predict`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 600000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 100000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
  }'
```

Example response shape:

```json
{
  "default_probability": 0.18,
  "default_prediction": 0,
  "threshold": 0.31,
  "Result": "Low Risk"
}
```

The exact values depend on the serialized model and threshold artifact in the repository.

## Deploy to Render

The repository includes [`render.yaml`](render.yaml), which configures a Python web service running:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

To deploy:

1. Push the repository to GitHub.
2. In Render, choose **New → Blueprint** or create a Python web service.
3. Select the repository and deploy the service defined by `render.yaml`.
4. Confirm the service uses Python 3.12. The repository’s [`runtime.txt`](runtime.txt) contains the required runtime.

If Render uses a different runtime, add this environment variable in the service settings:

```text
PYTHON_VERSION=3.12.10
```

## Model and data notes

The API expects the model artifact to accept a pandas DataFrame with the exact feature names used by `LoanApplication`. If the model is retrained, preserve:

- the feature names and categorical-value conventions;
- the preprocessing behavior used during training;
- the model serialization format;
- the threshold artifact expected by `main.py`.

The serialized model is a calibrated scikit-learn estimator containing the preprocessing transformer and XGBoost classifier. Loading it with `joblib` requires compatible versions of Python, scikit-learn, XGBoost, pandas, NumPy, and joblib. Keep the training and serving environments aligned when replacing either `.pkl` artifact.

The dataset and model files are included for this project’s reproducibility and demonstration workflow. In a production system, model artifacts should be versioned, validated, monitored, and stored separately from application code.

## Responsible use

Credit-risk predictions can affect access to financial services. Before using a system like this in production, add appropriate governance, including:

- fairness and disparate-impact evaluation;
- explainability and adverse-action support;
- privacy, security, and data-retention controls;
- model validation and drift monitoring;
- human review and regulated lending compliance;
- authentication, authorization, rate limiting, and audit logging.

Do not use the demo output as an automated approval or rejection without appropriate legal, compliance, and risk review.

## License

No license is currently specified for this repository. Add a license file before distributing or reusing the project publicly.

## Author

Built as a practical credit-risk machine-learning demonstration with FastAPI, XGBoost, and a browser-based underwriting interface.
