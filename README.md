# DiaPredict — Explainable AI-Based Diabetes Risk Prediction

DiaPredict is a full-stack academic major project that uses Machine Learning to estimate diabetes risk based on selected health-related input features. It combines a Logistic Regression prediction model with Explainable AI (SHAP) to help users understand the factors influencing model predictions.

The application provides user authentication, prediction history, analytics, visual explanations, and downloadable reports through a web-based interface.

> **Disclaimer:** This application is developed for academic and research purposes only. It is not a medical diagnostic tool and must not replace professional medical advice, diagnosis, or treatment.

## Key Features

- **User Authentication:** User registration, login, JWT-based authentication, and protected endpoints.
- **Diabetes Risk Prediction:** Machine Learning predictions using eight input features.
- **Explainable AI:** SHAP-based explanations of individual and global model behaviour.
- **Dashboard:** Prediction summaries and analytical visualizations.
- **Prediction History:** View, search, and filter previous predictions.
- **Prediction Reports:** Generate individual prediction PDF reports.
- **Research Reports:** Generate downloadable research reports.
- **Data Export:** Export prediction history as CSV.
- **Model Insights:** View model performance and global feature importance.
- **Admin Features:** Administrative endpoints for managing and reviewing application data.
- **Automated Testing:** Backend tests covering authentication, authorization, validation, prediction, and related services.

## Technology Stack

### Frontend
- React 19
- Vite
- JavaScript
- HTML and CSS

### Backend
- Python
- FastAPI
- SQLAlchemy
- Pydantic
- JWT Authentication

### Database
- MySQL
- PyMySQL

### Machine Learning and Explainable AI
- Scikit-learn
- Logistic Regression
- NumPy
- Pandas
- SciPy
- SHAP
- Joblib

### Reporting and Testing
- ReportLab
- Matplotlib
- Pytest
- HTTPX

## Machine Learning Model

The selected model is **Logistic Regression**.

The prediction pipeline includes:
1. Missing-value imputation.
2. IQR-based outlier clipping.
3. Feature scaling.
4. Logistic Regression classification.
5. Probability-based classification using a configurable threshold.

The current research threshold is **0.35**. This is a provisional threshold selected through academic evaluation, not a clinically validated cutoff.

### Model Evaluation

The following results were obtained during the independent test evaluation at the provisional 0.35 threshold:

| Metric | Result |
|---|---:|
| Accuracy | 72.08% |
| Precision | 58.46% |
| Recall | 70.37% |
| Specificity | 73.00% |
| F1 Score | 63.87% |
| ROC-AUC | 80.85% |

These results are from an independent test evaluation using a temporary model trained on the training portion of the dataset. They should not be interpreted as a guarantee of real-world or clinical performance.

## Project Structure

```text
diabetes-prediction-major/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   │
│   ├── tests/
│   └── practice/
│
├── frontend/
│   ├── public/
│   └── src/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── results/
├── src/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amit958345/diabetes-prediction-major.git
cd diabetes-prediction-major
```

### 2. Create and Activate a Python Virtual Environment

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure MySQL

Make sure MySQL Server is installed and running.

Create the project database:

```sql
CREATE DATABASE diabetes_prediction_major;
```

### 5. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a reference.

Update the database credentials and JWT configuration in your local `.env` file.

Use a strong, randomly generated JWT secret. Never commit your actual `.env` file or expose its credentials.

### 6. Start the Backend

From the project root:

```bash
python -m uvicorn backend.app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend development URL:

```text
http://localhost:5173
```

Keep both the backend and frontend terminals running while using the application.

## Running Automated Tests

From the project root, with the virtual environment activated:

```bash
python -m pytest -q
```

The backend test suite previously completed with **53 passed, 0 failed**. Test results may vary if the code or environment changes.

## Research and Documentation

The `reports/` directory contains project evaluation and research documentation.

The `results/` directory contains evaluation outputs and supporting result files.

## Security Notes

- Keep database credentials and JWT secrets in `.env`.
- Do not commit `.env` or other files containing secrets.
- Use authentication for protected API endpoints.
- Use appropriate production CORS origins before deployment.
- Configure HTTPS and production-grade secret management before public hosting.

## Future Improvements

- Production deployment and environment-specific configuration.
- Stronger production authentication and session management.
- Additional model validation using independent datasets.
- Further optimization of database queries and API performance.
- Expanded monitoring and model evaluation.

## Academic Project

**Project:** DiaPredict — Explainable AI-Based Diabetes Risk Prediction and Decision Support System

**Purpose:** Academic major project demonstrating full-stack development, Machine Learning integration, and Explainable AI.

---

**Important:** DiaPredict is an educational research application, not a clinically validated medical device.