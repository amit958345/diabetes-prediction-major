# DiaPredict — Model Validation Report

## 1. Introduction

This report documents the testing and validation activities performed for the DiaPredict diabetes prediction system.

The project uses a Logistic Regression machine learning pipeline to generate diabetes risk predictions. Model evaluation includes cross-validation, independent hold-out testing, classification threshold analysis, SHAP explainability verification, and frontend-backend integration testing.

The purpose of this validation is to evaluate software functionality and experimental model performance.

## 2. Model Configuration

**Selected Algorithm:** Logistic Regression

**Preprocessing Pipeline:**
1. Median Imputation
2. IQR-based Outlier Clipping
3. Standard Scaling
4. Logistic Regression Classification

**Dataset:** Pima Indians Diabetes Dataset

**Input Features:** 8

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

**Classification Threshold:** 0.35 (provisional research threshold)

## 3. Cross-Validation Results

Five-fold Stratified Cross-Validation was performed to compare five machine learning algorithms.

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 77.21% | 71.29% | 59.34% | 64.50% | 83.95% |
| Decision Tree | 72.25% | 61.77% | 58.18% | 58.65% | 77.26% |
| Random Forest | 76.16% | 68.11% | 59.66% | 63.54% | 82.44% |
| KNN | 74.87% | 65.70% | 58.94% | 62.07% | 78.01% |
| SVM | 75.78% | 68.77% | 56.72% | 61.99% | 82.21% |

### Observation

Logistic Regression achieved the highest overall accuracy, F1 score, and ROC-AUC among the evaluated models.

Therefore, Logistic Regression was selected as the primary model for the application.

Random Forest achieved slightly higher recall, which is also an important consideration for diabetes risk prediction.

## 4. Independent Test Evaluation

An 80:20 stratified train-test split was used.

- Training records: 614
- Testing records: 154
- Random state: 42

The temporary evaluation pipeline used the same preprocessing order and Logistic Regression configuration as the production pipeline.

### Results at the default 0.50 threshold

| Metric | Result |
|---|---:|
| Accuracy | 71.43% |
| Precision | 60.87% |
| Recall | 51.85% |
| F1 Score | 56.00% |
| ROC-AUC | 80.85% |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 82 | 18 |
| Actual Positive | 26 | 28 |

The model correctly identified 28 positive cases and missed 26 positive cases at the default classification threshold.

## 5. Classification Threshold Analysis

Threshold analysis was conducted using out-of-fold training predictions.

The threshold comparison showed that 0.35 achieved the highest training out-of-fold F1 score.

| Threshold | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 0.30 | 75.73% | 61.99% | 78.50% | 69.28% |
| 0.35 | 77.52% | 66.10% | 72.90% | 69.33% |
| 0.40 | 78.34% | 69.19% | 68.22% | 68.71% |
| 0.45 | 79.15% | 72.87% | 64.02% | 68.16% |
| 0.50 | 78.34% | 73.68% | 58.88% | 65.45% |

### Selected Threshold: 0.35

The 0.35 threshold was selected provisionally because it achieved the highest training out-of-fold F1 score.

### Held-out Test Comparison

| Metric | Threshold 0.35 | Threshold 0.50 |
|---|---:|---:|
| Accuracy | 72.08% | 71.43% |
| Precision | 58.46% | 60.87% |
| Recall | 70.37% | 51.85% |
| F1 Score | 63.87% | 56.00% |
| Specificity | 73.00% | 82.00% |
| ROC-AUC | 80.85% | 80.85% |

### Confusion Matrix Comparison

**Threshold 0.35**

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 73 | 27 |
| Actual Positive | 16 | 38 |

**Threshold 0.50**

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 82 | 18 |
| Actual Positive | 26 | 28 |

### Observation

The 0.35 threshold improved recall and F1 score on the held-out test set, while reducing specificity and precision.

The threshold therefore represents a trade-off between identifying more positive cases and generating more false-positive predictions.

The threshold is a provisional academic research choice and has not been clinically validated.

**Evaluation note:** The held-out threshold comparison used a temporary model trained on the 614 training records. The saved production model was trained on the full dataset. These evaluation results should not be interpreted as an exact independent evaluation of the full-data saved model.

## 6. SHAP Explainable AI Validation

SHAP (SHapley Additive exPlanations) was used to explain individual model predictions and feature contributions.

A consistency test was performed using a sample input.

### Verification Results

| Parameter | Result |
|---|---:|
| Decision Score | -1.622773 |
| SHAP Reconstructed Score | -1.622773 |
| Difference | 0.0 |
| Verification | PASSED |

The SHAP reconstructed model output matched the Logistic Regression decision score for the tested input.

This confirms additive consistency for the tested sample.

SHAP contributions describe how features influence the model output. They do not establish medical causation.

## 7. Saved Model Inference Testing

The saved Logistic Regression pipeline was successfully loaded using Joblib.

The inference test verified:

- Successful model loading.
- Successful feature preparation.
- Successful probability generation.
- Successful classification output.

The test demonstrated that the saved model could generate predictions without retraining.

## 8. FastAPI Backend Integration Testing

The prediction endpoint was tested using FastAPI Swagger UI.

### Test Results

- HTTP Status Code: 200 OK
- Predicted Class: 1
- Class 0 Probability: 31.33%
- Class 1 Probability: 68.67%
- Classification Threshold: 0.35
- Predicted Class Probability: 68.67%

The backend returned the prediction and SHAP explanation successfully.

The API result matched the direct Python inference result for the same input.

## 9. React Frontend Integration Testing

The React frontend was tested using the same input supplied to the FastAPI endpoint.

The following components were verified:

- Prediction result display.
- Predicted class display.
- Class 0 probability.
- Class 1 probability.
- Predicted class probability.
- SHAP baseline value.
- SHAP model output.
- Individual feature contributions.

The frontend successfully displayed the prediction and its explanation.

## 10. Threshold Logic Testing

Two synthetic software-test inputs were used to verify threshold classification.

| Test | Class 1 Probability | Threshold | Predicted Class | Result |
|---|---:|---:|---:|---|
| Test 1 | 16.48% | 35% | 0 | Passed |
| Test 2 | 68.67% | 35% | 1 | Passed |

Both threshold conditions were verified successfully.

## 11. Limitations

1. The evaluation was performed using a single dataset.
2. External dataset validation has not yet been performed.
3. The selected threshold is provisional and not clinically validated.
4. The held-out evaluation was performed on a temporary model trained on the training split.
5. Model predictions must not be interpreted as medical diagnoses.
6. SHAP explanations indicate model behavior and do not establish causal relationships.
7. Further testing is required before considering real-world healthcare deployment.

## 12. Conclusion

The DiaPredict system successfully completed its core software validation activities.

Cross-validation demonstrated that Logistic Regression performed competitively among the evaluated algorithms.

Threshold analysis indicated that a classification threshold of 0.35 improved recall and F1 score in the held-out comparison, with a corresponding reduction in specificity.

SHAP consistency, saved model inference, FastAPI integration, and React frontend integration tests were also completed successfully.

The results support the functional implementation of the academic research prototype. Further external validation, security assessment, performance testing, and clinical evaluation would be necessary before any real-world healthcare application.

---

**Project:** DiaPredict — Diabetes Prediction Using Machine Learning

**Validation Phase:** Phase 15 — Testing and Model Validation

**Report Type:** Academic Project Validation Report

**Disclaimer:** This system is developed for academic research and educational purposes only. It is not a medical diagnostic tool and must not replace professional medical advice.