
# Phase 10: ML Research and Model Improvement

## 1. Introduction

This phase focuses on evaluating and improving the baseline machine learning model developed for diabetes prediction.

The experiments investigate:
- Comparison of multiple classification algorithms.
- Impact of preprocessing order.
- Handling of missing and invalid values.
- Logistic Regression hyperparameter optimization.
- Decision threshold selection.
- Independent test-set evaluation.

The objective is to improve the model's predictive performance while maintaining a scientifically defensible evaluation process.

## 2. Dataset Description

Dataset: Pima Indians Diabetes Dataset

Total records: 768

Input features: 8

Target variable: Outcome

- 0: Non-diabetic
- 1: Diabetic

The dataset contains 500 non-diabetic and 268 diabetic records.

Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI were treated as invalid or missing measurements for preprocessing experiments.

## 3. Baseline Model Comparison

Five classification algorithms were evaluated using 5-fold Stratified Cross-Validation.

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 77.21% | 71.29% | 59.34% | 64.50% | 83.95% |
| Decision Tree | 72.25% | 61.77% | 58.18% | 58.65% | 77.26% |
| Random Forest | 76.16% | 68.11% | 59.66% | 63.54% | 82.44% |
| KNN | 74.87% | 65.70% | 58.94% | 62.07% | 78.01% |
| SVM | 75.78% | 68.77% | 56.72% | 61.99% | 82.21% |

### Observation

Logistic Regression achieved the highest accuracy, F1-score, and ROC-AUC among the five baseline models.

Random Forest achieved the highest recall in this baseline comparison.

Logistic Regression was retained for further research and optimization.

## 4. Preprocessing Comparison

Three preprocessing strategies were evaluated using Logistic Regression.

| Preprocessing strategy | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Impute → IQR → Scale | 77.21% | 71.29% | 59.34% | 64.50% | 83.95% |
| IQR → Impute → Scale | 78.12% | 74.03% | 58.60% | 65.13% | 83.80% |
| Impute → Scale | 77.21% | 72.61% | 57.11% | 63.58% | 83.66% |

### Observation

The IQR-first preprocessing strategy achieved the highest accuracy, precision, and F1-score among the three experiments.

The original preprocessing strategy achieved slightly higher recall and ROC-AUC.

The differences were relatively small. Statistical significance was not established.

## 5. Insulin Missing-Value Investigation

Insulin preprocessing was investigated because a large proportion of the dataset contained zero values.

| Measurement | Result |
|---|---:|
| Total records | 768 |
| Zero insulin values | 374 |
| Percentage of zero values | 48.70% |
| Median after zero replacement | 125.0 |
| IQR before imputation | 113.75 |
| IQR after median imputation | 5.75 |

After median imputation, the interquartile range became substantially narrower.

The resulting IQR clipping boundaries were:

- Lower boundary: 112.875
- Upper boundary: 135.875

### Research finding

Median imputation before IQR calculation can reduce the observed variation in Insulin because a large proportion of observations receive the same imputed value.

This may lead to overly narrow clipping boundaries and loss of genuine variation.

This is an identified preprocessing concern, not conclusive proof that the original model's performance was harmed.

## 6. IQR-First Model Comparison

Five classification algorithms were compared using the IQR → Impute → Scale preprocessing strategy.

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 78.12% | 74.03% | 58.60% | 65.13% | 83.80% |
| Decision Tree | 71.47% | 61.06% | 57.06% | 57.63% | 76.35% |
| Random Forest | 76.04% | 68.54% | 58.58% | 63.09% | 81.97% |
| KNN | 72.39% | 62.13% | 54.46% | 57.89% | 77.22% |
| SVM | 76.30% | 70.63% | 55.22% | 61.89% | 82.37% |

### Observation

Logistic Regression achieved the highest accuracy, precision, F1-score, and ROC-AUC in this comparison.

Its recall was nearly identical to Random Forest.

It remained the strongest overall candidate for the subsequent tuning experiment.

## 7. Logistic Regression Hyperparameter Optimization

GridSearchCV was used to optimize Logistic Regression.

The search investigated:

- C: 0.01, 0.1, 1, 10, 100
- Regularization: L1 and L2
- Class weights: None and balanced
- Solver: liblinear

Five-fold Stratified Cross-Validation was used.

Total configurations: 20

Total model fits: 100

### Selected configuration

| Parameter | Selected value |
|---|---|
| C | 0.01 |
| Class weight | balanced |
| Regularization | L2 |
| Solver | liblinear |

### Training cross-validation results

| Metric | Result |
|---|---:|
| Accuracy | 74.10% |
| Precision | 60.62% |
| Recall | 73.83% |
| F1-score | 66.49% |
| ROC-AUC | 84.22% |

### Observation

The balanced class-weight configuration improved sensitivity compared with the original baseline model.

This came with a reduction in accuracy and precision.

The optimization was therefore treated as a recall-focused trade-off rather than an improvement in every metric.

## 8. Decision Threshold Analysis

The selected Logistic Regression model was evaluated using out-of-fold training probabilities.

Different classification thresholds were compared.

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.30 | 54.89% | 43.53% | 99.07% | 60.49% |
| 0.35 | 61.56% | 47.47% | 96.26% | 63.58% |
| 0.40 | 68.08% | 52.38% | 92.52% | 66.89% |
| 0.45 | 71.34% | 55.79% | 85.51% | 67.53% |
| 0.50 | 74.10% | 60.54% | 73.83% | 66.53% |
| 0.55 | 77.52% | 68.10% | 66.82% | 67.45% |
| 0.60 | 77.36% | 71.19% | 58.88% | 64.45% |
| 0.65 | 76.55% | 74.65% | 49.53% | 59.55% |
| 0.70 | 74.92% | 80.61% | 36.92% | 50.64% |

### Selected threshold: 0.45

The threshold of 0.45 achieved the highest F1-score among the evaluated thresholds.

It also retained higher recall than the default threshold of 0.50.

This threshold was selected using training-only predictions and was fixed before independent test evaluation.

## 9. Independent Test-Set Evaluation

The dataset was divided into training and independent test partitions.

| Partition | Records |
|---|---:|
| Training | 614 |
| Independent testing | 154 |
| Total | 768 |

The split used:

- Test size: 20%
- Random state: 42
- Stratified sampling

No record overlap was found.

The hyperparameter search and threshold selection used training data only.

### Final independent test results

| Metric | Result |
|---|---:|
| Accuracy | 70.13% |
| Precision | 55.00% |
| Recall / Sensitivity | 81.48% |
| Specificity | 64.00% |
| F1-score | 65.67% |
| ROC-AUC | 79.74% |

### Confusion matrix

| Actual / Predicted | Non-diabetic (0) | Diabetic (1) |
|---|---:|---:|
| Non-diabetic (0) | 64 | 36 |
| Diabetic (1) | 10 | 44 |

### Interpretation

The final model correctly identified 44 of the 54 actual positive cases.

It missed 10 positive cases and incorrectly classified 36 negative cases as positive.

The recall of 81.48% indicates that the model detected a substantial proportion of positive cases in this test partition.

The precision of 55% and specificity of 64% show that false positive predictions remain an important limitation.

## 10. Research Conclusions

The experiments produced the following findings:

1. Logistic Regression was the strongest overall baseline model in the original comparison.
2. IQR-first preprocessing produced modest improvements in selected metrics.
3. Insulin missing-value handling requires further investigation.
4. Class-weight balancing increased recall during cross-validation.
5. Decision threshold selection changed the sensitivity-specificity trade-off.
6. The selected threshold of 0.45 achieved the best F1-score among the tested training-only thresholds.
7. Independent evaluation produced 70.13% accuracy, 81.48% recall, and 79.74% ROC-AUC.

The independent test results should be reported separately from development cross-validation results.

## 11. Limitations

- The dataset contains only 768 records.
- The dataset may not represent all populations.
- The independent test set contains only 154 records.
- Statistical significance testing has not been performed.
- The threshold was selected from training out-of-fold predictions after hyperparameter selection; nested cross-validation could provide a more rigorous estimate of the model-selection process.
- Probability calibration has not yet been evaluated.
- The model is not clinically validated and must not be used as a substitute for professional medical diagnosis.

## 12. Future Research Directions

- Evaluate additional preprocessing techniques.
- Investigate alternative missing-value imputation strategies.
- Perform probability calibration.
- Evaluate confidence intervals and statistical significance.
- Explore additional classification algorithms.
- Investigate external validation on an independent dataset.
- Continue integrating explainable AI into the prediction system.

## 13. Generated Research Files

The following files contain the experimental results:

- results/cross_validation_comparison.csv
- results/preprocessing_comparison.csv
- results/model_comparison_iqr_first.csv
- results/logistic_regression_tuning.csv
- results/threshold_analysis.csv
- results/research_results_consolidated.csv
- results/independent_training_tuning.csv
- results/independent_training_threshold_analysis.csv
- results/independent_test_evaluation.csv
