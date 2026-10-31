# ==========================================
# DIABETES PREDICTION - LOGISTIC REGRESSION
# ==========================================

# Import reusable preprocessing function
from preprocessing import preprocess_data

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import Accuracy, Confusion Matrix, Precision, Recall,
# F1 and ROC-AUC metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------
# 1. Get preprocessed data
# ------------------------------------------

X_train, X_test, y_train, y_test = preprocess_data()


# ------------------------------------------
# 2. Create Logistic Regression model
# ------------------------------------------

model = LogisticRegression(random_state=42)


# ------------------------------------------
# 3. Train the model
# ------------------------------------------

model.fit(X_train, y_train)


# ------------------------------------------
# 4. Make predictions on test data
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# 5. Calculate Accuracy
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------
# 6. Calculate Confusion Matrix
# ------------------------------------------
# The confusion matrix shows:
# TN = True Negative
# FP = False Positive
# FN = False Negative
# TP = True Positive
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------
# 7. Calculate Precision, Recall and F1
# ------------------------------------------

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# ------------------------------------------
# 8. Calculate ROC-AUC
# ------------------------------------------
# ROC-AUC uses the probability of class 1
# rather than the final 0/1 prediction.
# ------------------------------------------

y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

# ------------------------------------------
# 9. Display Logistic Regression Results
# ------------------------------------------

print("\n====================================")
print("LOGISTIC REGRESSION RESULTS")
print("====================================")

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

print("Precision:", precision)
print("Precision (%):", precision * 100)

print("Recall:", recall)
print("Recall (%):", recall * 100)

print("F1-Score:", f1)
print("F1-Score (%):", f1 * 100)

print("ROC-AUC:", roc_auc)
print("ROC-AUC (%):", roc_auc * 100)