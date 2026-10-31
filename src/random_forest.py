# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# RANDOM FOREST CLASSIFIER
# ==========================================


# Import our reusable preprocessing function
from preprocessing import preprocess_data


# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# STEP 1: LOAD PREPROCESSED DATA
# ==========================================

# Call the preprocessing function
# It returns:
# X_train = training features
# X_test  = testing features
# y_train = training target
# y_test  = testing target
X_train, X_test, y_train, y_test = preprocess_data()


# ==========================================
# STEP 2: CREATE RANDOM FOREST MODEL
# ==========================================

# Create the Random Forest classifier
#
# n_estimators=100:
# Creates 100 decision trees
#
# random_state=42:
# Makes the experiment reproducible
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================

# Train all the decision trees using
# the training dataset
random_forest.fit(X_train, y_train)


# ==========================================
# STEP 4: MAKE PREDICTIONS
# ==========================================

# Predict the outcomes for the test dataset
#
# 0 = predicted non-diabetic
# 1 = predicted diabetic
rf_pred = random_forest.predict(X_test)


# ==========================================
# STEP 5: CALCULATE EVALUATION METRICS
# ==========================================

# Calculate Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)


# Calculate Precision
rf_precision = precision_score(y_test, rf_pred)


# Calculate Recall
rf_recall = recall_score(y_test, rf_pred)


# Calculate F1-score
rf_f1 = f1_score(y_test, rf_pred)


# ==========================================
# STEP 6: CALCULATE ROC-AUC
# ==========================================

# Get the probability of class 1
#
# Class 1 represents the positive diabetes outcome
rf_probability = random_forest.predict_proba(X_test)[:, 1]


# Calculate ROC-AUC
rf_roc_auc = roc_auc_score(y_test, rf_probability)


# ==========================================
# STEP 7: CONFUSION MATRIX
# ==========================================

# Create the confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)


# ==========================================
# STEP 8: DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("RANDOM FOREST RESULTS")
print("====================================")

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(rf_cm)

print("\n====================================")
print("Random Forest evaluation completed")
print("====================================")