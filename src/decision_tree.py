# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# DECISION TREE CLASSIFIER
# ==========================================


# Import our reusable preprocessing function
# This function performs:
# train-test split
# invalid zero handling
# median imputation
# IQR-based outlier clipping
# standard scaling
from preprocessing import preprocess_data


# Import Decision Tree Classifier from scikit-learn
from sklearn.tree import DecisionTreeClassifier


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

# Call our preprocessing function
# It returns training features, testing features,
# training target and testing target
X_train, X_test, y_train, y_test = preprocess_data()


# ==========================================
# STEP 2: CREATE DECISION TREE MODEL
# ==========================================

# Create the Decision Tree classifier
#
# random_state=42:
# Makes the experiment reproducible
#
# max_depth=5:
# Limits the maximum depth of the tree
# to help control overfitting
decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)


# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================

# Train the Decision Tree using training data
#
# X_train = input features
# y_train = actual target values
decision_tree.fit(X_train, y_train)


# ==========================================
# STEP 4: MAKE PREDICTIONS
# ==========================================

# Predict diabetes outcomes for the test dataset
#
# The result will contain:
# 0 = predicted non-diabetic
# 1 = predicted diabetic
dt_pred = decision_tree.predict(X_test)


# ==========================================
# STEP 5: CALCULATE EVALUATION METRICS
# ==========================================

# Calculate Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)


# Calculate Precision
dt_precision = precision_score(y_test, dt_pred)


# Calculate Recall
dt_recall = recall_score(y_test, dt_pred)


# Calculate F1-score
dt_f1 = f1_score(y_test, dt_pred)


# ==========================================
# STEP 6: CALCULATE ROC-AUC
# ==========================================

# Get probability of class 1
#
# Class 1 represents the positive diabetes outcome
dt_probability = decision_tree.predict_proba(X_test)[:, 1]


# Calculate ROC-AUC score
dt_roc_auc = roc_auc_score(y_test, dt_probability)


# ==========================================
# STEP 7: CONFUSION MATRIX
# ==========================================

# Create confusion matrix
dt_cm = confusion_matrix(y_test, dt_pred)


# ==========================================
# STEP 8: DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("DECISION TREE RESULTS")
print("====================================")

print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1 Score  : {dt_f1:.4f}")
print(f"ROC-AUC   : {dt_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(dt_cm)

print("\n====================================")
print("Decision Tree evaluation completed")
print("====================================")