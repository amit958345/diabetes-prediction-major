# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# SUPPORT VECTOR MACHINE CLASSIFIER
# ==========================================


# Import our reusable preprocessing function
from preprocessing import preprocess_data


# Import Support Vector Machine classifier
from sklearn.svm import SVC


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
#
# It returns:
# X_train = training features
# X_test  = testing features
# y_train = training target
# y_test  = testing target
X_train, X_test, y_train, y_test = preprocess_data()


# ==========================================
# STEP 2: CREATE SVM MODEL
# ==========================================

# Create the Support Vector Machine classifier
#
# kernel="rbf":
# Allows the model to learn nonlinear
# decision boundaries
#
# C=1.0:
# Controls the penalty for classification errors
#
# probability=True:
# Allows the model to calculate class probabilities
# which we need for ROC-AUC
#
# random_state=42:
# Makes the experiment reproducible where applicable
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    random_state=42
)


# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================

# Train the SVM using the training dataset
svm_model.fit(X_train, y_train)


# ==========================================
# STEP 4: MAKE PREDICTIONS
# ==========================================

# Predict outcomes for the test dataset
#
# 0 = predicted non-diabetic
# 1 = predicted diabetic
svm_pred = svm_model.predict(X_test)


# ==========================================
# STEP 5: CALCULATE EVALUATION METRICS
# ==========================================

# Calculate Accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)


# Calculate Precision
svm_precision = precision_score(y_test, svm_pred)


# Calculate Recall
svm_recall = recall_score(y_test, svm_pred)


# Calculate F1-score
svm_f1 = f1_score(y_test, svm_pred)


# ==========================================
# STEP 6: CALCULATE ROC-AUC
# ==========================================

# Get probability of class 1
#
# Class 1 represents the positive diabetes outcome
svm_probability = svm_model.predict_proba(X_test)[:, 1]


# Calculate ROC-AUC
svm_roc_auc = roc_auc_score(y_test, svm_probability)


# ==========================================
# STEP 7: CONFUSION MATRIX
# ==========================================

# Create the confusion matrix
svm_cm = confusion_matrix(y_test, svm_pred)


# ==========================================
# STEP 8: DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("SVM RESULTS")
print("====================================")

print(f"Accuracy  : {svm_accuracy:.4f}")
print(f"Precision : {svm_precision:.4f}")
print(f"Recall    : {svm_recall:.4f}")
print(f"F1 Score  : {svm_f1:.4f}")
print(f"ROC-AUC   : {svm_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(svm_cm)

print("\n====================================")
print("SVM evaluation completed")
print("====================================")