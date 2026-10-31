# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# K-NEAREST NEIGHBORS CLASSIFIER
# ==========================================


# Import our reusable preprocessing function
from preprocessing import preprocess_data


# Import K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier


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
# STEP 2: CREATE KNN MODEL
# ==========================================

# Create the KNN classifier
#
# n_neighbors=5:
# The model considers the 5 nearest neighbors
#
# metric="euclidean":
# Uses Euclidean distance to measure
# the distance between observations
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)


# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================

# Store the training data in the KNN model
#
# KNN does not build a traditional mathematical
# model like Logistic Regression.
knn.fit(X_train, y_train)


# ==========================================
# STEP 4: MAKE PREDICTIONS
# ==========================================

# Predict outcomes for the test dataset
#
# 0 = predicted non-diabetic
# 1 = predicted diabetic
knn_pred = knn.predict(X_test)


# ==========================================
# STEP 5: CALCULATE EVALUATION METRICS
# ==========================================

# Calculate Accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)


# Calculate Precision
knn_precision = precision_score(y_test, knn_pred)


# Calculate Recall
knn_recall = recall_score(y_test, knn_pred)


# Calculate F1-score
knn_f1 = f1_score(y_test, knn_pred)


# ==========================================
# STEP 6: CALCULATE ROC-AUC
# ==========================================

# Get probability of class 1
#
# Class 1 represents the positive diabetes outcome
knn_probability = knn.predict_proba(X_test)[:, 1]


# Calculate ROC-AUC
knn_roc_auc = roc_auc_score(y_test, knn_probability)


# ==========================================
# STEP 7: CONFUSION MATRIX
# ==========================================

# Create the confusion matrix
knn_cm = confusion_matrix(y_test, knn_pred)


# ==========================================
# STEP 8: DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("KNN RESULTS")
print("====================================")

print(f"Accuracy  : {knn_accuracy:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall    : {knn_recall:.4f}")
print(f"F1 Score  : {knn_f1:.4f}")
print(f"ROC-AUC   : {knn_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(knn_cm)

print("\n====================================")
print("KNN evaluation completed")
print("====================================")