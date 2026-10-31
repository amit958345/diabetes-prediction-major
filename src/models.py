# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# FINAL MODEL COMPARISON
# ==========================================


# Import pandas
# We will use pandas to create the final comparison table
import pandas as pd


# Import our reusable preprocessing function
from preprocessing import preprocess_data


# Import Logistic Regression
from sklearn.linear_model import LogisticRegression


# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier


# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# Import K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier


# Import Support Vector Machine Classifier
from sklearn.svm import SVC


# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
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
# This performs the same preprocessing pipeline
# that we used for our individual models.
X_train, X_test, y_train, y_test = preprocess_data()


# ==========================================
# STEP 2: CREATE ALL FIVE MODELS
# ==========================================

# Create a dictionary containing all five models.
#
# The key is the model name.
# The value is the actual machine-learning model.

models = {

    # Logistic Regression model
    "Logistic Regression": LogisticRegression(
        random_state=42
    ),

    # Decision Tree model
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    # Random Forest model
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    # KNN model
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    ),

    # SVM model
    #
    # probability is not enabled here.
    # We will use decision_function() for ROC-AUC.
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )
}


# ==========================================
# STEP 3: CREATE A LIST FOR RESULTS
# ==========================================

# This empty list will store the results
# of every machine-learning model.
results = []


# ==========================================
# STEP 4: TRAIN AND EVALUATE EACH MODEL
# ==========================================

# Loop through every model in our dictionary.
#
# model_name = name of the model
# model      = actual machine-learning model
for model_name, model in models.items():

    # Display the name of the model currently
    # being trained.
    print("\nTraining:", model_name)

    # Train the current model using the
    # training dataset.
    model.fit(X_train, y_train)

    # Use the trained model to predict the
    # outcomes of the test dataset.
    predictions = model.predict(X_test)


    # ==========================================
    # CALCULATE BASIC CLASSIFICATION METRICS
    # ==========================================

    # Calculate Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Calculate Precision
    precision = precision_score(
        y_test,
        predictions
    )

    # Calculate Recall
    recall = recall_score(
        y_test,
        predictions
    )

    # Calculate F1-score
    f1 = f1_score(
        y_test,
        predictions
    )


    # ==========================================
    # CALCULATE ROC-AUC
    # ==========================================

    # SVM is handled differently because
    # probability=True is not being used.
    if model_name == "SVM":

        # Get the decision scores produced by SVM.
        # These scores indicate how strongly the
        # observations are associated with a class.
        scores = model.decision_function(X_test)

    else:

        # For the other models, obtain the probability
        # of the positive class (class 1).
        scores = model.predict_proba(X_test)[:, 1]

    # Calculate ROC-AUC using the scores.
    roc_auc = roc_auc_score(
        y_test,
        scores
    )


    # ==========================================
    # STORE THE RESULTS
    # ==========================================

    # Add the current model's results to the
    # results list.
    results.append({

        # Store model name
        "Model": model_name,

        # Store Accuracy
        "Accuracy": accuracy,

        # Store Precision
        "Precision": precision,

        # Store Recall
        "Recall": recall,

        # Store F1-score
        "F1 Score": f1,

        # Store ROC-AUC
        "ROC-AUC": roc_auc
    })


# ==========================================
# STEP 5: CREATE COMPARISON TABLE
# ==========================================

# Convert the results list into a pandas DataFrame.
# A DataFrame is a table-like data structure.
comparison_df = pd.DataFrame(results)

# Save the model comparison results as a CSV file
# This file can later be used for visualization
# and report preparation.
comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ==========================================
# STEP 6: DISPLAY COMPARISON TABLE
# ==========================================

print("\n")
print("============================================================")
print("FINAL MODEL COMPARISON")
print("============================================================")


# Print the comparison table.
#
# round(4) displays the metric values
# up to four decimal places.
print(
    comparison_df.round(4).to_string(index=False)
)


print("\n============================================================")
print("MODEL COMPARISON COMPLETED")
print("============================================================")