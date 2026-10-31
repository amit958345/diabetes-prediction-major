# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# CONFUSION MATRIX VISUALIZATION
# ==========================================


# Import pandas
# Used for working with tabular data if needed
import pandas as pd


# Import matplotlib
# Used for creating and saving figures
import matplotlib.pyplot as plt


# Import seaborn
# Used to create the confusion matrix heatmaps
import seaborn as sns


# Import our reusable preprocessing function
from preprocessing import preprocess_data


# Import the five machine-learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Import confusion_matrix function
from sklearn.metrics import confusion_matrix


# ==========================================
# STEP 1: LOAD PREPROCESSED DATA
# ==========================================

# Run our existing preprocessing pipeline.
#
# This gives us the same training and testing
# data used in our model comparison.
X_train, X_test, y_train, y_test = preprocess_data()


# ==========================================
# STEP 2: CREATE ALL FIVE MODELS
# ==========================================

# Store all five models in a dictionary.
models = {

    "Logistic Regression": LogisticRegression(
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )
}


# ==========================================
# STEP 3: TRAIN AND CREATE MATRICES
# ==========================================

# Loop through every model.
for model_name, model in models.items():

    # Display the current model name.
    print("\nCreating confusion matrix for:", model_name)


    # Train the current model.
    model.fit(X_train, y_train)


    # Predict the test dataset.
    predictions = model.predict(X_test)


    # Calculate the confusion matrix.
    cm = confusion_matrix(
        y_test,
        predictions
    )


    # ==========================================
    # STEP 4: CREATE HEATMAP
    # ==========================================

    # Create a new figure.
    plt.figure(figsize=(7, 5))


    # Create the confusion matrix heatmap.
    #
    # annot=True:
    # Displays the numerical values inside
    # the cells.
    #
    # fmt="d":
    # Displays values as integers.
    #
    # xticklabels:
    # Names the predicted classes.
    #
    # yticklabels:
    # Names the actual classes.
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        yticklabels=["Non-Diabetic (0)", "Diabetic (1)"]
    )


    # Add chart title.
    plt.title(
        f"{model_name} - Confusion Matrix"
    )


    # Label the horizontal axis.
    plt.xlabel("Predicted Class")


    # Label the vertical axis.
    plt.ylabel("Actual Class")


    # Adjust spacing.
    plt.tight_layout()


    # Create a filename based on the model name.
    filename = model_name.lower().replace(
        " ",
        "_"
    )


    # Save the heatmap.
    plt.savefig(
        f"results/confusion_matrix_{filename}.png",
        dpi=300
    )


    # Display the heatmap.
    plt.show()


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n====================================")
print("CONFUSION MATRIX VISUALIZATION")
print("COMPLETED")
print("====================================")