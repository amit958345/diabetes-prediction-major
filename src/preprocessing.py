# ==========================================
# DIABETES PREDICTION - DATA PREPROCESSING
# ==========================================

# ------------------------------------------
# 1. Import required libraries
# ------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ------------------------------------------
# 2. Create preprocessing function
# ------------------------------------------
# This function performs all preprocessing
# steps and returns the processed data.
# ------------------------------------------

def preprocess_data():

    # --------------------------------------
    # 3. Load the diabetes dataset
    # --------------------------------------

    df = pd.read_csv("data/diabetes.csv")


    # --------------------------------------
    # 4. Separate input features and target
    # --------------------------------------
    # X = input features
    # y = target variable (Outcome)
    #
    # Outcome:
    # 0 = No diabetes
    # 1 = Diabetes
    # --------------------------------------

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]


    # --------------------------------------
    # 5. Split data into training and testing
    # --------------------------------------
    # 80% → training data
    # 20% → testing data
    #
    # random_state=42 makes the split
    # reproducible.
    #
    # stratify=y keeps the class distribution
    # approximately similar in train and test.
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # --------------------------------------
    # 6. Display dataset sizes
    # --------------------------------------

    print("Original dataset:", X.shape)
    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())


    # --------------------------------------
    # 7. Select columns where zero may
    #    represent a missing measurement
    # --------------------------------------
    #
    # We do NOT include Pregnancies because
    # 0 pregnancies is a valid value.
    #
    # We do NOT include Outcome because it
    # is the target variable.
    # --------------------------------------

    columns_with_invalid_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    # --------------------------------------
    # 8. Create copies of training and
    #    testing data
    # --------------------------------------
    # We use copies so that the original
    # X_train and X_test remain unchanged.
    # --------------------------------------

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()


    # --------------------------------------
    # 9. Convert selected zero values
    #    into NaN (missing values)
    # --------------------------------------

    X_train_clean[columns_with_invalid_zero] = (
        X_train_clean[columns_with_invalid_zero].replace(0, np.nan)
    )

    X_test_clean[columns_with_invalid_zero] = (
        X_test_clean[columns_with_invalid_zero].replace(0, np.nan)
    )


    # --------------------------------------
    # 10. Check missing values BEFORE
    #     median imputation
    # --------------------------------------

    print("\nMissing values before imputation:")
    print(X_train_clean.isnull().sum())


    # --------------------------------------
    # 11. Create the median imputer
    # --------------------------------------
    # SimpleImputer will replace missing
    # numerical values with the median
    # of their respective feature.
    # --------------------------------------

    imputer = SimpleImputer(strategy="median")


    # --------------------------------------
    # 12. Learn median values from
    #     TRAINING data only
    # --------------------------------------
    # This prevents data leakage.
    # --------------------------------------

    imputer.fit(X_train_clean)


    # --------------------------------------
    # 13. Apply median imputation
    # --------------------------------------

    X_train_imputed = imputer.transform(X_train_clean)
    X_test_imputed = imputer.transform(X_test_clean)


    # --------------------------------------
    # 14. Convert imputed arrays back
    #     into pandas DataFrames
    # --------------------------------------

    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_imputed = pd.DataFrame(
        X_test_imputed,
        columns=X_test.columns,
        index=X_test.index
    )


    # --------------------------------------
    # 15. Check missing values AFTER
    #     median imputation
    # --------------------------------------

    print("\nMissing values after imputation:")
    print(X_train_imputed.isnull().sum())


    # --------------------------------------
    # 16. Display sample of training data
    #     after median imputation
    # --------------------------------------

    print("\nTraining data after imputation:")
    print(X_train_imputed.head())


    # --------------------------------------
    # 17. Create copies for IQR clipping
    # --------------------------------------

    X_train_clipped = X_train_imputed.copy()
    X_test_clipped = X_test_imputed.copy()


    # --------------------------------------
    # 18. Detect statistical outliers
    #     using the IQR method
    # --------------------------------------
    #
    # IQR = Q3 - Q1
    #
    # Lower Bound = Q1 - 1.5 × IQR
    # Upper Bound = Q3 + 1.5 × IQR
    # --------------------------------------

    print("\nIQR-based outlier counts:")

    for column in X_train_imputed.columns:

        # Calculate first quartile
        Q1 = X_train_imputed[column].quantile(0.25)

        # Calculate third quartile
        Q3 = X_train_imputed[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Calculate lower boundary
        lower_bound = Q1 - 1.5 * IQR

        # Calculate upper boundary
        upper_bound = Q3 + 1.5 * IQR

        # Find values outside the boundaries
        outliers = X_train_imputed[
            (X_train_imputed[column] < lower_bound) |
            (X_train_imputed[column] > upper_bound)
        ]

        # Display number of statistical outliers
        print(column, ":", len(outliers))


    # --------------------------------------
    # 19. Apply IQR clipping
    # --------------------------------------
    # Instead of deleting outlier rows,
    # extreme values are capped at the
    # calculated IQR boundaries.
    #
    # Boundaries are calculated using
    # TRAINING data only.
    # --------------------------------------

    for column in X_train_clipped.columns:

        # Calculate Q1 and Q3
        Q1 = X_train_clipped[column].quantile(0.25)
        Q3 = X_train_clipped[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Calculate boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Clip training values
        X_train_clipped[column] = X_train_clipped[column].clip(
            lower_bound,
            upper_bound
        )

        # Apply the SAME boundaries to test data
        X_test_clipped[column] = X_test_clipped[column].clip(
            lower_bound,
            upper_bound
        )


    # --------------------------------------
    # 20. Display training data after
    #     IQR clipping
    # --------------------------------------

    print("\nTraining data after IQR clipping:")
    print(X_train_clipped.head())


    # --------------------------------------
    # 21. Create StandardScaler
    # --------------------------------------

    scaler = StandardScaler()


    # --------------------------------------
    # 22. Learn scaling parameters from
    #     TRAINING data only
    # --------------------------------------

    scaler.fit(X_train_clipped)


    # --------------------------------------
    # 23. Standardize training and
    #     testing data
    # --------------------------------------

    X_train_scaled = scaler.transform(X_train_clipped)
    X_test_scaled = scaler.transform(X_test_clipped)


    # --------------------------------------
    # 24. Convert scaled arrays back
    #     into DataFrames
    # --------------------------------------

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )


    # --------------------------------------
    # 25. Display training data after
    #     Standard Scaling
    # --------------------------------------

    print("\nTraining data after Standard Scaling:")
    print(X_train_scaled.head())


    # --------------------------------------
    # 26. Return processed data
    # --------------------------------------
    # These four objects will be used later
    # by our machine learning models.
    # --------------------------------------

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


# ==========================================
# 27. Test the preprocessing function
# ==========================================
# This section runs only when we directly
# execute preprocessing.py.
# ==========================================

if __name__ == "__main__":

    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()


    # --------------------------------------
    # 28. Final verification
    # --------------------------------------

    print("\n====================================")
    print("Preprocessing completed successfully")
    print("====================================")

    print("\nTraining features:", X_train_scaled.shape)
    print("Testing features:", X_test_scaled.shape)

    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)