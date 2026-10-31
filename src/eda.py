import pandas as pd

# Load the dataset
df = pd.read_csv("data/diabetes.csv")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nStatistical summary:")
print(df.describe())

# Zero values
print("\nZero values:")
print((df == 0).sum())

# Target distribution
print("\nOutcome distribution:")
print(df["Outcome"].value_counts())