import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Create results folder if needed
import os
os.makedirs("results", exist_ok=True)


# 1. Outcome Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("results/outcome_distribution.png")
plt.show()


# 2. Zero Value Analysis
features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

zero_counts = [(df[feature] == 0).sum() for feature in features]

plt.figure(figsize=(8, 5))
plt.bar(features, zero_counts)
plt.title("Zero Values in Selected Medical Features")
plt.xlabel("Feature")
plt.ylabel("Number of Zero Values")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/zero_value_analysis.png")
plt.show()


# 3. Feature Distributions
df[features].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Selected Medical Features")
plt.tight_layout()
plt.savefig("results/feature_distributions.png")
plt.show()


# 4. Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.show()


# 5. Boxplots
plt.figure(figsize=(12, 7))
sns.boxplot(data=df[features])
plt.title("Boxplot of Selected Medical Features")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/boxplots.png")
plt.show()

print("\nEDA visualizations created successfully!")
print("Check the 'results' folder.")