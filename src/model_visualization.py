# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# MODEL COMPARISON VISUALIZATION
# ==========================================


# Import pandas
# Used to read the CSV comparison file
import pandas as pd


# Import matplotlib for creating charts
import matplotlib.pyplot as plt


# ==========================================
# STEP 1: READ MODEL RESULTS
# ==========================================

# Read the model comparison CSV file
# that was created by models.py
results = pd.read_csv(
    "results/model_comparison.csv"
)


# Print the loaded results so we can
# verify that the data was read correctly.
print("\nModel comparison data:")
print(results)


# ==========================================
# STEP 2: CREATE ACCURACY CHART
# ==========================================

# Create a new figure
plt.figure(figsize=(10, 6))


# Create a bar chart
#
# results["Model"] contains model names
# results["Accuracy"] contains accuracy values
plt.bar(
    results["Model"],
    results["Accuracy"]
)


# Add chart title
plt.title("Model Accuracy Comparison")


# Add label to the vertical axis
plt.ylabel("Accuracy")


# Set vertical axis range from 0 to 1
# because our accuracy values are between
# 0 and 1.
plt.ylim(0, 1)


# Rotate model names so they are easier to read
plt.xticks(rotation=20)


# Add grid lines to make comparison easier
plt.grid(axis="y", alpha=0.3)


# Adjust the layout so labels fit properly
plt.tight_layout()


# Save the chart as a PNG image
plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)


# Display the chart
plt.show()


# ==========================================
# STEP 3: CREATE PRECISION CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Precision"]
)

plt.title("Model Precision Comparison")

plt.ylabel("Precision")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/precision_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# STEP 4: CREATE RECALL CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Recall"]
)

plt.title("Model Recall Comparison")

plt.ylabel("Recall")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/recall_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# STEP 5: CREATE F1-SCORE CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["F1 Score"]
)

plt.title("Model F1-Score Comparison")

plt.ylabel("F1 Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/f1_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# STEP 6: CREATE ROC-AUC CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["ROC-AUC"]
)

plt.title("Model ROC-AUC Comparison")

plt.ylabel("ROC-AUC")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/roc_auc_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n====================================")
print("MODEL VISUALIZATION COMPLETED")
print("====================================")