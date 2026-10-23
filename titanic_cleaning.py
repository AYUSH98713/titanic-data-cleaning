import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# =========================================================
# ASSIGNMENT 1: Load dataset and summarize basic stats
# =========================================================
df = pd.read_csv("titanic.csv")
print("Shape:", df.shape)
print(df.head(), "\n")

print("---- df.info() ----")
df.info()

print("\n---- df.describe() (numeric) ----")
print(df.describe())

print("\n---- df.describe(include='object') (categorical) ----")
print(df.describe(include="object"))

# =========================================================
# ASSIGNMENT 2: Handle missing data (mean / median imputation)
# =========================================================
print("\n---- Missing values BEFORE ----")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Age (numeric, skewed) -> median imputation
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)

# Embarked (categorical, only 2 missing) -> mode imputation
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# Cabin: ~77% missing -> too sparse to impute, drop the column
df = df.drop(columns=["Cabin"])

# (Mean imputation demo, for the assignment: how you'd do it on Fare)
# df["Fare"] = df["Fare"].fillna(df["Fare"].mean())   # Fare has no NaNs here

print(f"\nAge filled with median = {age_median}")
print(f"Embarked filled with mode = {embarked_mode}")
print("\n---- Missing values AFTER ----")
print(df.isnull().sum().sum(), "missing values remain")

# =========================================================
# ASSIGNMENT 3: Encode categorical variables
# =========================================================
# LabelEncoder -> Sex (binary: 0/1)
le = LabelEncoder()
df["Sex_encoded"] = le.fit_transform(df["Sex"])
print("\nLabelEncoder mapping for Sex:", dict(zip(le.classes_, le.transform(le.classes_))))

# OneHotEncoder -> Embarked (nominal: C, Q, S, no natural order)
ohe = OneHotEncoder(sparse_output=False, dtype=int)
emb = ohe.fit_transform(df[["Embarked"]])
emb_cols = [f"Embarked_{c}" for c in ohe.categories_[0]]
emb_df = pd.DataFrame(emb, columns=emb_cols, index=df.index)
df = pd.concat([df, emb_df], axis=1)
print("OneHotEncoder columns for Embarked:", emb_cols)

# =========================================================
# MINI PROJECT: Visualize age distribution
# =========================================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(df["Age"], bins=30, kde=True, color="steelblue", ax=axes[0])
axes[0].axvline(age_median, color="red", linestyle="--", label=f"Median = {age_median:.0f}")
axes[0].set_title("Age Distribution (after imputation)")
axes[0].set_xlabel("Age"); axes[0].legend()

sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
             palette={0: "tomato", 1: "seagreen"}, ax=axes[1])
axes[1].set_title("Age Distribution by Survival (0 = No, 1 = Yes)")

plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
print("\nSaved age_distribution.png")

# =========================================================
# MINI PROJECT: Output cleaned dataset
# =========================================================
df.to_csv("titanic_cleaned.csv", index=False)
print("Saved titanic_cleaned.csv with shape", df.shape)
print(df.head())
