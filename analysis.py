import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/employees.csv")


# -----------------------------
# Basic Dataset Information
# -----------------------------

print("Total Employees:", len(df))
print("\nEmployees by Department:")
print(df["Department"].value_counts())


# -----------------------------
# Salary Analysis
# -----------------------------

print("\nAverage Salary:", df["Salary"].mean())

print("\nAverage Salary by Department:")
print(df.groupby("Department")["Salary"].mean())

avg_salary_by_department = df.groupby("Department")["Salary"].mean()

print("\nDepartment with Highest Average Salary:")
print(avg_salary_by_department.idxmax())

print("\nAverage Salary by Job Role:")
print(df.groupby("Job_Role")["Salary"].mean())


# -----------------------------
# Department Distribution
# -----------------------------

print("\nDepartment Percentage:")
print(df["Department"].value_counts(normalize=True) * 100)


# -----------------------------
# Visualizations
# -----------------------------

# 1. Employees by Department
sb.countplot(data=df, x="Department")
plt.title("Employees by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.show()


# 2. Salary Distribution
sb.histplot(data=df, x="Salary")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.show()


# 3. Salary Outliers
sb.boxplot(data=df, y="Salary")
plt.title("Salary Outliers")
plt.ylabel("Salary")
plt.show()


# 4. Experience vs Salary
sb.scatterplot(data=df, x="Experience_Years", y="Salary")
plt.title("Experience vs Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.show()


# 5. Correlation Heatmap
correlation = df.corr(numeric_only=True)

sb.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()