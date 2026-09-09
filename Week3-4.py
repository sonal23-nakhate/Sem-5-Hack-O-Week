"""
Hack-o-Week: Week 3 & 4 Plan
Topics Covered:
1. Python Essentials (OOP, Comprehensions)
2. NumPy (Arrays, Broadcasting, Vectorization)
3. Pandas (DataFrames, Cleaning, Merging, GroupBy)
4. Data Visualization (Matplotlib, Seaborn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# STEP 1: PYTHON ESSENTIALS & OOP
# ==========================================
print("--- Step 1: Python Essentials & OOP ---")

class DatasetGenerator:
    """Class to generate synthetic employee data with missing values for practice."""
    def __init__(self, num_records=100, seed=42):
        np.random.seed(seed)
        self.num_records = num_records

    def generate_data(self):
        departments = ['Engineering', 'Data Science', 'Product', 'HR', 'Marketing']
        
        # List comprehensions and random data generation
        data = {
            'Emp_ID': [f'EMP-{1000 + i}' for i in range(self.num_records)],
            'Department': [departments[i] for i in np.random.randint(0, len(departments), self.num_records)],
            'Base_Salary': np.random.normal(loc=75000, scale=15000, size=self.num_records).round(2),
            'Bonus_Pct': np.random.uniform(0.05, 0.25, size=self.num_records).round(2),
            'Years_Experience': np.random.randint(1, 15, size=self.num_records)
        }
        
        # Create DataFrame and introduce missing values (NaNs)
        df = pd.DataFrame(data)
        df.loc[::15, 'Base_Salary'] = np.nan
        df.loc[::20, 'Department'] = np.nan
        return df

# Instantiating OOP Class
generator = DatasetGenerator(num_records=100)
raw_df = generator.generate_data()

print("Raw Dataset Sample:")
print(raw_df.head(), "\n")


# ==========================================
# STEP 2: NUMPY (VECTORIZATION & BROADCASTING)
# ==========================================
print("--- Step 2: NumPy Vectorized Operations ---")

# Convert Pandas series to NumPy arrays
base_salaries = raw_df['Base_Salary'].to_numpy()
bonus_pcts = raw_df['Bonus_Pct'].to_numpy()

# 1. Vectorized Calculation (handles NaNs gracefully)
total_comp = np.where(
    np.isnan(base_salaries), 
    np.nan, 
    base_salaries * (1 + bonus_pcts)
)

# 2. Broadcasting (Applying a global 5% adjustment matrix/scalar)
bonus_adjustment_factor = np.array([1.05])
adjusted_comp = total_comp * bonus_adjustment_factor

# Assigning vectorized results back to DataFrame
raw_df['Total_Comp'] = total_comp.round(2)

print("Vectorized Outputs:")
print(raw_df[['Emp_ID', 'Base_Salary', 'Bonus_Pct', 'Total_Comp']].head(), "\n")


# ==========================================
# STEP 3: PANDAS (CLEANING, MERGING & GROUPBY)
# ==========================================
print("--- Step 3: Pandas Cleaning & Aggregations ---")

cleaned_df = raw_df.copy()

# 1. Data Cleaning
cleaned_df['Department'] = cleaned_df['Department'].fillna('Unassigned')

# Impute missing salary using department-wise median
cleaned_df['Base_Salary'] = cleaned_df.groupby('Department')['Base_Salary'].transform(
    lambda x: x.fillna(x.median())
)

# Recalculate Total_Comp after filling NaNs
cleaned_df['Total_Comp'] = (cleaned_df['Base_Salary'] * (1 + cleaned_df['Bonus_Pct'])).round(2)

# 2. Merging Datasets
project_info = pd.DataFrame({
    'Department': ['Engineering', 'Data Science', 'Product', 'HR', 'Marketing', 'Unassigned'],
    'Active_Projects': [12, 8, 5, 3, 6, 0]
})

merged_df = pd.merge(cleaned_df, project_info, on='Department', how='left')

# 3. GroupBy Summary
dept_summary = merged_df.groupby('Department').agg(
    Avg_Total_Comp=('Total_Comp', 'mean'),
    Employee_Count=('Emp_ID', 'count'),
    Avg_Experience=('Years_Experience', 'mean'),
    Active_Projects=('Active_Projects', 'first')
).reset_index()

print("Department Summary:")
print(dept_summary, "\n")


# ==========================================
# STEP 4: DATA VISUALIZATION (SEABORN & MATPLOTLIB)
# ==========================================
print("--- Step 4: Generating Plots ---")

sns.set_theme(style="whitegrid")

# Create figure canvas with 2 side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Average Compensation per Department
sns.barplot(
    data=dept_summary,
    x='Department',
    y='Avg_Total_Comp',
    ax=axes[0],
    palette='crest'
)
axes[0].set_title('Average Compensation by Department', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Comp ($)')
axes[0].tick_params(axis='x', rotation=30)

# Plot 2: Experience vs. Total Compensation
sns.regplot(
    data=merged_df,
    x='Years_Experience',
    y='Total_Comp',
    ax=axes[1],
    color='steelblue',
    scatter_kws={'alpha': 0.7}
)
axes[1].set_title('Years of Experience vs Total Compensation', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Total Comp ($)')

plt.tight_layout()
plt.show()

print("Pipeline executed successfully!")