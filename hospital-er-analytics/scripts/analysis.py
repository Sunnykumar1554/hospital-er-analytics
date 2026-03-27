import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ────────────────────────────────────────────────────────────────
# 1. LOAD & BASIC INSPECTION
# ────────────────────────────────────────────────────────────────
df = pd.read_csv('hospital_er_data.csv')
print(f"Shape: {df.shape}")
print(df.info())
print(df.isnull().sum())

# ────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ────────────────────────────────────────────────────────────────
# Parse dates
df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
df['Discharge_Time'] = pd.to_datetime(df['Discharge_Time'], errors='coerce')

# Fill missing values
df['Blood_Type'].fillna('Unknown', inplace=True)
df['Payment_Mode'].fillna('Not Specified', inplace=True)

# Remove duplicates
df.drop_duplicates(subset=['Patient_ID'], inplace=True)

# Drop invalid rows
df = df[(df['Age'] > 0) & (df['Age'] <= 120)]
df = df[df['Treatment_Cost'] > 0]
df = df[df['Wait_Time_Minutes'] > 0]

# ────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────
df['Month']             = df['Admission_Date'].dt.month_name()
df['Day_of_Week']      = df['Admission_Date'].dt.day_name()
df['Hour']             = pd.to_datetime(df['Admission_Time'],
                              format='%H:%M').dt.hour

df['Stay_Hours'] = (
    (df['Discharge_Time'] -
     pd.to_datetime(df['Admission_Date'].astype(str) + ' ' + df['Admission_Time']))
    .dt.total_seconds() / 3600
).clip(lower=0)

df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 12, 17, 40, 60, 120],
    labels=['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior']
)

df['Wait_Category'] = pd.cut(
    df['Wait_Time_Minutes'],
    bins=[0, 30, 60, 120, float('inf')],
    labels=['Fast', 'Normal', 'Slow', 'Critical']
)

# ────────────────────────────────────────────────────────────────
# 4. PEAK HOURS & PATIENT FLOW
# ────────────────────────────────────────────────────────────────
peak_hours = (df.groupby('Hour')
               .agg(Patients=('Patient_ID', 'count'),
                    Avg_Wait=('Wait_Time_Minutes', 'mean'))
               .round(2)
               .reset_index())

print("\n── Peak Hours ──")
print(peak_hours.nlargest(5, 'Patients'))

# ────────────────────────────────────────────────────────────────
# 5. DEPARTMENT BOTTLENECK SCORE
# ────────────────────────────────────────────────────────────────
dept_score = (df.groupby('Department')
               .agg(Volume=('Patient_ID', 'count'),
                    Avg_Wait=('Wait_Time_Minutes', 'mean'))
               .reset_index())
dept_score['Bottleneck_Score'] = (dept_score['Volume'] *
                                    dept_score['Avg_Wait'])
dept_score['Bottleneck_Score'] = (
    (dept_score['Bottleneck_Score'] -
     dept_score['Bottleneck_Score'].min()) /
    (dept_score['Bottleneck_Score'].max() -
     dept_score['Bottleneck_Score'].min())
).round(4)  # normalised 0-1

print("\n── Bottleneck Scores ──")
print(dept_score.sort_values('Bottleneck_Score', ascending=False))

# ────────────────────────────────────────────────────────────────
# 6. OUTLIER DETECTION — IQR + Z-SCORE
# ────────────────────────────────────────────────────────────────
Q1  = df['Wait_Time_Minutes'].quantile(0.25)
Q3  = df['Wait_Time_Minutes'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Wait_Outlier_IQR'] = (
    (df['Wait_Time_Minutes'] < lower_bound) |
    (df['Wait_Time_Minutes'] > upper_bound)
)
df['Wait_ZScore'] = np.abs(stats.zscore(df['Wait_Time_Minutes']))
df['Wait_Outlier_Z'] = df['Wait_ZScore'] > 3

outliers_df = df[df['Wait_Outlier_IQR'] | df['Wait_Outlier_Z']]
print(f"\n── Outliers found: {len(outliers_df)} rows ──")

# ────────────────────────────────────────────────────────────────
# 7. VISUALISATION (optional but impressive for README)
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Hospital ER Analysis', fontsize=16, fontweight='bold')

# A) Hourly patient count
axes[0,0].bar(peak_hours['Hour'], peak_hours['Patients'], color='#3b82f6')
axes[0,0].set_title('Patient Volume by Hour')
axes[0,0].set_xlabel('Hour of Day')

# B) Avg wait time by department
dept_wait = df.groupby('Department')['Wait_Time_Minutes'].mean().sort_values()
dept_wait.plot(kind='barh', ax=axes[0,1], color='#00c8a0')
axes[0,1].set_title('Avg Wait Time by Department (min)')

# C) Wait time distribution with outliers highlighted
axes[1,0].hist(df['Wait_Time_Minutes'], bins=40, color='#6366f1', alpha=0.7)
axes[1,0].axvline(upper_bound, color='#ef4444', ls='--', label=f'IQR upper={upper_bound:.0f}')
axes[1,0].set_title('Wait Time Distribution'); axes[1,0].legend()

# D) Monthly trend
monthly = df.groupby(df['Admission_Date'].dt.to_period('M')).size()
monthly.plot(ax=axes[1,1], color='#f59e0b', marker='o')
axes[1,1].set_title('Monthly Admissions Trend')

plt.tight_layout()
plt.savefig('er_analysis_charts.png', dpi=150)

# ────────────────────────────────────────────────────────────────
# 8. EXPORT FINAL CLEAN CSV FOR POWER BI
# ────────────────────────────────────────────────────────────────
export_cols = [
    'Patient_ID', 'Age', 'Gender', 'Blood_Type',
    'Admission_Date', 'Hour', 'Month', 'Day_of_Week',
    'Department', 'Doctor_Name', 'Diagnosis',
    'Wait_Time_Minutes', 'Wait_Category',
    'Treatment_Cost', 'Payment_Mode', 'Status',
    'Age_Group', 'Stay_Hours', 'Wait_Outlier_IQR'
]
df[export_cols].to_csv('hospital_er_powerbi.csv', index=False)
print("\n✓ Exported hospital_er_powerbi.csv for Power BI")
print(f"  Final shape: {df[export_cols].shape}")
