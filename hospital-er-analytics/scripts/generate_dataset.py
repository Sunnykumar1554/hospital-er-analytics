import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────
N = 500
DEPARTMENTS  = ['Cardiology', 'Orthopedics', 'Neurology',
                'General Medicine', 'Pediatrics', 'Emergency',
                'Gynecology', 'ENT']
DIAGNOSES    = ['Chest Pain', 'Fracture', 'Appendicitis',
                'Migraine', 'Fever', 'Asthma', 'Stroke',
                'Heart Attack', 'Kidney Stones', 'Pneumonia',
                'Dengue', 'COVID-19', 'Hypertension']
DOCTORS      = ['Dr. Sharma', 'Dr. Patel', 'Dr. Iyer',
                'Dr. Khan', 'Dr. Verma', 'Dr. Singh',
                'Dr. Reddy', 'Dr. Nair']
BLOOD_TYPES  = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
PAYMENT_MODES= ['Insurance', 'Cash', 'Credit Card', 'Government Scheme']
STATUSES     = ['Admitted', 'Discharged']

# ── Generate ────────────────────────────────────────────────────
rows = []
for i in range(N):
    admit_dt = fake.date_time_between(
        start_date='-1y', end_date='now'
    )
    wait_min  = int(np.random.lognormal(3.5, 0.6))  # realistic skew
    stay_hrs  = random.randint(1, 72)
    discharge_dt = admit_dt + timedelta(hours=stay_hrs)
    dept      = random.choice(DEPARTMENTS)
    rows.append({
        'Patient_ID'       : f'ER{i+1:04d}',
        'Patient_Name'     : fake.name(),
        'Age'              : random.randint(1, 90),
        'Gender'           : random.choice(['Male', 'Female', 'Other']),
        'Blood_Type'       : random.choice(BLOOD_TYPES),
        'Admission_Date'   : admit_dt.strftime('%Y-%m-%d'),
        'Admission_Time'   : admit_dt.strftime('%H:%M'),
        'Discharge_Time'   : discharge_dt.strftime('%Y-%m-%d %H:%M'),
        'Department'       : dept,
        'Doctor_Name'      : random.choice(DOCTORS),
        'Diagnosis'        : random.choice(DIAGNOSES),
        'Wait_Time_Minutes': wait_min,
        'Treatment_Cost'   : round(random.uniform(500, 85000), 2),
        'Payment_Mode'     : random.choice(PAYMENT_MODES),
        'Status'           : random.choices(STATUSES, weights=[40, 60])[0],
    })

df = pd.DataFrame(rows)

# ── Inject ~5% realistic missing values ─────────────────────────
for col in ['Blood_Type', 'Discharge_Time', 'Payment_Mode']:
    mask = np.random.random(N) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv('hospital_er_data.csv', index=False)
print(f"✓ Generated {N} rows → hospital_er_data.csv")