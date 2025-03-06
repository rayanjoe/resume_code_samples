

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# Define Chicago suburbs (locations)
suburbs = [
    "Chicago", "Evanston", "Oak Park", "Schaumburg", "Skokie",
    "Arlington Heights", "Des Plaines", "Oak Brook", "Naperville",
    "Cicero", "Berwyn", "Wheaton", "Elmhurst", "Lombard", "Downers Grove"
]

probabilities = np.array([0.4, 0.1, 0.05, 0.03, 0.02, 0.01, 0.07, 0.04, 0.059, 0.023, 0.05, 0.05, 0.042, 0.033, 0.05])
probabilities = probabilities / probabilities.sum()

diseases_by_specialization = {
    "Cardiology": [
        "Hypertension", "Coronary Artery Disease", "Heart Failure", "Arrhythmia",
        "Myocardial Infarction", "Valvular Heart Disease", "Cardiomyopathy", "Pericarditis"
    ],
    "Orthopedics": [
        "Osteoarthritis", "Rheumatoid Arthritis", "Osteoporosis", "Fractures",
        "Spinal Stenosis", "Scoliosis", "Tendinitis", "Rotator Cuff Injury"
    ],
    "Neurology": [
        "Migraine", "Epilepsy", "Stroke", "Parkinson's Disease",
        "Multiple Sclerosis", "Alzheimer's Disease", "Neuropathy", "Brain Tumor"
    ],
    "Pediatrics": [
        "Asthma", "ADHD", "Autism Spectrum Disorder", "Chickenpox",
        "Measles", "Ear Infections", "Croup", "Diabetes Type 1"
    ],
    "Oncology": [
        "Breast Cancer", "Prostate Cancer", "Lung Cancer", "Colorectal Cancer",
        "Leukemia", "Lymphoma", "Ovarian Cancer", "Melanoma"
    ],
    "Dermatology": [
        "Acne", "Eczema", "Psoriasis", "Rosacea", "Skin Cancer",
        "Hives", "Fungal Infections", "Vitiligo"
    ]
}

# Gender-specific diseases
gender_specific_diseases = {
    "Male": ["Prostate Cancer", "Erectile Dysfunction"],
    "Female": ["Ovarian Cancer", "Endometriosis", "Breast Cancer"]
}

def assign_disease(specialization, gender):
    diseases = diseases_by_specialization[specialization]
    
    # Add gender-specific diseases if applicable
    if gender in gender_specific_diseases:
        diseases += gender_specific_diseases[gender]
    
    # Assign weights for non-uniform distribution
    weights = np.random.dirichlet(np.ones(len(diseases))) # Random weights
    return np.random.choice(diseases, p=weights)


# Generate 200 Doctors
doctor_ids = [f"D{i+1:03}" for i in range(200)]
specializations = ["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "Oncology", "Dermatology"]

doctors_data = {
    "Doctor ID": doctor_ids,
    "Name": [fake.name() for _ in range(200)],
    "Specialization": np.random.choice(specializations, size=200),
    "Location": np.random.choice(suburbs, size=200)
}
df_doctors = pd.DataFrame(doctors_data)

# Generate 1296 Patients
patient_ids = [f"P{i+1:04}" for i in range(1296)]
feedback_ids = [f"F{i+1:04}" for i in range(1296)]

patients_data = {
    "Patient ID": patient_ids,
    "Name": [fake.name() for _ in range(1296)],
    "Age": np.random.randint(1, 100, size=1296),
    "Gender": np.random.choice(["Male", "Female"], size=1296, p=[0.7, 0.3]),
    "Location": np.random.choice(suburbs, size=1296, p=probabilities),
    "Risk Level": np.random.choice(["Low", "Medium", "High"], size=1296, p=[0.6, 0.3, 0.1]),
    "Status": np.random.choice(["Died", "In Patient", "Out Patient", "Cured"], size=1296, p=[0.02, 0.3, 0.5, 0.18]),
    "Date of Visit": pd.date_range(start="2020-01-01", end="2022-12-31", periods=1296).strftime('%Y-%m-%d'),
    "Waiting Time": np.random.randint(5, 60, size=1296),
    "Feedback ID": feedback_ids,
    "Doctor ID": np.random.choice(doctor_ids, size=1296)  # Assign random doctors
}
patients_data["Disease"] = [
    assign_disease(df_doctors.loc[df_doctors["Doctor ID"] == doctor_id, "Specialization"].values[0], gender)
    for doctor_id, gender in zip(patients_data["Doctor ID"], patients_data["Gender"])
]
df_patients = pd.DataFrame(patients_data)


# Generate Appointments (1 per patient)
appointments_data = {
    "Appointment ID": [f"A{i+1:04}" for i in range(1296)],
    "Patient ID": np.random.choice(patient_ids, size=1296, replace=False),
    "Doctor ID": np.random.choice(doctor_ids, size=1296),
    "Appointment Date/Time": pd.date_range(start="2022-01-01", end="2022-12-31", periods=1296).strftime('%Y-%m-%d %H:%M:%S'),
    "Status": np.random.choice(["Completed", "Pending", "Cancelled"], size=1296, p=[0.7, 0.2, 0.1])
}
df_appointments = pd.DataFrame(appointments_data)

# Generate Billing (1 bill per patient)
billing_data = {
    "Bill ID": [f"B{i+1:04}" for i in range(1296)],
    "Patient ID": patient_ids,
    "Total Amount": np.round(np.random.uniform(100, 2000, size=1296), 2),
    "Amount Paid": lambda x: np.round(x["Total Amount"] * np.random.uniform(0.5, 1.0, size=1296), 2),
    "Outstanding Payment": lambda x: np.round(x["Total Amount"] - x["Amount Paid"], 2),
    "Payment Date": pd.date_range(start="2020-01-01", end="2022-12-31", periods=1296).strftime('%Y-%m-%d')
}
df_billing = pd.DataFrame(billing_data)
df_billing["Amount Paid"] = billing_data["Amount Paid"](df_billing)
df_billing["Outstanding Payment"] = billing_data["Outstanding Payment"](df_billing)

# Generate Inventory (50 items)
medical_items = ["Paracetamol", "Bandages", "Surgical Masks", "Antiseptic Cream", "Syringes",
                 "Gloves", "Thermometers", "Scissors", "Stethoscopes", "IV Bags"]
inventory_data = {
    "Item ID": [f"I{i+1:03}" for i in range(50)],
    "Item Name": np.random.choice(medical_items, size=50),
    "Quantity": np.random.randint(50, 2000, size=50),
    "Location": np.random.choice(suburbs, size=50),
    "Expiry Date": pd.date_range(start="2023-01-01", end="2025-12-31", periods=50).strftime('%Y-%m-%d')
}
df_inventory = pd.DataFrame(inventory_data)

# Generate Feedback (1 per patient)
feedback_data = {
    "Feedback ID": feedback_ids,
    "Experience (1-5 Stars)": np.random.randint(1, 6, size=1296),
    "Wait Time (1-5 Stars)": np.random.randint(1, 6, size=1296),
    "Cleanliness (1-5 Stars)": np.random.randint(1, 6, size=1296),
    "Care Received (1-5 Stars)": np.random.randint(1, 6, size=1296),
    "Referral (1-5 Stars)": np.random.randint(1, 6, size=1296),
    "Comments": [fake.sentence() if random.random() > 0.7 else "" for _ in range(1296)]
}
df_feedback = pd.DataFrame(feedback_data)

# Generate Profit Margin (per location)
profit_data = {
    "Location": suburbs,
    "Total Revenue": np.round(np.random.uniform(129600, 500000, size=len(suburbs)), 2),
    "Total Costs": np.round(np.random.uniform(50000, 300000, size=len(suburbs)), 2)
}
df_profit = pd.DataFrame(profit_data)
df_profit["Profit Margin (%)"] = np.round((df_profit["Total Revenue"] - df_profit["Total Costs"]) / df_profit["Total Revenue"] * 100, 2)

# Save to CSV
df_patients.to_csv("patients.csv", index=False)
df_doctors.to_csv("doctors.csv", index=False)
df_appointments.to_csv("appointments.csv", index=False)
df_billing.to_csv("billing.csv", index=False)
df_inventory.to_csv("inventory.csv", index=False)
df_feedback.to_csv("feedback.csv", index=False)
df_profit.to_csv("profit_margin.csv", index=False)

print("CSV files generated successfully!")


