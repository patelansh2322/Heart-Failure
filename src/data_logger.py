import csv
import os

CSV_FILE = "data/user_data.csv"

HEADERS = ['Age', 'Sex', 'ChestPainType',	'RestingBP', 'Cholesterol',	'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',	'Oldpeak',	'ST_Slope',	'HeartDisease']

def init_csv():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
    
def log_row(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, pred):
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f1,
            f2,
            f3, 
            f4,
            f5, 
            f6, 
            f7, 
            f8,
            f9,
            f10,
            f11,
            pred
        ])
