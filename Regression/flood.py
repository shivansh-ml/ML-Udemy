import pandas as pd
import numpy as np
import math

CSV_PATH = "rainfaLLIndia (1).csv"

MONTHS = ["JUN", "JUL", "AUG", "SEP"]

STATE_WEIGHTS = {
    "ASSAM & MEGHALAYA": 1.45,
    "BIHAR": 1.45,
    "SUB-HIMALAYAN WEST BENGAL": 1.45,
    "GANGETIC WEST BENGAL": 1.45,
    "EAST UTTAR PRADESH": 1.30,
    "WEST UTTAR PRADESH": 1.30,
    "ODISHA": 1.30,
    "ORISSA": 1.30,
    "COASTAL ANDHRA PRADESH": 1.25,
    "KERALA": 1.25,
    "KONKAN & GOA": 1.25,
    "GUJARAT": 1.15,
    "CHHATTISGARH": 1.00,
    "MADHYA PRADESH": 1.00,
    "TELANGANA": 1.00,
    "RAYALSEEMA": 1.00,
    "MAHARASHTRA": 1.00,
    "KARNATAKA": 1.00,
    "ANDAMAN & NICOBAR ISLANDS": 1.15,
    "RAJASTHAN": 0.80,
    "WEST RAJASTHAN": 0.80,
    "EAST RAJASTHAN": 0.80
}

def normalize_name(s):
    return str(s).upper().replace("&", "AND").replace(".", "").strip()

df = pd.read_csv(CSV_PATH)
df.columns = [c.upper() for c in df.columns]

df["SUBDIVISION_NORM"] = df["SUBDIVISION"].apply(normalize_name)

month = input("Enter month (JUN/JUL/AUG/SEP): ").strip().upper()
if month not in MONTHS:
    raise ValueError("Only JUN, JUL, AUG, SEP are allowed")

results = []

for sub in sorted(df["SUBDIVISION_NORM"].unique()):
    sub_df = df[df["SUBDIVISION_NORM"] == sub]
    values = sub_df[month].dropna().astype(float).values

    if len(values) < 5:
        continue

    threshold = np.percentile(values, 85)
    exceed_count = np.sum(values >= threshold)
    raw_freq = exceed_count / len(values)

    weight = STATE_WEIGHTS.get(sub, 1.0)

    R = raw_freq * weight

    k = 12
    c = 0.18

    flood_prob = 1 / (1 + math.exp(-k * (R - c)))
    flood_prob = round(flood_prob * 100, 1)

    results.append([sub, flood_prob])

output = pd.DataFrame(results, columns=["SUBDIVISION", "FLOOD_PROB_%"])
output = output.sort_values("FLOOD_PROB_%", ascending=False)

print("\nSTATE-WISE FLOOD PROBABILITY (OFFLINE HARDLINE MODEL)\n")
print(output.to_string(index=False))
