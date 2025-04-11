import pandas as pd
import numpy as np

# === CONFIGURATION ===
csv_path = 'DOE/LHS_DO4_Combined.csv'  # <-- replace with your actual file path
baseline_values = {
    'Cost': 1395.0,
    'Lead Time': 113.1,
    'Effectivness': 0.275,
    'Average Iterations': 7.5,
    'First Pass Yield': 0.075,
    'Rel Cost Physical': 0.72,
    'Work Efficency': 0.49,
    'Average Consitency': 0.66
}
output_path = 'DOE/LHS_DO4_Combined_norm_nans.csv'  # <-- where you want to save the result

all = False

filter_column = 'New Tool'  # <-- name of the column used to filter
cols_to_null_if_false = ['T_Acc (new)','T_I (new)','T_Usab (new)'] 

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# === CONDITIONAL NULLING ===
false_mask = df[filter_column].astype(str).str.lower().isin(['false', '0', 'no'])
df.loc[false_mask, cols_to_null_if_false] = np.nan

# === NORMALIZE EACH COLUMN BY ITS SPECIFIC VALUE ===
for col, norm_value in baseline_values.items():
    if not all and col not in {'Cost', 'Lead Time'}:
        continue
    if col in df.columns:
        df[col] = df[col] / norm_value
    else:
        print(f"Warning: Column '{col}' not found in the CSV.")

# === SAVE RESULT ===
df.to_csv(output_path, index=False)
print(f"Normalized data saved to {output_path}")