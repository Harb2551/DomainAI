"""
Combine normal and edge case synthetic domain datasets into a single evaluation set.
"""
import pandas as pd

NORMAL_PATH = "./synthetic_domain_dataset_normal.csv"
EDGE_PATH = "./synthetic_domain_dataset_edge_cases.csv"
EVAL_PATH = "./synthetic_domain_dataset_eval.csv"

# Load datasets
df_normal = pd.read_csv(NORMAL_PATH)
df_edge = pd.read_csv(EDGE_PATH)

# Optionally shuffle each set (uncomment if desired)
# df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
# df_edge = df_edge.sample(frac=1, random_state=42).reset_index(drop=True)

# Concatenate datasets
df_eval = pd.concat([df_normal, df_edge], ignore_index=True)

# Save combined evaluation set
df_eval.to_csv(EVAL_PATH, index=False)
print(f"Combined evaluation set saved to {EVAL_PATH}")
