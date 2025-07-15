"""
Split the combined evaluation dataset into train/val/test sets, ensuring all label types from both normal and edge cases are represented in each split.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

EVAL_PATH = "./synthetic_domain_dataset_eval.csv"
TRAIN_PATH = "./synthetic_domain_dataset_train.csv"
VAL_PATH = "./synthetic_domain_dataset_val.csv"
TEST_PATH = "./synthetic_domain_dataset_test.csv"

# Load combined evaluation dataset
df = pd.read_csv(EVAL_PATH)

# Ensure no missing labels
df = df.dropna(subset=["label"]).reset_index(drop=True)

# Remove labels with only 1 sample (cannot stratify)
label_counts = df['label'].value_counts()
df = df[df['label'].isin(label_counts[label_counts > 1].index)].reset_index(drop=True)

# Stratified split to preserve label distribution
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# Save splits
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"Train/val/test splits saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
