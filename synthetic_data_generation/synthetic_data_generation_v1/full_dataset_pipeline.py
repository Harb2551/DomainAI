"""
Pipeline to generate normal and edge datasets, combine them, and split into train/val/test sets for fine-tuning and evaluation.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from generate_normal_cases import NormalCaseGenerator
from generate_edge_cases import EdgeCaseGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NORMAL_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_normal.csv")
EDGE_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_edge_cases.csv")
EVAL_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_eval.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_val.csv")
TEST_PATH = os.path.join(BASE_DIR, "synthetic_domain_dataset_test.csv")

# 1. Generate normal cases
def generate_normal():
    print("Generating normal cases...")
    normal_gen = NormalCaseGenerator(n=1000)
    normal_gen.save(NORMAL_PATH)

# 2. Generate edge/inappropriate cases
def generate_edge():
    print("Generating edge/inappropriate cases...")
    edge_gen = EdgeCaseGenerator(inappropriate_count=50)
    edge_df = edge_gen.generate()
    edge_df.to_csv(EDGE_PATH, index=False)
    print(f"Edge/inappropriate cases saved to {EDGE_PATH}")

# 3. Combine datasets
def combine():
    print("Combining normal and edge datasets...")
    df_normal = pd.read_csv(NORMAL_PATH)
    df_edge = pd.read_csv(EDGE_PATH)
    df_eval = pd.concat([df_normal, df_edge], ignore_index=True)
    df_eval.to_csv(EVAL_PATH, index=False)
    print(f"Combined evaluation set saved to {EVAL_PATH}")

# 4. Split for fine-tuning and evaluation
def split():
    print("Splitting combined dataset into train/val/test...")
    df = pd.read_csv(EVAL_PATH)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    label_counts = df['label'].value_counts()
    df = df[df['label'].isin(label_counts[label_counts > 1].index)].reset_index(drop=True)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    label_counts_temp = temp_df['label'].value_counts()
    rare_labels_temp = label_counts_temp[label_counts_temp == 1].index.tolist()
    if rare_labels_temp:
        print(f"Warning: The following labels in temp set have only 1 sample and will be dropped before val/test split: {rare_labels_temp}")
        temp_df = temp_df[~temp_df['label'].isin(rare_labels_temp)].reset_index(drop=True)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Train/val/test splits saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")

if __name__ == "__main__":
    generate_normal()
    generate_edge()
    combine()
    split()
    print("All datasets generated, combined, and split!")
