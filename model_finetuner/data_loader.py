import os
import pandas as pd
from datasets import Dataset

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self, split):
        file = os.path.join(self.data_dir, f"synthetic_domain_dataset_{split}.csv")
        df = pd.read_csv(file)
        def make_prompt(row):
                return f"Suggest a domain name for this business: {row['business_description']}\nDomain:"
        def make_label(row):
            if row['label'] == 'normal':
                return row['ideal_domain'] if pd.notnull(row['ideal_domain']) else ""
            else:
                return "[EDGE_CASE]"
        df['text'] = df.apply(make_prompt, axis=1)
        df['labels'] = df.apply(make_label, axis=1)
        return Dataset.from_pandas(df[['text', 'labels']])
