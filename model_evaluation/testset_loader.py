import pandas as pd
from typing import List

class TestsetLoader:
    def __init__(self, testset_path: str):
        self.testset_path = testset_path
        self._df = None

    def load_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.testset_path)
        return self._df

    def load_descriptions(self) -> List[str]:
        df = self.load_dataframe()
        if 'business_description' in df.columns:
            return df['business_description'].tolist()
        return df.iloc[:, 0].tolist()

    def load_labels(self) -> List[str]:
        df = self.load_dataframe()
        if 'label' in df.columns:
            return df['label'].tolist()
        return ["" for _ in range(len(df))]
