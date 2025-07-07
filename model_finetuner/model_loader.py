import os
import pandas as pd
from datasets import Dataset

class ModelLoader:
    def __init__(self, model_name, local_dir):
        self.model_name = model_name
        self.local_dir = local_dir
        self.model_path = os.path.join(self.local_dir, self.model_name.replace('/', '-'))

    def get_model_path(self):
        from huggingface_hub import snapshot_download
        if not os.path.exists(self.model_path):
            print(f"Model not found locally. Downloading {self.model_name} to {self.model_path}...")
            snapshot_download(self.model_name, local_dir=self.model_path)  # Download all files, including .safetensors
        else:
            print(f"Model found locally at {self.model_path}.")
        return self.model_path
