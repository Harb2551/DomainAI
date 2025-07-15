import json
from typing import Any
import os

class SaveUtils:
    @staticmethod
    def save_json(data: Any, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
