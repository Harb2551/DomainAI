import os
from domain_model_trainer import DomainModelTrainer
import transformers
print('Transformers version:', transformers.__version__)
print('Transformers file:', transformers.__file__)

class FinetunePipeline:
    def __init__(self, model_name, data_dir, output_dir, local_model_dir, trainer_cls):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.local_model_dir = local_model_dir
        self.trainer_cls = trainer_cls

    def ensure_directories(self):
        # Ensure all necessary directories exist
        for path in [self.data_dir, self.output_dir, self.local_model_dir]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def run(self):
        self.ensure_directories()
        trainer = self.trainer_cls(
            self.model_name,
            self.output_dir,
            self.data_dir,
            self.local_model_dir
        )
        trainer.train()

if __name__ == "__main__":
    # Hard code the model directories found in local_models (excluding tokenizer dirs)
    LOCAL_MODEL_DIR = "./local_models"
    DATA_DIR = "./datasets/datasets_v1"  # Current directory (DomainAI)
    FINE_TUNED_MODELS_DIR = "./fine_tuned_models"
    if not os.path.exists(FINE_TUNED_MODELS_DIR):
        os.makedirs(FINE_TUNED_MODELS_DIR, exist_ok=True)
    MODEL_DIRS = [
        "Qwen-Qwen2-7B-Instruct",
        "meta-llama-Llama-2-7b-hf",
        "mistralai-Mistral-7B-v0.1"
    ]
    for model_dir in MODEL_DIRS:
        model_path = os.path.join(LOCAL_MODEL_DIR, model_dir)
        model_short_name = model_dir.replace('.', '-').replace('_', '-')
        OUTPUT_DIR = os.path.join(FINE_TUNED_MODELS_DIR, f"{model_short_name}-finetuned-domainai")
        print(f"\n[INFO] Fine-tuning model: {model_path}")
        pipeline = FinetunePipeline(model_path, DATA_DIR, OUTPUT_DIR, LOCAL_MODEL_DIR, DomainModelTrainer)
        pipeline.run()
