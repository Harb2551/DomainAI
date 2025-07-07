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
    # Change the model name here to switch between models
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct" 
    DATA_DIR = "./"  # Current directory (DomainAI)
    # Set output dir based on model name for clarity
    model_short_name = MODEL_NAME.split("/")[-1].replace('.', '-').replace('_', '-')
    OUTPUT_DIR = f"./{model_short_name}-finetuned-domainai"
    LOCAL_MODEL_DIR = "./local_models"
    pipeline = FinetunePipeline(MODEL_NAME, DATA_DIR, OUTPUT_DIR, LOCAL_MODEL_DIR, DomainModelTrainer)
    pipeline.run()
