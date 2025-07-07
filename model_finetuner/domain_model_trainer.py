from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model_and_tokenizer_manager import ModelAndTokenizerManager
from experiment_logger import ExperimentLogger
from system_info import get_system_info
from data_loader import DataLoader
from domain_dataset_preprocessor import DomainDatasetPreprocessor
from peft import LoraConfig, get_peft_model
from model_loader import ModelLoader
import os
import torch
import json
import transformers  
from datetime import datetime
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
    print("GPUtil not installed. GPU details will be limited.")

class DomainModelTrainer:
    def __init__(self, model_name, output_dir, data_dir, local_model_dir="../local_models"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.local_model_dir = local_model_dir
        # Set up results directory structure inside DomainAI
        model_short_name = self.model_name.split("/")[-1].replace('.', '-').replace('_', '-')
        self.results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fine-tuning", model_short_name)
        self.results_dir = os.path.abspath(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        experiment_id = f"{model_name.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = ExperimentLogger(experiment_id, model_name, self.results_dir)
        # LoRA config
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Model/tokenizer manager
        self.model_mgr = ModelAndTokenizerManager(model_name, local_model_dir, self.lora_config)
        print("Getting model path...")
        model_path = ModelLoader(self.model_name, self.local_model_dir).get_model_path()
        print("Loading tokenizer from", model_path)
        self.model_mgr.load_tokenizer(model_path)
        print("Tokenizer loaded.")
        print("Saving tokenizer...")
        tokenizer_save_path = self.model_mgr.save_tokenizer()
        print("Tokenizer saved.")
        print("Loading model from", model_path)
        self.model_mgr.load_model(model_path)
        print("Model loaded.")
        print("Resizing model embeddings if needed...")
        self.model_mgr.resize_model_embeddings()
        # Do NOT apply LoRA or save/reload here!
        self.tokenizer = self.model_mgr.tokenizer
        self.model = self.model_mgr.model
        self.data_loader = DataLoader(self.data_dir)
        self.preprocessor = DomainDatasetPreprocessor(self.tokenizer)
        # Log system info
        self.logger.log_system_info(get_system_info())
        # Log LoRA config
        self.logger.log_config({'lora_config': {
            'r': self.lora_config.r,
            'lora_alpha': self.lora_config.lora_alpha,
            'target_modules': self.lora_config.target_modules,
            'lora_dropout': self.lora_config.lora_dropout,
            'bias': self.lora_config.bias,
            'task_type': self.lora_config.task_type
        }})

    def prepare_datasets(self):
        print("Loading datasets...")
        train_dataset = self.data_loader.load("train")
        val_dataset = self.data_loader.load("val")
        print("First train example keys:", train_dataset[0].keys())
        self.logger.log_dataset_info({
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'total_size': len(train_dataset) + len(val_dataset),
            'train_val_split': f"{len(train_dataset)}/{len(val_dataset)}",
            'data_dir': self.data_dir,
            'sample_train_example': {
                'text_length': len(train_dataset[0]['text']) if 'text' in train_dataset[0] else 0,
                'keys': list(train_dataset[0].keys())
            }
        })
        print("Preprocessing datasets...")
        train_dataset = train_dataset.map(self.preprocessor, remove_columns=["text", "labels"])
        val_dataset = val_dataset.map(self.preprocessor, remove_columns=["text", "labels"])
        print(f"Dataset prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_dataset, val_dataset

    def train(self):
        print("=" * 60)
        print(f"STARTING EXPERIMENT: {self.logger.experiment_log['experiment_id']}")
        print("=" * 60)
        train_dataset, val_dataset = self.prepare_datasets()
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=1,
            learning_rate=2e-4,
            fp16=True,
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
            save_total_limit=2,
            report_to="none",
            push_to_hub=False,
            max_grad_norm=1.0,
            optim="adamw_torch",
            warmup_steps=10,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        self.logger.log_config({'training_args': {
            'per_device_train_batch_size': training_args.per_device_train_batch_size,
            'per_device_eval_batch_size': training_args.per_device_eval_batch_size,
            'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
            'effective_batch_size': training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * max(1, torch.cuda.device_count()),
            'num_train_epochs': training_args.num_train_epochs,
            'learning_rate': training_args.learning_rate,
            'fp16': training_args.fp16,
            'gradient_checkpointing': training_args.gradient_checkpointing,
            'max_grad_norm': training_args.max_grad_norm,
            'optimizer': training_args.optim,
            'warmup_steps': training_args.warmup_steps,
            'eval_strategy': training_args.eval_strategy,
            'save_strategy': training_args.save_strategy,
            'logging_steps': training_args.logging_steps
        }})
        total_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs
        self.logger.experiment_log['config']['estimated_total_steps'] = total_steps
        # Apply LoRA and set model to train mode just before Trainer
        print("Applying LoRA...")
        self.model = self.model_mgr.apply_lora()
        self.model.train()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        print("Checking model parameters before training...")
        trainable_count = 0
        trainable_params_detail = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                if trainable_count <= 5:
                    param_info = {
                        'name': name,
                        'shape': list(param.shape),
                        'dtype': str(param.dtype),
                        'requires_grad': param.requires_grad,
                        'numel': param.numel()
                    }
                    trainable_params_detail.append(param_info)
                    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")
        self.logger.log_model_stats({
            'trainable_params_detail': trainable_params_detail,
            'total_trainable_params_verified': trainable_count
        })
        print(f"Total trainable parameters verified: {trainable_count}")
        self.logger.save("_pre_training")
        print("Starting training...")
        start_time = datetime.now()
        self.logger.experiment_log['training_start_time'] = start_time.isoformat()
        try:
            result = trainer.train()
            training_successful = True
        except Exception as e:
            print(f"Training failed with error: {e}")
            self.logger.log_metrics({'training_successful': False, 'error_details': str(e)})
            training_successful = False
            result = None
        end_time = datetime.now()
        self.logger.experiment_log['training_end_time'] = end_time.isoformat()
        self.logger.experiment_log['total_training_duration'] = str(end_time - start_time)
        if training_successful and result:
            self.logger.log_metrics({
                'training_successful': True,
                'final_train_loss': result.training_loss,
                'train_runtime_seconds': result.metrics.get('train_runtime', 0),
                'train_runtime_minutes': result.metrics.get('train_runtime', 0) / 60,
                'train_samples_per_second': result.metrics.get('train_samples_per_second', 0),
                'train_steps_per_second': result.metrics.get('train_steps_per_second', 0),
                'epoch': trainer.state.epoch,
                'global_step': result.global_step,
                'total_flos': result.metrics.get('total_flos', 0)
            })
            if hasattr(trainer.state, 'log_history'):
                self.logger.log_training_history(trainer.state.log_history)
                train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
                eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
                learning_rates = [log['learning_rate'] for log in trainer.state.log_history if 'learning_rate' in log]
                self.logger.log_metrics({'training_progression': {
                    'initial_train_loss': train_losses[0] if train_losses else None,
                    'final_train_loss': train_losses[-1] if train_losses else None,
                    'loss_improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else None,
                    'best_eval_loss': min(eval_losses) if eval_losses else None,
                    'final_eval_loss': eval_losses[-1] if eval_losses else None,
                    'peak_learning_rate': max(learning_rates) if learning_rates else None,
                    'final_learning_rate': learning_rates[-1] if learning_rates else None
                }})
            print("Saving trained model...")
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            if torch.cuda.is_available():
                self.logger.log_metrics({'final_gpu_memory': {
                    'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
                }})
        self.logger.save()
        self.logger.create_summary()
        self.logger.create_training_report()
        print("=" * 60)
        print(f"EXPERIMENT COMPLETED: {self.logger.experiment_log['experiment_id']}")
        print(f"Results saved to: {self.output_dir}")
        if training_successful:
            print(f"Final Training Loss: {self.logger.experiment_log['metrics'].get('final_train_loss', 'N/A')}")
            print(f"Training Time: {self.logger.experiment_log['metrics'].get('train_runtime_minutes', 0):.1f} minutes")
        print("=" * 60)
        return self.logger.experiment_log