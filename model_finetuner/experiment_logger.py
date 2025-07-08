import os
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_id, model_name, output_dir):
        self.experiment_log = {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'metrics': {},
            'system_info': {},
            'training_history': [],
            'dataset_info': {},
            'model_stats': {}
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def log_config(self, config_dict):
        self.experiment_log['config'].update(config_dict)

    def log_metrics(self, metrics_dict):
        self.experiment_log['metrics'].update(metrics_dict)

    def log_system_info(self, system_info_dict):
        self.experiment_log['system_info'].update(system_info_dict)

    def log_training_history(self, history):
        self.experiment_log['training_history'] = history

    def log_dataset_info(self, dataset_info):
        self.experiment_log['dataset_info'] = dataset_info

    def log_model_stats(self, model_stats):
        self.experiment_log['model_stats'] = model_stats

    def save(self, suffix=""):
        log_file = os.path.join(self.output_dir, f'experiment_log{suffix}.json')
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        print(f"Experiment log saved to {log_file}")

    def create_summary(self):
        summary = {
            'experiment_id': self.experiment_log['experiment_id'],
            'timestamp': self.experiment_log['timestamp'],
            'model_name': self.experiment_log['model_name'],
            'training_successful': self.experiment_log['metrics'].get('training_successful', False),
            'key_metrics': {
                'final_train_loss': self.experiment_log['metrics'].get('final_train_loss'),
                'training_time_minutes': self.experiment_log['metrics'].get('train_runtime_minutes'),
                'trainable_parameters': self.experiment_log['model_stats'].get('trainable_parameters'),
                'trainable_percentage': self.experiment_log['model_stats'].get('trainable_percentage'),
                'train_samples': self.experiment_log['dataset_info'].get('train_size'),
                'val_samples': self.experiment_log['dataset_info'].get('val_size')
            },
            'configuration': {
                'lora_rank': self.experiment_log['config'].get('lora_config', {}).get('r'),
                'learning_rate': self.experiment_log['config'].get('training_args', {}).get('learning_rate'),
                'batch_size': self.experiment_log['config'].get('training_args', {}).get('per_device_train_batch_size'),
                'epochs': self.experiment_log['config'].get('training_args', {}).get('num_train_epochs')
            }
        }
        summary_file = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Experiment summary saved to {summary_file}")

    def create_training_report(self):
        report_file = os.path.join(self.output_dir, 'training_report.md')
        with open(report_file, 'w') as f:
            f.write(f"# Training Report: {self.experiment_log['experiment_id']}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Executive Summary\n")
            training_success = self.experiment_log['metrics'].get('training_successful', False)
            f.write(f"- **Training Status:** {'✅ SUCCESS' if training_success else '❌ FAILED'}\n")
            if training_success:
                f.write(f"- **Final Training Loss:** {self.experiment_log['metrics'].get('final_train_loss', 'N/A'):.4f}\n")
                f.write(f"- **Training Duration:** {self.experiment_log['metrics'].get('train_runtime_minutes', 0):.1f} minutes\n")
                f.write(f"- **Training Speed:** {self.experiment_log['metrics'].get('train_samples_per_second', 0):.1f} samples/second\n")
            f.write(f"- **Trainable Parameters:** {self.experiment_log['model_stats'].get('trainable_parameters', 0):,} ({self.experiment_log['model_stats'].get('trainable_percentage', 0):.2f}%)\n\n")
            f.write("## Model Configuration\n")
            f.write(f"- **Base Model:** {self.experiment_log['model_name']}\n")
            f.write(f"- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)\n")
            lora_config = self.experiment_log['config'].get('lora_config', {})
            f.write(f"- **LoRA Rank (r):** {lora_config.get('r')}\n")
            f.write(f"- **LoRA Alpha:** {lora_config.get('lora_alpha')}\n")
            f.write(f"- **LoRA Dropout:** {lora_config.get('lora_dropout')}\n")
            f.write(f"- **Target Modules:** {', '.join(lora_config.get('target_modules', []))}\n\n")
            f.write("## Training Configuration\n")
            train_config = self.experiment_log['config'].get('training_args', {})
            f.write(f"- **Learning Rate:** {train_config.get('learning_rate')}\n")
            f.write(f"- **Batch Size:** {train_config.get('per_device_train_batch_size')}\n")
            f.write(f"- **Gradient Accumulation Steps:** {train_config.get('gradient_accumulation_steps')}\n")
            f.write(f"- **Effective Batch Size:** {train_config.get('effective_batch_size')}\n")
            f.write(f"- **Number of Epochs:** {train_config.get('num_train_epochs')}\n")
            f.write(f"- **Optimizer:** {train_config.get('optimizer')}\n")
            f.write(f"- **Mixed Precision (FP16):** {'Yes' if train_config.get('fp16') else 'No'}\n")
            f.write(f"- **Gradient Checkpointing:** {'Yes' if train_config.get('gradient_checkpointing') else 'No'}\n\n")
            dataset_info = self.experiment_log['dataset_info']
            f.write("## Dataset Information\n")
            f.write(f"- **Training Samples:** {dataset_info.get('train_size', 'N/A'):,}\n")
            f.write(f"- **Validation Samples:** {dataset_info.get('val_size', 'N/A'):,}\n")
            f.write(f"- **Total Samples:** {dataset_info.get('total_size', 'N/A'):,}\n")
            f.write(f"- **Train/Val Split:** {dataset_info.get('train_val_split', 'N/A')}\n")
            if 'max_sequence_length' in dataset_info:
                f.write(f"- **Max Sequence Length:** {dataset_info['max_sequence_length']}\n")
            f.write(f"- **Data Directory:** {dataset_info.get('data_dir', 'N/A')}\n\n")
            system_info = self.experiment_log['system_info']
            f.write("## System Information\n")
            f.write(f"- **GPU Available:** {'Yes' if system_info.get('gpu_available') else 'No'}\n")
            f.write(f"- **Number of GPUs:** {system_info.get('gpu_count', 0)}\n")
            if system_info.get('gpu_info'):
                for gpu in system_info['gpu_info']:
                    if 'name' in gpu:
                        f.write(f"- **GPU Model:** {gpu['name']}\n")
                    if 'memory_total' in gpu:
                        f.write(f"- **GPU Memory:** {gpu['memory_total']}\n")
            f.write(f"- **CUDA Version:** {system_info.get('cuda_version', 'N/A')}\n")
            f.write(f"- **PyTorch Version:** {system_info.get('torch_version', 'N/A')}\n")
            f.write(f"- **Transformers Version:** {system_info.get('transformers_version', 'N/A')}\n")
            f.write(f"- **CPU Cores:** {system_info.get('cpu_count', 'N/A')}\n")
            f.write(f"- **System Memory:** {system_info.get('memory_total_gb', 'N/A')} GB\n\n")
            if training_success and 'training_progression' in self.experiment_log['metrics']:
                f.write("## Training Progress\n")
                progress = self.experiment_log['metrics']['training_progression']
                f.write(f"- **Initial Training Loss:** {progress.get('initial_train_loss', 'N/A'):.4f}\n")
                f.write(f"- **Final Training Loss:** {progress.get('final_train_loss', 'N/A'):.4f}\n")
                if progress.get('loss_improvement'):
                    f.write(f"- **Loss Improvement:** {progress['loss_improvement']:.4f}\n")
                if progress.get('best_eval_loss'):
                    f.write(f"- **Best Validation Loss:** {progress['best_eval_loss']:.4f}\n")
                if progress.get('final_eval_loss'):
                    f.write(f"- **Final Validation Loss:** {progress['final_eval_loss']:.4f}\n")
                f.write("\n")
            f.write("## Generated Files\n")
            f.write(f"- **Model Directory:** `{self.output_dir}`\n")
            f.write(f"- **Experiment Log:** `experiment_log.json`\n")
            f.write(f"- **Training Summary:** `experiment_summary.json`\n")
            f.write(f"- **This Report:** `training_report.md`\n")
            f.write(f"- **Trained Model:** `pytorch_model.bin` + config files\n")
            f.write(f"- **Tokenizer:** tokenizer files\n\n")
            f.write("## Timeline\n")
            f.write(f"- **Experiment Start:** {self.experiment_log.get('timestamp')}\n")
            f.write(f"- **Training Start:** {self.experiment_log.get('training_start_time')}\n")
            f.write(f"- **Training End:** {self.experiment_log.get('training_end_time')}\n")
            f.write(f"- **Total Duration:** {self.experiment_log.get('total_training_duration')}\n")
        print(f"Training report saved to {report_file}")
