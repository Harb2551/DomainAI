# Training Report: .-local_models-meta-llama-Llama-2-7b-hf_20250712_145547

**Generated:** 2025-07-12 15:06:44

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.1294
- **Training Duration:** 10.0 minutes
- **Training Speed:** 21.2 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/meta-llama-Llama-2-7b-hf
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** down_proj, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj

## Training Configuration
- **Learning Rate:** 0.0002
- **Batch Size:** 8
- **Gradient Accumulation Steps:** 4
- **Effective Batch Size:** 32
- **Number of Epochs:** 3
- **Optimizer:** adamw_torch
- **Mixed Precision (FP16):** Yes
- **Gradient Checkpointing:** No

## Dataset Information
- **Training Samples:** 4,240
- **Validation Samples:** 530
- **Total Samples:** 4,770
- **Train/Val Split:** 4240/530
- **Data Directory:** ./datasets/datasets_v4

## System Information
- **GPU Available:** Yes
- **Number of GPUs:** 1
- **CUDA Version:** 12.6
- **PyTorch Version:** 2.7.1+cu126
- **Transformers Version:** N/A
- **CPU Cores:** 256
- **System Memory:** 1006.93 GB

## Training Progress
- **Initial Training Loss:** 3.3100
- **Final Training Loss:** 0.8578
- **Loss Improvement:** 2.4522
- **Best Validation Loss:** 1.1415
- **Final Validation Loss:** 1.1499

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/meta-llama-Llama-2-7b-hf`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-12T14:55:47.162599
- **Training Start:** 2025-07-12T14:56:43.703932
- **Training End:** 2025-07-12T15:06:43.999281
- **Total Duration:** 0:10:00.295349
