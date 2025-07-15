# Training Report: .-local_models-mistralai-Mistral-7B-v0.1_20250712_150644

**Generated:** 2025-07-12 15:18:05

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.0157
- **Training Duration:** 10.5 minutes
- **Training Speed:** 20.2 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/mistralai-Mistral-7B-v0.1
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
- **Initial Training Loss:** 3.6521
- **Final Training Loss:** 0.6540
- **Loss Improvement:** 2.9981
- **Best Validation Loss:** 1.1662
- **Final Validation Loss:** 1.2669

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/mistralai-Mistral-7B-v0-1`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-12T15:06:44.498465
- **Training Start:** 2025-07-12T15:07:36.350957
- **Training End:** 2025-07-12T15:18:05.395701
- **Total Duration:** 0:10:29.044744
