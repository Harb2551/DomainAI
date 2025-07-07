# Training Report: mistralai-Mistral-7B-v0.1_20250706_054802

**Generated:** 2025-07-06 05:50:26

## Executive Summary
- **Training Status:** SUCCESS
- **Final Training Loss:** 1.1685
- **Training Duration:** 2.3 minutes
- **Training Speed:** 19.8 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** mistralai/Mistral-7B-v0.1
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** up_proj, q_proj, v_proj, gate_proj, k_proj, down_proj, o_proj

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
- **Training Samples:** 896
- **Validation Samples:** 112
- **Total Samples:** 1,008
- **Train/Val Split:** 896/112
- **Data Directory:** ./

## System Information
- **GPU Available:** Yes
- **Number of GPUs:** 1
- **CUDA Version:** 12.6
- **PyTorch Version:** 2.7.1+cu126
- **Transformers Version:** N/A
- **CPU Cores:** 256
- **System Memory:** 1006.93 GB

## Training Progress
- **Initial Training Loss:** 4.4707
- **Final Training Loss:** 0.7517
- **Loss Improvement:** 3.7190
- **Best Validation Loss:** 0.8208
- **Final Validation Loss:** 0.8208

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/Mistral-7B-v0-1`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-06T05:48:02.902879
- **Training Start:** 2025-07-06T05:48:09.618748
- **Training End:** 2025-07-06T05:50:25.528428
- **Total Duration:** 0:02:15.909680
