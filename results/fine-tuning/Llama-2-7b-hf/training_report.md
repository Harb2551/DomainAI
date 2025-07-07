# Training Report: meta-llama-Llama-2-7b-hf_20250706_132320

**Generated:** 2025-07-06 13:27:21

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.1488
- **Training Duration:** 2.2 minutes
- **Training Speed:** 20.5 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** meta-llama/Llama-2-7b-hf
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** o_proj, gate_proj, down_proj, up_proj, v_proj, q_proj, k_proj

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
- **Initial Training Loss:** 3.7581
- **Final Training Loss:** 0.7398
- **Loss Improvement:** 3.0183
- **Best Validation Loss:** 0.7700
- **Final Validation Loss:** 0.7700

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/Llama-2-7b-hf`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-06T13:23:20.812969
- **Training Start:** 2025-07-06T13:25:09.225642
- **Training End:** 2025-07-06T13:27:20.632326
- **Total Duration:** 0:02:11.406684
