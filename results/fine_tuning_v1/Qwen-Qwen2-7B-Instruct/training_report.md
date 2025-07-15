# Training Report: .-local_models-Qwen-Qwen2-7B-Instruct_20250713_153256

**Generated:** 2025-07-13 15:35:53

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.4315
- **Training Duration:** 2.6 minutes
- **Training Speed:** 17.0 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/Qwen-Qwen2-7B-Instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** gate_proj, down_proj, k_proj, v_proj, o_proj, q_proj, up_proj

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
- **Data Directory:** ./datasets/datasets_v1

## System Information
- **GPU Available:** Yes
- **Number of GPUs:** 1
- **CUDA Version:** 12.6
- **PyTorch Version:** 2.7.1+cu126
- **Transformers Version:** N/A
- **CPU Cores:** 256
- **System Memory:** 1006.93 GB

## Training Progress
- **Initial Training Loss:** 5.0932
- **Final Training Loss:** 0.8968
- **Loss Improvement:** 4.1964
- **Best Validation Loss:** 0.9877
- **Final Validation Loss:** 0.9877

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine_tuning_v1/Qwen-Qwen2-7B-Instruct`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-13T15:32:56.636599
- **Training Start:** 2025-07-13T15:33:09.026837
- **Training End:** 2025-07-13T15:35:47.554561
- **Total Duration:** 0:02:38.527724
