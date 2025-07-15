# Training Report: .-local_models-mistralai-Mistral-7B-v0.1_20250713_153811

**Generated:** 2025-07-13 15:40:34

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.1784
- **Training Duration:** 2.3 minutes
- **Training Speed:** 19.7 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/mistralai-Mistral-7B-v0.1
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
- **Initial Training Loss:** 4.4954
- **Final Training Loss:** 0.7631
- **Loss Improvement:** 3.7323
- **Best Validation Loss:** 0.8352
- **Final Validation Loss:** 0.8352

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine_tuning_v1/mistralai-Mistral-7B-v0-1`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-13T15:38:11.239674
- **Training Start:** 2025-07-13T15:38:17.132191
- **Training End:** 2025-07-13T15:40:33.683729
- **Total Duration:** 0:02:16.551538
