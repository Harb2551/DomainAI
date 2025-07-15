# Training Report: .-local_models-meta-llama-Llama-2-7b-hf_20250713_153553

**Generated:** 2025-07-13 15:38:11

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.1393
- **Training Duration:** 2.2 minutes
- **Training Speed:** 20.6 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/meta-llama-Llama-2-7b-hf
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
- **Initial Training Loss:** 3.7385
- **Final Training Loss:** 0.7415
- **Loss Improvement:** 2.9970
- **Best Validation Loss:** 0.7765
- **Final Validation Loss:** 0.7765

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine_tuning_v1/meta-llama-Llama-2-7b-hf`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-13T15:35:53.836366
- **Training Start:** 2025-07-13T15:35:59.799681
- **Training End:** 2025-07-13T15:38:10.726095
- **Total Duration:** 0:02:10.926414
