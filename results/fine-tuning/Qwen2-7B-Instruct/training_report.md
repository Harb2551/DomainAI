# Training Report: Qwen-Qwen2-7B-Instruct_20250706_134919

**Generated:** 2025-07-06 13:53:21

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.4152
- **Training Duration:** 2.6 minutes
- **Training Speed:** 17.3 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** Qwen/Qwen2-7B-Instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** v_proj, up_proj, o_proj, k_proj, q_proj, gate_proj, down_proj

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
- **Initial Training Loss:** 5.0149
- **Final Training Loss:** 0.8765
- **Loss Improvement:** 4.1384
- **Best Validation Loss:** 0.9619
- **Final Validation Loss:** 0.9619

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/Qwen2-7B-Instruct`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-06T13:49:19.997624
- **Training Start:** 2025-07-06T13:50:39.790878
- **Training End:** 2025-07-06T13:53:15.587321
- **Total Duration:** 0:02:35.796443
