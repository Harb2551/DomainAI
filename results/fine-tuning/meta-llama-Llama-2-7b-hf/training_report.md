# Training Report: .-local_models-meta-llama-Llama-2-7b-hf_20250707_025039

**Generated:** 2025-07-07 02:52:55

## Executive Summary
- **Training Status:** ✅ SUCCESS
- **Final Training Loss:** 1.1393
- **Training Duration:** 2.1 minutes
- **Training Speed:** 20.9 samples/second
- **Trainable Parameters:** 0 (0.00%)

## Model Configuration
- **Base Model:** ./local_models/meta-llama-Llama-2-7b-hf
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** v_proj, down_proj, gate_proj, up_proj, k_proj, q_proj, o_proj

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
- **Initial Training Loss:** 3.7385
- **Final Training Loss:** 0.7415
- **Loss Improvement:** 2.9970
- **Best Validation Loss:** 0.7765
- **Final Validation Loss:** 0.7765

## Generated Files
- **Model Directory:** `/mnt/bfx/agentic_project/DomainAI/results/fine-tuning/meta-llama-Llama-2-7b-hf`
- **Experiment Log:** `experiment_log.json`
- **Training Summary:** `experiment_summary.json`
- **This Report:** `training_report.md`
- **Trained Model:** `pytorch_model.bin` + config files
- **Tokenizer:** tokenizer files

## Timeline
- **Experiment Start:** 2025-07-07T02:50:39.735706
- **Training Start:** 2025-07-07T02:50:45.474179
- **Training End:** 2025-07-07T02:52:54.577700
- **Total Duration:** 0:02:09.103521
