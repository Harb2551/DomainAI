"""
Main entry point for Model Evaluation V3 - Multi-Judge Pipeline
"""

from .multijudge_pipeline import MultiJudgeEvaluationPipeline


def main():
    """
    Main function that configures and runs the multi-judge evaluation for all models in fine_tuned_models_v4.
    """
    
    # =================================================================
    # CONFIGURATION - Modify these variables as needed
    # =================================================================
    
    # Model and data paths
    model_dirs = [
        "./fine_tuned_models/Qwen-Qwen2-7B-Instruct-finetuned-domainai",
        "./fine_tuned_models/mistralai-Mistral-7B-v0-1-finetuned-domainai",  # Replace with actual model folder name
        "./fine_tuned_models/meta-llama-Llama-2-7b-hf-finetuned-domainai"   # Replace with actual model folder name
    ]
    tokenizer_dirs = [
        "./local_models/.-local_models-Qwen-Qwen2-7B-Instruct-tokenizer",
        "./local_models/.-local_models-mistralai-Mistral-7B-v0.1-tokenizer",  # Replace with actual tokenizer folder name
        "./local_models/.-local_models-meta-llama-Llama-2-7b-hf-tokenizer"   # Replace with actual tokenizer folder name
    ]
    testset_path = "./datasets/datasets_v4/synthetic_domain_dataset_test.csv"
    base_results_dir = "./results/evaluation_v3/evaluation_v3_new_testset"
    
    # Evaluation criteria
    criteria = ["relevance", "appropriateness", "creativity"]
    
    # Judge models for evaluation
    judge_models = [
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude Sonnet
        "us.deepseek.r1-v1:0",                            # DeepSeek R1
        "meta.llama3-70b-instruct-v1:0"               # Llama 3 70B
    ]
    
    # AWS configuration
    region = "us-east-1"
    
    # Generation parameters
    max_new_tokens = 10
    
    # =================================================================
    # EXECUTION
    # =================================================================
    
    # Loop over each model directory and its corresponding tokenizer
    for model_dir, tokenizer_dir in zip(model_dirs, tokenizer_dirs):
        print(f"Running evaluation for model: {model_dir}")
        pipeline = MultiJudgeEvaluationPipeline(
            model_dir=model_dir,
            testset_path=testset_path,
            base_results_dir=base_results_dir,
            criteria=criteria,
            judge_models=judge_models,
            region=region,
            max_new_tokens=max_new_tokens,
            tokenizer_dir=tokenizer_dir
        )
        
        pipeline.run_evaluation()


if __name__ == "__main__":
    main()