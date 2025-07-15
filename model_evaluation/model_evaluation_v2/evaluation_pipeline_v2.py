from typing import List
from ..model_evaluation_v1.evaluation_pipeline import evaluate_model
from ..model_evaluation_v1 import evaluation_pipeline as v1_eval
from .bedrock_llm_judge_v2 import BedrockLLMJudgeV2

class EnhancedEvaluationPipeline:
    
    def __init__(self, judge_class=BedrockLLMJudgeV2):
        """Initialize with judge dependency injection."""
        self._judge_class = judge_class
        self._original_judge = None
    
    def evaluate(self, model_dir: str, testset_path: str, results_dir: str, 
                criteria: List[str], bedrock_model_id: str, region: str = 'us-east-1',
                max_new_tokens: int = 10, tokenizer_dir: str = None):
        """Execute evaluation with enhanced judge."""
        
        # Store original and inject enhanced judge
        self._original_judge = v1_eval.BedrockLLMJudge
        v1_eval.BedrockLLMJudge = self._judge_class
        
        try:
            # Delegate to V1 evaluation (no redundancy)
            return evaluate_model(
                model_dir=model_dir,
                testset_path=testset_path,
                results_dir=results_dir,
                criteria=criteria,
                bedrock_model_id=bedrock_model_id,
                region=region,
                max_new_tokens=max_new_tokens,
                tokenizer_dir=tokenizer_dir
            )
        finally:
            # Restore original judge
            v1_eval.BedrockLLMJudge = self._original_judge

# Factory function for backward compatibility
def evaluate_model_v2(model_dir: str, testset_path: str, results_dir: str, 
                     criteria: List[str], bedrock_model_id: str, region: str = 'us-east-1',
                     max_new_tokens: int = 32, tokenizer_dir: str = None):
    """Factory function using enhanced pipeline."""
    pipeline = EnhancedEvaluationPipeline()
    return pipeline.evaluate(model_dir, testset_path, results_dir, criteria, 
                           bedrock_model_id, region, max_new_tokens, tokenizer_dir)

def main():
    """Example usage."""
    pipeline = EnhancedEvaluationPipeline()
    pipeline.evaluate(
        model_dir="./fine_tuned_models/meta-llama-Llama-2-7b-hf-finetuned-domainai",
        testset_path="./synthetic_domain_dataset_test.csv",
        results_dir="./results/evaluation_v2",
        criteria=["relevance", "appropriateness", "creativity"],
        bedrock_model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )

if __name__ == "__main__":
    main()