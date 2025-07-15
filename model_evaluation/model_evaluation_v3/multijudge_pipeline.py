"""
Multi-Judge Evaluation Pipeline
Pipeline that generates model outputs once and evaluates them with multiple judge models.
"""

import os
import json
from typing import List
from ..model_evaluation_v1.evaluation_pipeline import (
    load_test_data, 
    generate_domains, 
    score_domains
)
from ..model_evaluation_v1.hf_model_inferencer import HFModelInferencer
from ..model_evaluation_v1.bedrock_llm_judge import BedrockLLMJudge


class MultiJudgeEvaluationPipeline:
    """
    Pipeline that generates model outputs once and evaluates them with multiple judge models.
    """
    
    def __init__(self, model_dir: str, testset_path: str, base_results_dir: str,
                 criteria: List[str], judge_models: List[str], region: str = "us-east-1",
                 max_new_tokens: int = 10, tokenizer_dir: str = None):
        """
        Initialize the multi-judge evaluation pipeline.
        
        Args:
            model_dir: Path to the fine-tuned model
            testset_path: Path to test dataset
            base_results_dir: Base directory for results
            criteria: List of evaluation criteria
            judge_models: List of judge model IDs
            region: AWS region
            max_new_tokens: Max tokens for generation
            tokenizer_dir: Path to tokenizer directory
        """
        self.model_dir = model_dir
        self.testset_path = testset_path
        self.base_results_dir = base_results_dir
        self.criteria = criteria
        self.judge_models = judge_models
        self.region = region
        self.max_new_tokens = max_new_tokens
        self.tokenizer_dir = tokenizer_dir
    
    def run_evaluation(self):
        """
        Run evaluation with all judge models using the same generated outputs.
        """
        print("[INFO] Starting Model Evaluation V3 - Multi-Judge Pipeline")
        print(f"[INFO] Model: {self.model_dir}")
        print(f"[INFO] Test set: {self.testset_path}")
        print(f"[INFO] Will run evaluation with {len(self.judge_models)} different judges")
        
        # Step 1: Load test data
        descriptions, labels = load_test_data(self.testset_path)
        
        # Step 2: Load HF model and generate domains once
        print(f"[INFO] Loading HuggingFace model from {self.model_dir}")
        inferencer = HFModelInferencer(self.model_dir, tokenizer_dir=self.tokenizer_dir)
        hf_results, hf_total_time = generate_domains(inferencer, descriptions, self.max_new_tokens)
        
        # Step 3: Process predicted and ground truth labels
        predicted_labels = []
        ground_truth_labels = []
        
        for label, hf_result in zip(labels, hf_results):
            domain = hf_result["domain"].strip().lower() if hf_result["domain"] else ""
            
            # Predicted label: 'edge' if domain is an edge case marker, else 'normal'
            if domain.startswith("[edge") or domain.startswith("[edg") or domain.startswith("[ed"):
                predicted_labels.append("edge")
                hf_result["status"] = "blocked"
            else:
                predicted_labels.append("normal")
                hf_result["status"] = "success"
            
            # Ground truth label: 'edge' if label is not 'normal', else 'normal'
            if label and label != "normal":
                ground_truth_labels.append("edge")
            else:
                ground_truth_labels.append("normal")
        
        # Step 4: Evaluate with each judge model using the same hf_results
        model_name = self._get_model_name(self.model_dir)
        success_count = 0
        
        for i, judge_model in enumerate(self.judge_models):
            judge_name = self._get_judge_name(judge_model)
            results_dir = f"{self.base_results_dir}/{model_name}/{judge_name}"
            
            print(f"\n[INFO] Running evaluation {i+1}/{len(self.judge_models)} with judge: {judge_name}")
            print(f"[INFO] Results will be saved to: {results_dir}")
            
            try:
                # Create judge instance
                print(f"[INFO] Initializing Bedrock LLM Judge with model {judge_model}")
                judge = BedrockLLMJudge(judge_model, region=self.region)
                
                # Score domains with this judge using the same hf_results
                results = score_domains(
                    judge, descriptions, hf_results, ground_truth_labels, 
                    predicted_labels, self.criteria, hf_total_time
                )
                
                # Save results
                os.makedirs(results_dir, exist_ok=True)
                out_path = os.path.join(results_dir, 'evaluation_results.json')
                
                print(f"[INFO] Saving results to {out_path}")
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"[INFO] Saved evaluation results for {model_name} to {out_path}")
                
                # Compute and save metrics
                self._save_metrics(results, results_dir, model_name, hf_total_time)
                
                print(f"[SUCCESS] Evaluation with {judge_name} completed!")
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] Evaluation with {judge_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        self._print_summary(success_count)
    
    def _save_metrics(self, results, results_dir, model_name, hf_total_time):
        """Compute and save summary metrics."""
        print(f"[INFO] Computing summary metrics for {model_name}")
        
        # Extract scores and times
        n = len(results)
        sum_scores = {c: 0.0 for c in self.criteria}
        count_scores = {c: 0 for c in self.criteria}
        total_hf_time = 0.0
        total_llm_time = 0.0
        
        for r in results:
            for c in self.criteria:
                v = r['scores'].get(c)
                if isinstance(v, (int, float)):
                    sum_scores[c] += v
                    count_scores[c] += 1
            total_hf_time += r.get('hf_inference_time', 0.0)
            total_llm_time += r.get('llm_inference_time', 0.0)
        
        avg_scores = {c: (sum_scores[c] / count_scores[c] if count_scores[c] else None) for c in self.criteria}
        avg_hf_time = total_hf_time / n if n else None
        avg_llm_time = total_llm_time / n if n else None
        total_eval_time = hf_total_time + total_llm_time
        
        metrics = {
            'model_name': model_name,
            'num_cases': n,
            'criteria': self.criteria,
            'average_scores': avg_scores,
            'average_hf_inference_time': avg_hf_time,
            'average_llm_inference_time': avg_llm_time,
            'total_hf_inference_time': hf_total_time,
            'total_llm_inference_time': total_llm_time,
            'total_evaluation_time': total_eval_time
        }
        
        metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved summary metrics for {model_name} to {metrics_path}")
    
    def _get_judge_name(self, judge_model: str) -> str:
        """Extract readable name from judge model ID."""
        return judge_model.split('.')[-1]
    
    def _get_model_name(self, model_dir: str) -> str:
        """Extract model name from model directory path."""
        return os.path.basename(model_dir.rstrip('/'))
    
    def _print_summary(self, success_count: int):
        """Print evaluation summary."""
        model_name = self._get_model_name(self.model_dir)
        print(f"\n[SUMMARY] Completed {success_count}/{len(self.judge_models)} evaluations")
        print(f"[INFO] All judges evaluated the same model outputs (kept in memory)")
        print(f"[INFO] Results saved in separate directories under: {self.base_results_dir}/{model_name}/")
        print(f"[INFO] Check the following directories:")
        for judge_model in self.judge_models:
            judge_name = self._get_judge_name(judge_model)
            print(f"  - {self.base_results_dir}/{model_name}/{judge_name}/")