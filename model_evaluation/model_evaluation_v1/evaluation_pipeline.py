import os
import json
from typing import List, Dict
# If you see an import error for pydantic, run: pip install pydantic
from pydantic import BaseModel, Field, ValidationError
from .testset_loader import TestsetLoader
from .hf_model_inferencer import HFModelInferencer
from .bedrock_llm_judge import BedrockLLMJudge
import time
import concurrent.futures
import math

class DomainGenerationResult(BaseModel):
    description: str
    domain: str
    label: str

class LLMScoreResult(BaseModel):
    description: str
    domain: str
    scores: Dict[str, float] = Field(default_factory=dict)

def load_test_data(testset_path: str):
    print(f"[INFO] Loading test set from {testset_path}")
    loader = TestsetLoader(testset_path)
    df = loader.load_dataframe()
    print(f"[INFO] Loaded {len(df)} test cases.")
    descriptions = df['business_description'].tolist() if 'business_description' in df.columns else df.iloc[:, 0].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else ["" for _ in descriptions]
    return descriptions, labels

def generate_domains(inferencer, descriptions, max_new_tokens):
    print(f"[INFO] Generating domains with HF model...")
    hf_start_time = time.time()
    hf_results = inferencer.generate(descriptions, max_new_tokens=max_new_tokens)
    hf_total_time = time.time() - hf_start_time
    print(f"[INFO] Domain generation complete. Total inference time: {hf_total_time:.2f} seconds. Avg per sample: {hf_total_time/len(descriptions):.3f} seconds.")
    # Show a sample of generated domains
    print("[INFO] Sample generated domains:")
    for i, hf_result in enumerate(hf_results[:10]):
        print(f"  [{i+1}] {hf_result['domain']} (confidence: {hf_result['confidence']:.3f}, status: {hf_result['status']})")
    # Save all generated domains to a txt file for later inspection
    os.makedirs("./results/evaluation", exist_ok=True)
    with open("./results/evaluation/hf_model_outputs.json", "w") as f:
        json.dump(hf_results, f, indent=2)
    return hf_results, hf_total_time

def score_domains(judge, descriptions, hf_results, labels, predicted_labels, criteria, hf_total_time):
    results = []
    print(f"[INFO] Scoring all generated domains with LLM judge using 8 threads...")
    n_threads = 8

    def score_one(args):
        idx, desc, hf_result, label, predicted_label = args
        domain = hf_result["domain"]
        confidence = hf_result["confidence"]
        status = hf_result["status"]
        try:
            gen_result = DomainGenerationResult(description=desc, domain=domain, label=label)
        except ValidationError as e:
            print(f"[ERROR] DomainGenerationResult validation error at idx {idx}: {e}")
            return None
        llm_start_time = time.time()
        try:
            scores = judge.score(desc, gen_result.domain, criteria, label=label, predicted_label=predicted_label)
        except Exception as e:
            print(f"[ERROR] LLM judge scoring failed at idx {idx}: {e}")
            scores = {}
        llm_inference_time = time.time() - llm_start_time
        try:
            score_result = LLMScoreResult(description=desc, domain=gen_result.domain, scores=scores)
        except ValidationError as e:
            print(f"[ERROR] LLMScoreResult validation error at idx {idx}: {e}")
            return None
        result_dict = score_result.dict()
        result_dict['is_edge_case'] = bool(label and label != "normal")
        result_dict['ground_truth_label'] = label
        result_dict['predicted_label'] = predicted_label
        result_dict['scores'] = scores
        result_dict['hf_confidence'] = confidence
        result_dict['hf_status'] = status
        result_dict['hf_inference_time'] = hf_total_time / len(descriptions)  # avg per sample
        result_dict['llm_inference_time'] = llm_inference_time
        return result_dict

    args_list = list(enumerate(zip(descriptions, hf_results, labels, predicted_labels)))
    # Unpack for thread pool
    args_list = [(idx, desc, hf_result, label, predicted_label) for idx, (desc, hf_result, label, predicted_label) in args_list]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for idx, result in enumerate(executor.map(score_one, args_list)):
            if result is not None:
                results.append(result)
            if idx % 10 == 0:
                print(f"[INFO] Processed {idx+1}/{len(descriptions)} cases...")
    return results

def evaluate_model(
    model_dir: str,
    testset_path: str,
    results_dir: str,
    criteria: List[str],
    bedrock_model_id: str,
    region: str = 'us-east-1',
    max_new_tokens: int = 32,
    tokenizer_dir: str = None
):
    print(f"[INFO] Starting evaluation pipeline...")
    os.makedirs(results_dir, exist_ok=True)
    descriptions, labels = load_test_data(testset_path)
    print(f"[INFO] Loading HuggingFace model from {model_dir}")
    inferencer = HFModelInferencer(model_dir, tokenizer_dir=tokenizer_dir)
    hf_results, hf_total_time = generate_domains(inferencer, descriptions, max_new_tokens)
    predicted_labels = []
    ground_truth_labels = []
    for label, hf_result in zip(labels, hf_results):
        domain = hf_result["domain"].strip().lower() if hf_result["domain"] else ""
        # Predicted label: 'edge' if domain is an edge case marker, else 'normal'
        if domain.startswith("[edge") or domain.startswith("[edg") or domain.startswith("[ed"):
            predicted_labels.append("edge")
            status = "blocked"
        else:
            predicted_labels.append("normal")
            status = "success"
        # Ground truth label: 'edge' if label is not 'normal', else 'normal'
        if label and label != "normal":
            ground_truth_labels.append("edge")
        else:
            ground_truth_labels.append("normal")
        hf_result["status"] = status
    print(f"[INFO] Initializing Bedrock LLM Judge with model {bedrock_model_id}")
    judge = BedrockLLMJudge(bedrock_model_id, region=region)
    results = score_domains(judge, descriptions, hf_results, ground_truth_labels, predicted_labels, criteria, hf_total_time)
    model_name = os.path.basename(model_dir)
    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'evaluation_results.json')
    print(f"[INFO] Saving results to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved evaluation results for {model_name} to {out_path}")

    # Compute and save summary metrics for model comparison
    print(f"[INFO] Computing summary metrics for {model_name}")
    # Extract scores and times
    n = len(results)
    sum_scores = {c: 0.0 for c in criteria}
    count_scores = {c: 0 for c in criteria}
    total_hf_time = 0.0
    total_llm_time = 0.0
    for r in results:
        for c in criteria:
            v = r['scores'].get(c)
            if isinstance(v, (int, float)):
                sum_scores[c] += v
                count_scores[c] += 1
        total_hf_time += r.get('hf_inference_time', 0.0)
        total_llm_time += r.get('llm_inference_time', 0.0)
    avg_scores = {c: (sum_scores[c] / count_scores[c] if count_scores[c] else None) for c in criteria}
    avg_hf_time = total_hf_time / n if n else None
    avg_llm_time = total_llm_time / n if n else None
    total_eval_time = hf_total_time + total_llm_time
    metrics = {
        'model_name': model_name,
        'num_cases': n,
        'criteria': criteria,
        'average_scores': avg_scores,
        'average_hf_inference_time': avg_hf_time,
        'average_llm_inference_time': avg_llm_time,
        'total_hf_inference_time': hf_total_time,
        'total_llm_inference_time': total_llm_time,
        'total_evaluation_time': total_eval_time
    }
    metrics_path = os.path.join(out_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved summary metrics for {model_name} to {metrics_path}")

def main():
    # Set your arguments here directly
    model_dir = "./fine_tuned_models/Qwen-Qwen2-7B-Instruct-finetuned-domainai"
    tokenizer_dir = "./local_models/.-local_models-Qwen-Qwen2-7B-Instruct-tokenizer"
    testset_path = "./synthetic_domain_dataset_test.csv"
    results_dir = "./results/evaluation"
    criteria = ["relevance", "appropriateness", "creativity"]
    bedrock_model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    region = "us-east-1"
    max_new_tokens = 10

    evaluate_model(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        testset_path=testset_path,
        results_dir=results_dir,
        criteria=criteria,
        bedrock_model_id=bedrock_model_id,
        region=region,
        max_new_tokens=max_new_tokens
    )

if __name__ == "__main__":
    main()
