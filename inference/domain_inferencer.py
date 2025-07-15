"""
Domain Name Inferencer

This module provides inference capabilities for domain name suggestions
using the best performing model (Mistral 7B).
"""

import os
import sys
import time
import torch
import logging
from collections import Counter
from typing import Dict, Optional, List, Tuple, Set
from huggingface_hub import snapshot_download

# Add parent directory to path to import from model_evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_evaluation.model_evaluation_v1.hf_model_inferencer import HFModelInferencer

class DomainNameInferencer:
    """Domain name suggestion inference API"""
    
    def __init__(self, 
                 model_name: str = "mistralai-Mistral-7B-v0-1-finetuned-domainai", 
                 model_dir: Optional[str] = None):
        """
        Initialize the domain name inferencer.
        
        Args:
            model_name: Name of the model to use (default: mistralai-Mistral-7B-v0-1-finetuned-domainai)
            model_dir: Optional custom directory for the model, if not using the standard path
        """
        if model_dir is None:
            # Set default path to the fine-tuned model directory
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "fine_tuned_models_v4_downloaded", 
                model_name
            )
        
        # Define Hugging Face model repository
        hf_repo_id = "harshit2551/domain-name-generator-mistral7B-finetuned"
        
        # Check if model exists locally
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            logging.info(f"Model not found locally at {model_dir}. Downloading from Hugging Face Hub...")
            try:
                # Ensure the parent directory exists
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                
                # Download the model
                snapshot_download(
                    repo_id=hf_repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False  # Actual files, not symlinks
                )
                logging.info(f"Model downloaded successfully to {model_dir}")
            except Exception as e:
                logging.error(f"Error downloading model: {str(e)}")
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        logging.info(f"Loading model from {model_dir}")
        # Initialize the HFModelInferencer for model loading
        self.inferencer = HFModelInferencer(model_dir=model_dir)
        self.model_name = model_name
        self.model = self.inferencer.model
        self.tokenizer = self.inferencer.tokenizer
        self.device = self.inferencer.device
        
        # Valid TLDs for domain name filtering
        self.valid_tlds = ['.com', '.org', '.net']
    
    def generate_suggestion(self, description: str, max_new_tokens: int = 10, max_attempts: int = 20) -> Dict:
        """
        Generate domain name suggestion for a business description using direct generation,
        with multiple retry attempts to avoid edge cases.
        
        If at least one successful generation occurs in the retry attempts,
        returns the combined distinct domains from all successful attempts.
        Only marks as edge case if all attempts fail.
        
        Args:
            description: Business description
            max_new_tokens: Maximum number of new tokens to generate
            max_attempts: Maximum number of attempts to try (default: 10)
            
        Returns:
            Dictionary with raw output, parsed domains, edge case flag, and metadata
        """
        start_time = time.time()
        
        # Track successes and failures
        successful_attempts = 0
        failed_attempts = 0
        all_domains = set()
        domain_heads = set()
        best_confidence = 0.0
        
        # Track the output of the first generation for reporting
        first_raw_output = None
        
        # Try multiple times
        for attempt in range(max_attempts):
            try:
                # Prepare the prompt
                prompt = f"Suggest a domain name for this business: {description}\nDomain:"
                
                # Tokenize the prompt
                input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
                
                # Generate output with some randomness to get variety
                with torch.no_grad():  # Save memory during inference
                    gen_outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        temperature=0.7 + (attempt * 0.05),  # Slightly increase temperature each attempt
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode the generated tokens
                gen_tokens = gen_outputs.sequences
                gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                
                # Save the first raw output
                if attempt == 0:
                    first_raw_output = gen_text
                
                # Calculate confidence
                confidence = 1.0
                if hasattr(gen_outputs, 'scores') and gen_outputs.scores:
                    probs = [torch.softmax(score, dim=-1).max().item() for score in gen_outputs.scores]
                    confidence = float(sum(probs) / len(probs)) if probs else 1.0
                
                # Update best confidence score
                best_confidence = max(best_confidence, confidence)
                
                # Parse domains and check if it's an edge case
                domains, is_edge_case = self.parse_domains(gen_text)
                
                if is_edge_case:
                    failed_attempts += 1
                else:
                    successful_attempts += 1
                    
                    # Add any valid domains to our collection
                    if domains:
                        domain_list = domains.split()
                        for domain in domain_list:
                            # Extract domain head for deduplication
                            if any(domain.lower().endswith(tld) for tld in self.valid_tlds):
                                domain_head = domain.rsplit('.', 1)[0].lower()
                                
                                # Only add if we haven't seen this domain head before across any attempt
                                if domain_head not in domain_heads:
                                    all_domains.add(domain)
                                    domain_heads.add(domain_head)
                
            except Exception as e:
                logging.error(f"Error during generation attempt {attempt+1}/{max_attempts}: {e}")
                failed_attempts += 1
        
        inference_time = time.time() - start_time
        
        # Determine overall result - only block if ALL attempts failed
        is_blocked = (successful_attempts == 0)
        
        # Format the domains as a space-separated string
        distinct_domains = " ".join(all_domains)
        
        return {
            "raw_output": first_raw_output or str(Exception("All generation attempts failed")),
            "domains": distinct_domains,
            "is_edge_case": is_blocked,
            "business_description": description,
            "confidence": best_confidence,
            "inference_time": inference_time,
            "model": self.model_name,
            "attempts": {
                "total": max_attempts,
                "successful": successful_attempts,
                "failed": failed_attempts
            }
        }
    
    def parse_domains(self, raw_output: str) -> Tuple[str, bool]:
        """
        Parse domain names from raw model output and check if it's an edge case.
        
        Args:
            raw_output: The raw text output from the model
            
        Returns:
            Tuple of (parsed domains as space-separated string, is_edge_case flag)
        """
        # Check if the exact string "Domain: [EDGE_CASE]" appears in the raw output
        if "Domain: [EDGE_CASE]" in raw_output:
            return "", True
        
        # Extract text after "Domain:" if it exists
        domains_text = raw_output
        if "Domain:" in raw_output:
            domains_text = raw_output.split("Domain:", 1)[1].strip()
        
        # Split by whitespace to get individual domain candidates
        domain_candidates = domains_text.split()
        
        # Filter and clean the domains
        filtered_domains = []
        domain_heads = set()  # Keep track of domain name heads to avoid duplicates
        
        for domain in domain_candidates:
            # Clean up the domain name by removing trailing punctuation
            domain = domain.strip('.,!?;:')
            
            # Check if the domain ends with a valid TLD
            if any(domain.lower().endswith(tld) for tld in self.valid_tlds):
                # Extract domain head (part before the TLD)
                domain_head = domain.rsplit('.', 1)[0].lower()
                
                # Only add if we haven't seen this domain head before
                if domain_head not in domain_heads:
                    filtered_domains.append(domain)
                    domain_heads.add(domain_head)
        
        # Join filtered domains with spaces
        return " ".join(filtered_domains), False