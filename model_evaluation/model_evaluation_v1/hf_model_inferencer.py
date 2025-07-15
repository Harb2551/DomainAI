from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import torch.nn.functional as F
from typing import List, Dict
import logging
import os
import os.path
from huggingface_hub import login

class HFModelInferencer:
    def __init__(self, model_dir: str, tokenizer_dir: str = None, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if tokenizer_dir is None:
            tokenizer_dir = model_dir
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load PEFT configuration to get base model path
            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model_path = peft_config.base_model_name_or_path
            
            # Check if the base model path is a local path that doesn't exist
            if base_model_path.startswith('./local_models/') and not os.path.exists(base_model_path):
                # Extract the model name from the path
                model_name = base_model_path.split('/')[-1]
                
                # Check for HF token in environment
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    logging.info("Hugging Face token found, attempting to login")
                    login(token=hf_token)
                
                # Initialize hf_model_id
                hf_model_id = None
                
                # Try an open model instead if no token
                if not hf_token:
                    # Use an open-access model instead
                    hf_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    logging.info(f"No HF token found and local model path {base_model_path} not found. Using open-access model {hf_model_id} instead")
                else:
                    # Use the original gated model
                    hf_model_id = "mistralai/Mistral-7B-v0.1"
                    logging.info(f"Local model path {base_model_path} not found, using HuggingFace model {hf_model_id} with authentication")
                
                base_model_path = hf_model_id
            
            logging.info(f"Loading PEFT model with base model: {base_model_path}")
            
            # Load base model first - will download from HF if needed
            # Check if we're using a gated model
            if "mistralai" in base_model_path and os.environ.get("HF_TOKEN"):
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    token=os.environ.get("HF_TOKEN")
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(base_model_path)
            
            # Resize embeddings to match tokenizer
            if self.model.config.vocab_size != len(self.tokenizer):
                logging.warning(f"Resizing base model embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move to device
            self.model = self.model.to(device)
            
            # Load the PEFT adapter
            self.model = PeftModel.from_pretrained(self.model, model_dir)
            self.model.eval()
            self.device = device
            
        except Exception as e:
            logging.error(f"Error loading PEFT model: {e}")
            raise

    def generate(self, descriptions: List[str], max_new_tokens: int = 32) -> List[Dict]:
        return [self._generate_single(desc, max_new_tokens) for desc in descriptions]

    def _generate_single(self, description: str, max_new_tokens: int) -> Dict:
        try:
            # Use the same prompt as used during fine-tuning
            prompt = (
                f"Suggest a domain name for this business: {description}\nDomain:"
            )
            
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            
            with torch.no_grad():  # Save memory during inference
                gen_outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    temperature=0.7,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_tokens = gen_outputs.sequences
            gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            
            # If 'Domain:' is present, take the first word after it; otherwise, take the first word of the output
            if 'Domain:' in gen_text:
                after_domain = gen_text.split('Domain:', 1)[1].strip()
                domain_word = after_domain.split()[0] if after_domain else ''
            else:
                domain_word = gen_text.split()[0] if gen_text.strip() else ''
            
            # Clean up domain (remove common suffixes if they appear broken)
            domain = domain_word.rstrip('.,!?;:') if domain_word else ''
            
            # Confidence: mean of max softmax probability at each step (if available)
            if hasattr(gen_outputs, 'scores') and gen_outputs.scores:
                probs = [F.softmax(score, dim=-1).max().item() for score in gen_outputs.scores]
                confidence = float(sum(probs) / len(probs)) if probs else 1.0
            else:
                confidence = 1.0
            
            # Do not set status here; let evaluation pipeline decide based on predicted label
            status = None
            
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            gen_text = str(e)
            domain = ''
            confidence = 0.0
            status = "error"
        
        return {
            "domain": domain, 
            "confidence": confidence, 
            "status": status, 
            "raw_output": gen_text
        }

    def __del__(self):
        """Clean up GPU memory when the object is destroyed"""
        if hasattr(self, 'model') and self.device == 'cuda':
            del self.model
            torch.cuda.empty_cache()