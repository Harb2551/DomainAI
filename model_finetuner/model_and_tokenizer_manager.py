import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class ModelAndTokenizerManager:
    def __init__(self, model_name, local_model_dir, lora_config=None):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.lora_config = lora_config
        self.tokenizer = None
        self.model = None
        self.tokenizer_modified = False

    def load_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer_modified = True
        return self.tokenizer

    def save_tokenizer(self):
        tokenizer_save_path = os.path.join(self.local_model_dir, f"{self.model_name.replace('/', '-')}-tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)
        return tokenizer_save_path

    def load_model(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        return self.model

    def resize_model_embeddings(self):
        if self.tokenizer_modified or len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def apply_lora(self):
        if self.lora_config is not None:
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.train()
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
        return self.model

    def save_model(self):
        model_save_path = os.path.join(self.local_model_dir, f"{self.model_name.replace('/', '-')}-model")
        self.model.save_pretrained(model_save_path)
        return model_save_path

    def reload_tokenizer(self, tokenizer_save_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path, use_fast=False, local_files_only=True)
        return self.tokenizer

    def reload_model(self, model_save_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_save_path, device_map="auto", torch_dtype="auto", local_files_only=True)
        return self.model
