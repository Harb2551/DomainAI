class DomainDatasetPreprocessor:
    def __init__(self, tokenizer, max_length=256):
        # Store the tokenizer and max sequence length for use in preprocessing
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example):
        # Extract prompt and label from the example
        prompt = example['text']
        label = example['labels']
        # Concatenate prompt and label for language modeling
        full = prompt + " " + label
        # Tokenize the combined string, pad/truncate to max_length
        tokenized = self.tokenizer(full, truncation=True, max_length=self.max_length, padding="max_length")
        # Find the length of the prompt after tokenization
        prompt_tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
        prompt_length = sum([1 for id in prompt_tokenized['input_ids'] if id != self.tokenizer.pad_token_id])
        # Set 'labels' to -100 for prompt tokens, and real ids for label tokens
        labels = [-100] * prompt_length + tokenized['input_ids'][prompt_length:]
        # Pad labels to max_length
        labels = labels[:self.max_length] + [-100] * (self.max_length - len(labels))
        tokenized["labels"] = labels
        return tokenized
