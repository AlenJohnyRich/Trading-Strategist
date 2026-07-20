"""
Deep learning utilities: language model wrapper and fine-tuning helpers.
Provides a light-weight LM wrapper using Hugging Face transformers suitable for
small-scale fine-tuning and inference. Uses distilgpt2 by default but accepts
any HF causal LM checkpoint.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict

logger = logging.getLogger('TradingAI.deep_learning')

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset
except Exception:
    torch = None


class LMWrapper:
    def __init__(self, model_name: str = 'distilgpt2', device: Optional[str] = None):
        if torch is None:
            raise RuntimeError('torch and transformers are required for LMWrapper')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate(self, prompt: str, max_length: int = 256, temperature: float = 0.8) -> str:
        tok = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        out = self.model.generate(**tok, max_new_tokens=max_length, temperature=temperature, do_sample=True)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def fine_tune(self, texts: List[str], output_dir: str = './models/lm_finetuned', epochs: int = 1, batch_size: int = 4, lr: float = 5e-5):
        """Simple fine-tuning routine using Hugging Face Trainer.
        texts: list of raw strings (prompt+response or conversational pairs concatenated)
        """
        dataset = Dataset.from_dict({'text': texts})
        def tokenize_fn(ex):
            return self.tokenizer(ex['text'], truncation=True, padding='max_length', max_length=512)
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size, learning_rate=lr, logging_steps=10, save_strategy='epoch')
        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
        trainer.train()
        trainer.save_model(output_dir)
        return output_dir


# Lightweight utility to load a fine-tuned model from path
def load_finetuned(model_dir: str) -> LMWrapper:
    if torch is None:
        raise RuntimeError('torch and transformers are required')
    w = LMWrapper(model_name=model_dir)
    return w
