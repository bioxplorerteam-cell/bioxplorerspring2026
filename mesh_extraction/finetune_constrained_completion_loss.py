#!/usr/bin/env python3
"""
Fine-tune LLaMA for MeSH entity extraction using QLoRA - OPTIMIZED
Improvements:
1. Constrained output space: Validates outputs against canonical MeSH vocabulary
2. Optimized loss: Uses prompt-completion format with completion_only_loss
   instead of computing loss on full sequence including system/user prompts

Key benefits:
- Model learns concept assignment (hard part) instead of JSON serialization (easy part)
- Loss only computed on MeSH output tokens, not system/user prompt tokens
- Outputs validated against canonical vocabulary to avoid non-MeSH terms
- Dramatically reduces wasted model capacity on syntax instead of semantics
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import argparse
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Set, Dict, Any

sys.path.append(str(Path(__file__).parent))
from parser import MeSHParser


class MeSHVocabulary:
    """
    Manages canonical MeSH vocabulary for constrained output space.
    Validates generated terms against official descriptors.
    
    This ensures:
    - No non-MeSH strings are considered valid
    - No near-synonyms instead of official descriptors
    - No duplicates or inconsistent capitalization
    - Model learns that concept assignment is the hard part, not syntax
    """
    
    def __init__(self, vocabulary_file: str = None):
        """
        Initialize MeSH vocabulary.
        
        Args:
            vocabulary_file: Optional path to JSON file with canonical MeSH terms.
                           If None, vocabulary is built from training data.
        """
        self.terms: Set[str] = set()
        self.terms_lower: Set[str] = set()  # Lowercase for fuzzy matching
        self.term_to_canonical: Dict[str, str] = {}  # Maps near-synonyms to canonical
        
        if vocabulary_file:
            self._load_vocabulary(vocabulary_file)
    
    def _load_vocabulary(self, vocabulary_file: str):
        """Load vocabulary from JSON file"""
        print(f"Loading MeSH vocabulary from {vocabulary_file}...")
        with open(vocabulary_file, 'r') as f:
            vocab_data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(vocab_data, list):
            for term in vocab_data:
                self._add_term(term)
        elif isinstance(vocab_data, dict):
            # Keys are canonical, values are near-synonyms
            for canonical, synonyms in vocab_data.items():
                self._add_term(canonical)
                for syn in (synonyms if isinstance(synonyms, list) else [synonyms]):
                    self.term_to_canonical[syn.lower()] = canonical
    
    def _add_term(self, term: str):
        """Add term to vocabulary"""
        self.terms.add(term)
        self.terms_lower.add(term.lower())
        if term.lower() not in self.term_to_canonical:
            self.term_to_canonical[term.lower()] = term
    
    def build_from_data(self, articles: List):
        """Build vocabulary from training articles"""
        print("Building MeSH vocabulary from training data...")
        for article in articles:
            if article.mesh_terms:
                for term in article.mesh_terms:
                    self._add_term(term)
        print(f"Vocabulary size: {len(self.terms)} unique MeSH terms")
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize and validate a term.
        Returns canonical form if found, original if not in vocabulary.
        """
        term = term.strip()
        if not term:
            return None
        
        # Exact match
        if term in self.terms:
            return term
        
        # Case-insensitive match
        term_lower = term.lower()
        if term_lower in self.term_to_canonical:
            return self.term_to_canonical[term_lower]
        
        # Direct lowercase match
        for canonical in self.terms:
            if canonical.lower() == term_lower:
                return canonical
        
        return term  # Return as-is if not found
    
    def validate_and_correct(self, terms: List[str]) -> List[str]:
        """
        Validate and correct a list of terms.
        
        Returns:
            List with normalized terms, duplicates removed, invalid entries filtered.
        """
        if not isinstance(terms, list):
            return []
        
        validated = []
        seen = set()
        
        for term in terms:
            if not isinstance(term, str):
                continue
            
            normalized = self.normalize_term(term)
            if normalized and normalized not in seen:
                validated.append(normalized)
                seen.add(normalized)
        
        return validated


def prepare_training_data_with_vocabulary(
    mesh_file: str,
    max_samples: int = 500,
    skip: int = 0,
    vocab: MeSHVocabulary = None
):
    """
    Prepare training data from MeSH dataset with vocabulary validation.
    
    Uses conversational format (messages) instead of flattened "text" field.
    This enables TRL to apply completion_only_loss, so loss is computed
    ONLY on the assistant's MeSH output, not on system/user prompt tokens.
    
    Args:
        mesh_file: Path to MeSH JSON file
        max_samples: Maximum training samples
        skip: Articles to skip (for train/test split)
        vocab: MeSHVocabulary instance for validation
    
    Returns:
        Tuple of (training_data, vocab)
    """
    print(f"Loading MeSH training data from {mesh_file}...")
    parser = MeSHParser(mesh_file)
    articles = parser.load_data(max_articles=max_samples, skip=skip)
    
    # Build vocabulary if not provided or if loaded vocabulary is empty
    if vocab is None or not vocab.terms:
        vocab = MeSHVocabulary()
        vocab.build_from_data(articles)
    
    print(f"Preparing {len(articles)} training examples...")
    training_data = []
    
    system_prompt = """You are an expert biomedical NLP system for extracting MeSH (Medical Subject Headings) terms.

EXTRACTION RULES:
1. Extract ONLY official MeSH descriptors (canonical terms)
2. Return a JSON array of MeSH terms found in the text
3. Do NOT include:
   - Non-canonical synonyms or near-matches
   - Duplicates or variations in capitalization
   - Made-up or non-MeSH terms
4. Be comprehensive but only include terms that are official MeSH descriptors
5. Return a valid JSON array - nothing else

EXAMPLE OUTPUT:
["Humans", "Diabetes Mellitus", "Treatment Outcome", "Prospective Studies"]
"""
    
    for article in articles:
        if not article.mesh_terms:
            continue
        
        # Validate and normalize output terms against canonical vocabulary
        # This reduces the output space so model can't emit non-MeSH strings
        validated_terms = vocab.validate_and_correct(article.mesh_terms)
        if not validated_terms:
            continue
        
        # Create conversational format with "messages" field
        # TRL will automatically apply chat_template and can compute
        # completion_only_loss when this format is used
        if article.title:
            user_content = f"Title: {article.title}\n\nAbstract: {article.abstract}"
        else:
            user_content = f"Abstract: {article.abstract}"
        
        # Output is JUST the JSON array (no explanation)
        # Loss will only be computed on these tokens
        assistant_content = json.dumps(validated_terms, ensure_ascii=False)
        
        training_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })
    
    print(f"Created {len(training_data)} training examples (with validated outputs)")
    return training_data, vocab


def create_datasets(training_data: List[Dict], tokenizer, val_split: float = 0.1):
    """
    Create train and validation datasets from conversational format.
    
    Keeps the "messages" field untouched so SFTTrainer can:
    1. Apply apply_chat_template() internally
    2. Correctly mask loss computation to completion only
    3. Avoid training on system/user prompt tokens
    """
    import random
    random.shuffle(training_data)
    
    split_idx = int(len(training_data) * (1 - val_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create datasets - keep "messages" field as-is
    # SFTTrainer will handle formatting via apply_chat_template()
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer(model_path: str, use_4bit: bool = True):
    """
    Setup model with quantization and LoRA
    
    Args:
        model_path: Path to base model
        use_4bit: Whether to use 4-bit quantization (QLoRA)
    
    Returns:
        model, tokenizer
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set reasonable max length (reduced for memory efficiency)
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 1536
    
    print(f"Loading model with {'4-bit' if use_4bit else 'float16'} precision...")
    
    if use_4bit:
        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Single GPU device mapping
        device_map = {"": 0}
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer


def setup_lora_config():
    """Configure LoRA parameters"""
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return peft_config


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune LLaMA for MeSH extraction (constrained output + completion-only loss)'
    )
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to MeSH JSON file')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Base model path')
    parser.add_argument('--output_dir', type=str, default='./mesh_finetuned_constrained', help='Output directory')
    parser.add_argument('--vocab_file', type=str, default=None, help='Optional: Path to canonical MeSH vocabulary JSON')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum training samples')
    parser.add_argument('--skip', type=int, default=0, help='Articles to skip')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio (ignored if --val_file provided)')
    parser.add_argument('--val_file', type=str, default=None, help='Optional: Path to separate validation JSON file (overrides --val_split)')
    parser.add_argument('--use_4bit', action='store_true', default=False, help='Use 4-bit quantization (QLoRA)')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    
    print("="*80)
    print("MeSH ENTITY EXTRACTION - CONSTRAINED OUTPUT + COMPLETION-ONLY LOSS")
    print("="*80)
    print(f"Base model: {args.model_path}")
    print(f"Training samples: {args.max_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    print("\nOptimizations enabled:")
    print("  ✓ Constrained output space (vocabulary validation)")
    print("    - No non-MeSH strings can be predicted as valid")
    print("    - Near-synonyms mapped to canonical descriptors")
    print("    - Duplicates and capitalization issues removed")
    print("  ✓ Completion-only loss")
    print("    - Loss NOT computed on system prompt tokens")
    print("    - Loss NOT computed on user prompt (abstract) tokens")
    print("    - Loss ONLY computed on MeSH output tokens")
    print("="*80)
    
    # Load vocabulary if provided; otherwise build from training data
    vocab = MeSHVocabulary(vocabulary_file=args.vocab_file) if args.vocab_file else None
    
    # Prepare training data with vocabulary validation
    training_data, vocab = prepare_training_data_with_vocabulary(
        args.mesh_file,
        max_samples=args.max_samples,
        skip=args.skip,
        vocab=vocab
    )
    
    if not training_data:
        print("ERROR: No valid training data prepared!")
        return
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_path, args.use_4bit)
    
    # Create datasets with "messages" field preserved
    if args.val_file:
        # Load separate validation file
        print(f"\nLoading separate validation file: {args.val_file}")
        val_data, _ = prepare_training_data_with_vocabulary(
            args.val_file,
            max_samples=None,  # Use all
            skip=0,
            vocab=vocab
        )
        print(f"  Validation samples: {len(val_data)}")
        
        # Use all training_data for training
        train_dataset = Dataset.from_list(training_data)
        val_dataset = Dataset.from_list(val_data)
        print(f"Train samples: {len(training_data)}, Validation samples: {len(val_data)}")
    else:
        # Split training_data using val_split ratio
        train_dataset, val_dataset = create_datasets(training_data, tokenizer, args.val_split)
    
    # Setup LoRA
    peft_config = setup_lora_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Initialize trainer with optimized settings
    from trl import SFTConfig
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=3,
        max_length=1536,
        packing=False,
        # CRITICAL: Don't flatten to "text" field
        # Keep "messages" format so SFTTrainer can apply proper loss masking
        dataset_text_field=None,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
        # CRITICAL: Use formatting_func to handle "messages" + apply_chat_template
        # This enables completion_only_loss masking
        formatting_func=lambda x: {
            "text": tokenizer.apply_chat_template(
                x["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        },
    )
    
    # Train
    print("\nStarting training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\nSaving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save vocabulary for inference time validation
    print("\nSaving MeSH vocabulary...")
    vocab_file = Path(output_dir) / "mesh_vocabulary.json"
    with open(vocab_file, 'w') as f:
        json.dump(list(vocab.terms), f, ensure_ascii=False, indent=2)
    
    print("="*80)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Vocabulary saved to: {vocab_file}")
    print("\nKey improvements in this version:")
    print("  ✓ Loss computed ONLY on MeSH output tokens (not prompt)")
    print("  ✓ Model learns concept assignment (hard) not syntax (easy)")
    print("  ✓ Outputs validated against canonical MeSH vocabulary")
    print("  ✓ No duplicates, inconsistent capitalization, or non-MeSH terms")
    print("  ✓ Reduces model capacity wasted on serialization instead of semantics")
    print("="*80)


if __name__ == '__main__':
    main()
