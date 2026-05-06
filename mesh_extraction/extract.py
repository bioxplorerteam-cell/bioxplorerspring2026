#!/usr/bin/env python3
"""
Zero-shot and Few-shot MeSH entity extraction using LLaMA
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from parser import MeSHParser

# Load few-shot examples from file
def load_few_shot_examples():
    """Load few-shot examples from JSON file"""
    examples_file = Path(__file__).parent / "few_shot_examples.json"
    with open(examples_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format expected by prompts
    examples = []
    for article in data.get('articles', []):
        examples.append({
            "title": article.get('title', ''),
            "abstract": article.get('abstract', ''),
            "entities": article.get('mesh_terms', [])
        })
    return examples

# Load examples at module level
FEW_SHOT_EXAMPLES = load_few_shot_examples()

def create_zero_shot_prompt(abstract: str, title: str = "") -> tuple:
    """Create zero-shot prompt for entity extraction"""
    
    system_prompt = """You are a biomedical expert specialized in extracting MeSH (Medical Subject Headings) terms from scientific abstracts.

Your task: Extract ALL relevant MeSH terms that are explicitly mentioned or clearly described in the abstract.

CRITICAL INSTRUCTIONS:
- PRIORITIZE RECALL: Extract ALL entities, be comprehensive
- ALWAYS include:
  * Organisms: Humans, Animals, Mice, Rats, specific species (Chickens, etc.)
  * Demographics: Age groups (Adult, Child, Aged, Middle Aged), Male, Female
  * Diseases and conditions
  * Chemicals, drugs, and biological substances
  * Anatomical terms and body parts
  * Procedures, techniques, and methods
  * Geographic locations when relevant
- Use proper MeSH format: "Influenza A virus" not "Influenza A Virus"
- Handle word order variations: "Vaccines, Attenuated" = "Attenuated Vaccines"
- Extract terms present or clearly implied by the text
- Return ONLY a valid JSON array

Output format: ["MeSH Term 1", "MeSH Term 2", "MeSH Term 3", ...]"""

    if title:
        user_prompt = f"""Title: {title}

Abstract: {abstract}

Extract MeSH terms. Return ONLY JSON array:
["term1", "term2", "term3"]"""
    else:
        user_prompt = f"""Abstract: {abstract}

Extract MeSH terms. Return ONLY JSON array:
["term1", "term2", "term3"]"""
    
    return system_prompt, user_prompt


def create_few_shot_prompt(abstract: str, title: str = "", n_examples: int = 3) -> tuple:
    """Create few-shot prompt with examples"""
    
    system_prompt = """You are a biomedical expert specialized in extracting MeSH (Medical Subject Headings) terms from scientific abstracts.

Your task: Extract ALL relevant MeSH terms that are explicitly mentioned or clearly described in the abstract.

CRITICAL INSTRUCTIONS:
- PRIORITIZE RECALL: Extract ALL entities, be comprehensive
- ALWAYS include:
  * Organisms: Humans, Animals, Mice, Rats, specific species mentioned
  * Demographics: Age groups (Adult, Child, Aged, Middle Aged), Male, Female when relevant
  * All diseases, conditions, and symptoms
  * All chemicals, drugs, and biological substances
  * Anatomical terms and body parts
  * All procedures, techniques, and methods
  * Geographic locations when relevant
- Use proper MeSH terminology and format
- Extract terms present or clearly described in the text
- DO NOT hallucinate or invent entities not in the text
- Return ONLY a valid JSON array

Study these examples carefully:"""

    # Build examples
    examples_text = ""
    for i, example in enumerate(FEW_SHOT_EXAMPLES[:n_examples], 1):
        examples_text += f"\n\nExample {i}:\nTitle: {example['title']}\nAbstract: {example['abstract']}\nMeSH Terms: {json.dumps(example['entities'])}"
    
    if title:
        user_prompt = f"""{examples_text}

Now extract from this abstract:

Title: {title}

Abstract: {abstract}

Extract MeSH terms. Return ONLY JSON array:
["term1", "term2", "term3"]"""
    else:
        user_prompt = f"""{examples_text}

Now extract from this abstract:

Abstract: {abstract}

Extract MeSH terms. Return ONLY JSON array:
["term1", "term2", "term3"]"""
    
    return system_prompt, user_prompt


def extract_entities(model, tokenizer, abstract: str, title: str = "", 
                     mode: str = "zero-shot", n_examples: int = 3, device: str = "cuda") -> str:
    """Extract entities using specified mode"""
    
    # Create prompt
    if mode == "zero-shot":
        system_prompt, user_prompt = create_zero_shot_prompt(abstract, title)
    else:
        system_prompt, user_prompt = create_few_shot_prompt(abstract, title, n_examples)
    
    # Format for LLaMA chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    max_length = 3072 if mode == "few-shot" else 2048
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with deterministic greedy decoding (temperature=0 → do_sample=False)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.0,
            do_sample=False,  # Greedy decoding for fully deterministic output
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (not the input prompt)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def parse_response(response_text: str) -> list:
    """Parse model response to extract entity list with robust error handling"""
    # Strategy 1: Try to find JSON array
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx+1]
            # Clean up common formatting issues
            json_str = json_str.replace('\n', ' ').replace('\r', '')
            entities = json.loads(json_str)
            if isinstance(entities, list):
                # Clean and deduplicate
                cleaned = [str(e).strip() for e in entities if e]
                return list(dict.fromkeys(cleaned))  # Preserve order while deduping
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Try line-by-line extraction for malformed JSON
    try:
        if '[' in response_text and ']' in response_text:
            content = response_text[response_text.find('['):response_text.rfind(']')+1]
            # Try to extract quoted strings
            import re
            entities = re.findall(r'["\']([^"\',\[\]]+)["\']', content)
            if entities:
                cleaned = [e.strip() for e in entities if e.strip()]
                return list(dict.fromkeys(cleaned))
    except:
        pass
    
    return []


def main():
    parser = argparse.ArgumentParser(description='Extract MeSH terms using zero-shot or few-shot prompting')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to MeSH JSON file')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model path')
    parser.add_argument('--mode', type=str, choices=['zero-shot', 'few-shot'], required=True, help='Extraction mode')
    parser.add_argument('--max_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--skip', type=int, default=0, help='Articles to skip')
    parser.add_argument('--n_examples', type=int, default=3, help='Number of few-shot examples (if few-shot mode)')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file for results')
    args = parser.parse_args()
    
    # Default output file
    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"mesh_extraction_{args.mode}_{timestamp}.json"
    
    print("="*60)
    print(f"MeSH ENTITY EXTRACTION - {args.mode.upper()} MODE")
    print("="*60)
    
    # Load model
    print("\nLoading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load data
    print(f"\nLoading MeSH data...")
    mesh_parser = MeSHParser(args.mesh_file)
    articles = mesh_parser.load_data(max_articles=args.max_samples, skip=args.skip)
    
    # Extract entities
    results = []
    print(f"\nExtracting entities from {len(articles)} articles...")
    
    for i, article in enumerate(articles):
        print(f"Processing {i+1}/{len(articles)}: {article.pmid}")
        
        response = extract_entities(
            model, tokenizer, 
            article.abstract, 
            article.title,
            mode=args.mode,
            n_examples=args.n_examples,
            device=device
        )
        
        predicted = parse_response(response)
        
        result = {
            'pmid': article.pmid,
            'title': article.title,
            'abstract': article.abstract,
            'ground_truth': article.mesh_terms,
            'predicted': predicted,
            'raw_response': response
        }
        results.append(result)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Processed {len(results)} articles")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
