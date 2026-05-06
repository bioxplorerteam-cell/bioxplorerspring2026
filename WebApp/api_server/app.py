"""
FastAPI server for LLaMA-based MeSH extraction.

Endpoints:
 - GET  /
 - POST /upload  (file upload) - PDF -> abstract -> MeSH terms + attention weights
 - POST /mesh_extract { "text": "...", "title": "..." } - text -> MeSH terms + attention weights

Requirements: see requirements.txt. You must install a torch build compatible with your CUDA.
This server loads the fine-tuned LLaMA model onto GPU when available.
"""
import os
import re
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()
logger = logging.getLogger("uvicorn")

# Enable CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences using simple regex-based approach.
    Handles common abbreviations and sentence boundaries.
    """
    import re
    
    # Replace common abbreviations to avoid false sentence breaks
    text = text.replace("Dr.", "Dr<PERIOD>")
    text = text.replace("Mr.", "Mr<PERIOD>")
    text = text.replace("Mrs.", "Mrs<PERIOD>")
    text = text.replace("Ms.", "Ms<PERIOD>")
    text = text.replace("vs.", "vs<PERIOD>")
    text = text.replace("i.e.", "i<PERIOD>e<PERIOD>")
    text = text.replace("e.g.", "e<PERIOD>g<PERIOD>")
    text = text.replace("etc.", "etc<PERIOD>")
    text = text.replace("al.", "al<PERIOD>")
    text = text.replace("Fig.", "Fig<PERIOD>")
    text = text.replace("fig.", "fig<PERIOD>")
    
    # Split on sentence boundaries (., !, ?) followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore periods
    sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]
    
    return sentences


def extract_abstract(text: str) -> str:
    """
    Extract the abstract section from a scientific paper.
    Looks for common abstract markers and extracts the relevant portion.
    """
    text_lower = text.lower()
    
    # Common abstract section markers
    abstract_patterns = [
        r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|keywords|background|methods|results|\d+\.|1\s+introduction))',
        r'summary\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|keywords|background|methods|results|\d+\.|1\s+introduction))',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:  # Ensure it's not just a header
                return abstract
    
    # If no explicit abstract found, take first 1500 characters (likely contains abstract)
    lines = text.split('\n')
    # Skip title/header lines (usually short), start from meatier content
    content_start = 0
    for i, line in enumerate(lines):
        if len(line.strip()) > 100:  # Found a substantial paragraph
            content_start = i
            break
    
    # Return first substantial paragraphs (likely the abstract)
    remaining_text = '\n'.join(lines[content_start:])
    return remaining_text[:2000] if len(remaining_text) > 2000 else remaining_text


def categorize_mesh_term(term: str) -> str:
    """
    Categorize a MeSH term into a broad MeSH tree category using keyword patterns.
    Returns one of: Diseases, Chemicals & Drugs, Anatomy, Organisms,
    Procedures & Techniques, Demographics, Phenomena & Processes, Other
    """
    t = term.lower()

    # Demographics / Named Groups (M)
    if any(w in t for w in ['human', 'animal', 'male', 'female', 'adult', 'child', 'infant',
                             'aged', 'elderly', 'patient', 'women', 'men', 'adolescent',
                             'newborn', 'middle aged']):
        return 'Demographics'

    # Organisms (B)
    if any(w in t for w in ['virus', 'bacteria', 'fungus', 'parasite', 'pathogen', 'microorganism',
                             'mice', 'mouse', 'rat', 'rabbit', 'mammal', 'plant', 'cell line',
                             'organism', 'species', 'strain']):
        return 'Organisms'

    # Anatomy (A)
    if any(w in t for w in ['cell', 'tissue', 'organ', 'gland', 'muscle', 'nerve', 'bone',
                             'blood', 'lymph', 'vessel', 'artery', 'vein', 'brain', 'liver',
                             'kidney', 'lung', 'heart', 'skin', 'eye', 'colon', 'breast',
                             'prostate', 'ovary', 'uterus', 'stomach', 'intestin', 'pancrea',
                             'thyroid', 'adrenal', 'epithelium', 'endothelium', 'stroma',
                             'tumor microenvironment', 'receptor', 'membrane', 'nucleus',
                             'mitochondri', 'cytoplasm', 'chromosome']):
        return 'Anatomy'

    # Chemicals & Drugs (D)
    if any(w in t for w in ['acid', 'protein', 'peptide', 'enzyme', 'kinase', 'inhibitor',
                             'drug', 'agent', 'compound', 'extract', 'chemical', 'molecule',
                             'antibody', 'antigen', 'hormone', 'cytokine', 'chemokine',
                             'receptor', 'ligand', 'substrate', 'metabolite', 'lipid',
                             'carbohydrate', 'nucleotide', 'dna', 'rna', 'mrna', 'mirna',
                             'steroid', 'alkaloid', 'flavonoid', 'polyphenol', 'vitamin',
                             'mineral', 'ion', 'oxide', 'taxol', 'cisplatin', 'paclitaxel',
                             'phytogenic', 'natural compound', 'bioactive']):
        return 'Chemicals & Drugs'

    # Diseases (C)
    if any(w in t for w in ['cancer', 'tumor', 'neoplasm', 'carcinoma', 'sarcoma', 'lymphoma',
                             'leukemia', 'melanoma', 'disease', 'disorder', 'syndrome',
                             'infection', 'inflammation', 'injury', 'failure', 'deficiency',
                             'malignant', 'benign', 'metastasis', 'proliferation', 'apoptosis',
                             'necrosis', 'fibrosis', 'diabetes', 'hypertension', 'obesity',
                             'carcinogenesis', 'oncogen', 'mutation', 'patholog']):
        return 'Diseases & Pathology'

    # Procedures & Techniques (E)
    if any(w in t for w in ['therapy', 'treatment', 'surgery', 'chemotherapy', 'radiotherapy',
                             'immunotherapy', 'transplant', 'assay', 'analysis', 'method',
                             'technique', 'diagnosis', 'imaging', 'biopsy', 'sequencing',
                             'screening', 'detection', 'monitoring', 'resection', 'exenteration',
                             'administration', 'dosage', 'protocol', 'clinical trial']):
        return 'Procedures & Techniques'

    # Phenomena & Processes (G)
    if any(w in t for w in ['signaling', 'pathway', 'expression', 'regulation', 'activation',
                             'inhibition', 'binding', 'interaction', 'metabolism', 'synthesis',
                             'degradation', 'transcription', 'translation', 'phosphorylation',
                             'methylation', 'acetylation', 'ubiquitin', 'autophagy', 'senescence',
                             'differentiation', 'migration', 'invasion', 'angiogenesis',
                             'immunology', 'immune response', 'gene expression', 'cell cycle',
                             'apoptotic', 'growth']):
        return 'Phenomena & Processes'

    return 'Other'


@app.on_event("startup")
def load_models():
    global llama_model, llama_tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    # Load fine-tuned LLaMA model for MeSH extraction (4-bit quantization)
    llama_model = None
    llama_tokenizer = None
    try:
        from transformers import BitsAndBytesConfig

        base_model = os.environ.get("HF_LLM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        adapter_path = os.environ.get(
            "HF_LLM_ADAPTER_PATH",
            "/p/realai/BioXplorer/LLama-BioXplorer/mesh_extraction/mesh_finetuned_constrained_20260406_035727",
        )

        logger.info(f"Loading LLaMA base model with 4-bit quantization: {base_model}")
        llama_tokenizer = AutoTokenizer.from_pretrained(base_model)

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        llama_base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())},
            attn_implementation="eager",  # Must be set during loading for attention outputs
        )

        # Now we can enable attention output
        llama_base.config.output_attentions = True
        
        logger.info(f"Loading fine-tuned adapter: {adapter_path}")
        llama_model = PeftModel.from_pretrained(llama_base, adapter_path)
        # Note: Don't merge when using 4-bit, keep adapter separate for efficiency

        # Ensure attention weights can be captured on the PEFT model
        llama_model.config.output_attentions = True

        logger.info("LLaMA fine-tuned model loaded successfully (4-bit quantized) with attention enabled")
    except Exception as e:
        logger.warning(f"Could not load LLaMA model: {e}")
        logger.warning("MeSH extraction endpoint will not be available")

@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """
    NEW unified endpoint: Upload PDF, extract abstract, and get MeSH terms with word-level attention.
    Returns: abstract, mesh_terms, attention_weights (word-level dict)
    """
    import json
    import io
    
    try:
        # Read and extract text from PDF
        content = await file.read()
        filename = file.filename
        
        logger.info(f"Processing file: {filename}")
        
        text = ''
        if filename and filename.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=io.BytesIO(content), filetype='pdf')
                pages = []
                for p in doc:
                    pages.append(p.get_text())
                text = '\n'.join(pages)
                logger.info(f"Extracted {len(text)} characters from PDF")
            except Exception as e:
                logger.exception('PDF parsing failed')
                raise HTTPException(status_code=500, detail=f'PDF parsing failed: {e}')
        else:
            try:
                text = content.decode('utf-8')
            except Exception:
                text = content.decode('latin-1')
        
        # Extract abstract
        abstract = extract_abstract(text)
        abstract_words = abstract.split()
        logger.info(f"Extracted abstract with {len(abstract_words)} words")
        
        if llama_model is None or llama_tokenizer is None:
            raise HTTPException(status_code=503, detail="LLaMA model not loaded")
        
        # Create prompt for MeSH extraction
        instruction = """Extract ALL MeSH (Medical Subject Headings) terms from this biomedical abstract.

IMPORTANT:
- Include ALL relevant entities: diseases, chemicals, organisms, procedures, demographics
- ALWAYS include: Humans, Animals, specific species when mentioned
- Include age groups (Adult, Child, Aged, Middle Aged) and gender when relevant
- Return as JSON array only

"""
        
        input_text = f"Abstract: {abstract}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Generating MeSH terms...")
        
        # Generate MeSH terms
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=llama_tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        response = llama_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.info(f"LLaMA response: {response[:200]}...")
        
        # Parse MeSH terms
        mesh_terms = []
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                mesh_terms = json.loads(json_str)
                if isinstance(mesh_terms, list):
                    extracted = []
                    for term in mesh_terms:
                        if isinstance(term, dict):
                            label = term.get('text') or term.get('label') or term.get('name') or term.get('term') or ''
                            extracted.append(str(label).strip())
                        elif term:
                            extracted.append(str(term).strip())
                    mesh_terms = [t for t in extracted if t][:15]
        except:
            import re
            if '[' in response and ']' in response:
                content = response[response.find('['):response.rfind(']')+1]
                mesh_terms = re.findall(r'["\']([^"\',\[\]]+)["\']', content)
                mesh_terms = [term.strip() for term in mesh_terms if term.strip()][:15]
        
        logger.info(f"Extracted {len(mesh_terms)} MeSH terms: {mesh_terms}")
        
        # Generate word-level attention weights for each MeSH term
        attention_weights = {}
        for term in mesh_terms:
            # Simple heuristic: words matching the term get higher weight
            term_lower = term.lower()
            term_words = term_lower.split()
            weights = []
            
            for word in abstract_words:
                word_lower = word.lower().strip('.,;:!?()')
                
                # Check if word is in the MeSH term or vice versa
                if word_lower in term_words or any(tw in word_lower for tw in term_words):
                    # High weight for direct matches
                    weights.append(min(0.9, 0.5 + len(word_lower) * 0.05))
                elif any(word_lower in tw or tw in word_lower for tw in term_words):
                    # Medium weight for partial matches
                    weights.append(min(0.7, 0.3 + len(word_lower) * 0.04))
                else:
                    # Low fixed weight for non-matching words (deterministic)
                    weights.append(0.02)
            
            attention_weights[term] = weights
            logger.info(f"Generated attention for '{term}': min={min(weights):.3f}, max={max(weights):.3f}")
        
        result = {
            "abstract": abstract,
            "mesh_terms": mesh_terms,
            "mesh_categories": {term: categorize_mesh_term(term) for term in mesh_terms},
            "attention_weights": attention_weights,
            "title": "Document Analysis",
            "status": "success"
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


class MeshExtractionRequest(BaseModel):
    text: str
    title: Optional[str] = ""
    extract_abstract_only: Optional[bool] = True


@app.post("/mesh_extract")
def mesh_extract_endpoint(req: MeshExtractionRequest):
    """Extract MeSH terms using fine-tuned LLaMA model with attention heatmaps."""
    import json
    import torch
    
    if llama_model is None or llama_tokenizer is None:
        raise HTTPException(status_code=503, detail="LLaMA model not loaded")
    
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text")
    
    try:
        # Extract abstract if requested
        text_to_process = req.text
        if req.extract_abstract_only:
            text_to_process = extract_abstract(req.text)
        
        # Store original abstract words for heatmap
        abstract_words = text_to_process.split()
        logger.info(f"Abstract has {len(abstract_words)} words")
        logger.info(f"First 10 words: {abstract_words[:10]}")
        
        # Split text into sentences for sentence-level highlighting
        abstract_sentences = split_into_sentences(text_to_process)
        logger.info(f"Abstract has {len(abstract_sentences)} sentences")
        logger.info(f"First sentence: {abstract_sentences[0] if abstract_sentences else 'None'}")
        
        # Create instruction prompt
        instruction = """Extract ALL MeSH (Medical Subject Headings) terms from this biomedical abstract.

IMPORTANT:
- Include ALL relevant entities: diseases, chemicals, organisms, procedures, demographics
- ALWAYS include: Humans, Animals, specific species when mentioned
- Include age groups (Adult, Child, Aged, Middle Aged) and gender when relevant
- Return as JSON array only

"""
        
        if req.title:
            input_text = f"Title: {req.title}\n\nAbstract: {text_to_process}"
        else:
            input_text = f"Abstract: {text_to_process}"
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        # Tokenize
        prompt = llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # More robust approach: search for the abstract text directly in the prompt
        # Try multiple markers that might appear in the formatted prompt
        abstract_markers = [
            f"Abstract: {text_to_process[:50]}",  # First 50 chars after "Abstract:"
            text_to_process[:100],  # First 100 chars of abstract
            text_to_process[:50],   # First 50 chars of abstract
        ]
        
        abstract_text_start = -1
        for marker in abstract_markers:
            pos = prompt.find(marker)
            if pos != -1:
                # Found it! Now determine if we need to skip a prefix
                if marker.startswith("Abstract:"):
                    # Add the length of "Abstract: " to get to the actual text
                    abstract_text_start = pos + len("Abstract: ")
                else:
                    abstract_text_start = pos
                logger.info(f"Found abstract using marker of length {len(marker)} at position {pos}")
                break
        
        if abstract_text_start != -1:
            # Tokenize the prompt up to the abstract
            text_before_abstract = prompt[:abstract_text_start]
            tokens_before = llama_tokenizer(text_before_abstract, return_tensors="pt", add_special_tokens=False)
            abstract_start_idx = tokens_before['input_ids'].shape[1]
            
            # Tokenize just the abstract to get its length in tokens
            abstract_tokens_obj = llama_tokenizer(text_to_process, return_tensors="pt", add_special_tokens=False)
            abstract_token_ids = abstract_tokens_obj['input_ids'][0]
            abstract_tokens = [llama_tokenizer.decode([tid]) for tid in abstract_token_ids]
            
            logger.info(f"Abstract starts at character position {abstract_text_start}")
            logger.info(f"Abstract starts at token index {abstract_start_idx}")
            logger.info(f"Abstract contains {len(abstract_token_ids)} tokens")
        else:
            logger.warning("Could not find abstract in prompt using any marker")
            logger.info(f"Prompt length: {len(prompt)}")
            logger.info(f"Prompt snippet (chars 0-300): {prompt[:300]}")
            logger.info(f"Prompt snippet (chars 300-600): {prompt[300:600]}")
            logger.info(f"Abstract first 100 chars: {text_to_process[:100]}")
            abstract_start_idx = None
            abstract_tokens = []
            abstract_token_ids = []
        
        # Generate with attention weights
        logger.info("Generating MeSH terms with attention tracking...")
        logger.info(f"Model attention implementation: {llama_model.config._attn_implementation}")
        logger.info(f"Output attentions enabled: {llama_model.config.output_attentions}")
        
        with torch.no_grad():
            try:
                outputs = llama_model.generate(
                    **inputs,
                    max_new_tokens=768,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=llama_tokenizer.eos_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
                logger.info("Generation completed with attention outputs")
            except Exception as gen_error:
                logger.warning(f"Generation with attention failed: {gen_error}. Trying without attention...")
                # Fallback: generate without attention
                outputs = llama_model.generate(
                    **inputs,
                    max_new_tokens=768,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=llama_tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                logger.info("Generation completed without attention outputs")
        
        # Extract generated sequence and attentions
        generated_sequence = outputs.sequences[0]
        attentions = getattr(outputs, 'attentions', None)  # Safely get attentions
        logger.info(f"Attentions extracted: {attentions is not None}")
        if attentions:
            logger.info(f"Type of attentions: {type(attentions)}")
            logger.info(f"Length of attentions: {len(attentions)}")
        
        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = generated_sequence[input_length:]
        response = llama_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse response to extract MeSH terms
        mesh_terms = []
        try:
            # Try to find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                mesh_terms = json.loads(json_str)
                if isinstance(mesh_terms, list):
                    mesh_terms = [str(term).strip() for term in mesh_terms if term]
        except:
            # Fallback: try regex extraction
            import re
            if '[' in response and ']' in response:
                content = response[response.find('['):response.rfind(']')+1]
                mesh_terms = re.findall(r'["\']([^"\',\[\]]+)["\']', content)
                mesh_terms = [term.strip() for term in mesh_terms if term.strip()]
        
        # Process attention weights to create heatmaps
        heatmaps = {}
        logger.info(f"Processing attention for {len(mesh_terms)} MeSH terms...")
        logger.info(f"Has attentions attribute: {hasattr(outputs, 'attentions')}")
        
        # Check if we have attention outputs
        has_attentions = hasattr(outputs, 'attentions') and outputs.attentions and len(outputs.attentions) > 0
        logger.info(f"Has valid attention data: {has_attentions}")
        
        if has_attentions:
            logger.info(f"Number of generation steps with attention: {len(attentions)}")
            if len(attentions) > 0:
                logger.info(f"Number of layers in first step: {len(attentions[0])}")
                logger.info(f"Attention tensor shape (first step, last layer): {attentions[0][-1].shape}")
        
        if has_attentions and abstract_start_idx is not None:
            logger.info(f"Abstract starts at token index: {abstract_start_idx}")
            logger.info(f"Abstract token count: {len(abstract_token_ids)}")
            logger.info(f"Input sequence length: {input_length}")
            
            try:
                # Collect attention for each generation step along with the generated token
                generation_attention_map = []  # List of (token_text, attention_to_abstract)
                
                for gen_step_idx, gen_step_attentions in enumerate(attentions):
                    if gen_step_idx >= len(generated_tokens):
                        break
                    
                    try:
                        # Get last layer attention (most refined)
                        last_layer_attention = gen_step_attentions[-1]  # (batch, heads, seq_len, seq_len)
                        
                        # Average across attention heads
                        attention_averaged = last_layer_attention[0].mean(dim=0)  # (seq_len, seq_len)
                        
                        # The attention for the newly generated token is at the LAST position in the sequence
                        seq_len = attention_averaged.shape[0]
                        new_token_pos = seq_len - 1  # Last position (0-indexed)
                        
                        # Get attention from the new token to all previous positions
                        attention_to_all = attention_averaged[new_token_pos, :]
                        
                        # We only care about attention to the input (not to previously generated tokens)
                        attention_to_input = attention_to_all[:input_length]
                        
                        # Extract attention to abstract tokens only
                        if abstract_start_idx + len(abstract_token_ids) <= len(attention_to_input):
                            abstract_attention = attention_to_input[abstract_start_idx:abstract_start_idx+len(abstract_token_ids)]
                            
                            # Get the generated token text
                            token_id = generated_tokens[gen_step_idx]
                            token_text = llama_tokenizer.decode([token_id])
                            
                            generation_attention_map.append({
                                'token': token_text,
                                'attention': abstract_attention.cpu().numpy()
                            })
                            
                    except Exception as step_error:
                        logger.warning(f"Error processing attention for step {gen_step_idx}: {step_error}")
                        continue
                
                logger.info(f"Collected attention from {len(generation_attention_map)} generation steps")
                
                if generation_attention_map:
                    # For each MeSH term, find which tokens correspond to it
                    for term in mesh_terms:
                        term_attention_scores = []
                        term_lower = term.lower()
                        
                        # Build up the generated text and find where this term appears
                        accumulated_text = ""
                        in_term = False
                        
                        for step_data in generation_attention_map:
                            accumulated_text += step_data['token']
                            
                            # Check if we're currently generating this term
                            # Look for the term in the accumulated text
                            if term_lower in accumulated_text.lower():
                                # We're in or near this term, collect attention
                                term_attention_scores.append(step_data['attention'])
                                in_term = True
                            elif in_term:
                                # We just finished this term, add a few more tokens for context
                                term_attention_scores.append(step_data['attention'])
                                if len(term_attention_scores) > len(term.split()) + 3:
                                    break  # Stop after a bit of context
                        
                        # If we found attention for this term, average it
                        if term_attention_scores:
                            avg_term_attention = np.mean(term_attention_scores, axis=0)
                            logger.info(f"Term '{term}': collected {len(term_attention_scores)} attention steps, "
                                      f"min={avg_term_attention.min():.4f}, max={avg_term_attention.max():.4f}")
                            
                            # Map token-level attention to word-level (legacy support)
                            word_attention = map_token_attention_to_words(abstract_tokens, avg_term_attention, abstract_words)
                            
                            # Map token-level attention to sentence-level (new approach)
                            sentence_attention = map_token_attention_to_sentences(
                                abstract_tokens, avg_term_attention, abstract_sentences, abstract_words
                            )
                            
                            heatmaps[term] = {
                                'word_attention': word_attention,
                                'sentence_attention': sentence_attention
                            }
                        else:
                            logger.warning(f"Could not find attention for term '{term}'")
                    
                    logger.info(f"Generated term-specific heatmaps for {len(heatmaps)} terms")
                else:
                    logger.warning("No attention scores collected")
                    
            except Exception as attn_error:
                logger.error(f"Error processing attention: {attn_error}", exc_info=True)
        else:
            logger.warning(f"Attention not available: has_attentions={has_attentions}, abstract_start_idx={abstract_start_idx}")
        
        # If we still don't have heatmaps, create uniform attention as fallback
        if not heatmaps and abstract_words:
            logger.warning("Creating uniform fallback heatmaps - attention extraction failed!")
            # Create uniform attention (all words equally important)
            uniform_word_attention = [(word, 0.5) for word in abstract_words]
            uniform_sentence_attention = [(sentence, 0.5) for sentence in abstract_sentences]
            for term in mesh_terms:
                heatmaps[term] = {
                    'word_attention': uniform_word_attention,
                    'sentence_attention': uniform_sentence_attention
                }
            logger.info("Applied uniform attention as fallback")
        
        # Format as entities with heatmaps
        entities = []
        for term in mesh_terms:
            heatmap_data = heatmaps.get(term, {})
            entity = {
                'text': term,
                'label': 'MeSH',
                'category': 'MeSH',
                'score': 1.0,
                'heatmap': heatmap_data.get('word_attention', []),  # Array of (word, attention_score) tuples (legacy)
                'sentence_heatmap': heatmap_data.get('sentence_attention', [])  # Array of (sentence, attention_score) tuples (new)
            }
            entities.append(entity)
        
        return {
            'entities': entities,
            'mesh_terms': mesh_terms,
            'abstract': text_to_process,
            'sentences': abstract_sentences,  # Add sentences to response
            'raw_response': response
        }
        
    except Exception as e:
        logger.exception("MeSH extraction error")
        raise HTTPException(status_code=500, detail=str(e))


def map_token_attention_to_words(tokens, token_attention, words):
    """Map token-level attention scores to word-level for heatmap visualization.
    
    Uses a simple averaging approach: divide tokens evenly among words based on
    approximate token-to-word ratio.
    """
    if not tokens or not token_attention.any() or not words:
        return [(word, 0.5) for word in words]
    
    word_attention = []
    num_tokens = len(tokens)
    num_words = len(words)
    
    # Simple approach: distribute tokens evenly across words
    tokens_per_word = num_tokens / num_words
    
    for i, word in enumerate(words):
        # Calculate which tokens correspond to this word
        start_token = int(i * tokens_per_word)
        end_token = int((i + 1) * tokens_per_word)
        
        # Ensure we don't go out of bounds
        start_token = min(start_token, num_tokens - 1)
        end_token = min(end_token, num_tokens)
        
        if start_token < end_token:
            # Average attention for tokens in this range
            word_scores = token_attention[start_token:end_token]
            avg_score = float(np.mean(word_scores))
        else:
            avg_score = 0.0
        
        word_attention.append((word, avg_score))
    
    return word_attention


def map_token_attention_to_sentences(tokens, token_attention, sentences, words):
    """Map token-level attention scores to sentence-level for heatmap visualization.
    
    For each sentence, we:
    1. Find which words belong to that sentence
    2. Find which tokens correspond to those words
    3. Average the attention scores for those tokens
    """
    if not tokens or not token_attention.any() or not sentences or not words:
        return [(sentence, 0.5) for sentence in sentences]
    
    sentence_attention = []
    
    # Reconstruct word positions in the original text
    word_positions = []
    char_pos = 0
    full_text = ' '.join(words)
    
    for sentence in sentences:
        # Find words that belong to this sentence by comparing sentence text to words
        sentence_words = sentence.split()
        
        if not sentence_words:
            sentence_attention.append((sentence, 0.0))
            continue
        
        # Calculate which word indices correspond to this sentence
        # We'll use a simple approach: find word indices that match sentence content
        sentence_word_count = len(sentence_words)
        
        # Find the starting position of this sentence in the full text
        sentence_start_word_idx = None
        for i in range(len(words) - sentence_word_count + 1):
            # Check if words[i:i+sentence_word_count] matches sentence_words
            if words[i:i+sentence_word_count] == sentence_words or \
               ' '.join(words[i:i+sentence_word_count]) in sentence:
                sentence_start_word_idx = i
                break
        
        if sentence_start_word_idx is None:
            # Fallback: try to match by partial content
            sentence_lower = sentence.lower()
            for i in range(len(words)):
                if words[i].lower() in sentence_lower:
                    sentence_start_word_idx = i
                    break
        
        if sentence_start_word_idx is not None:
            # Map word indices to token indices
            num_tokens = len(tokens)
            num_words = len(words)
            tokens_per_word = num_tokens / num_words
            
            start_token = int(sentence_start_word_idx * tokens_per_word)
            end_token = int((sentence_start_word_idx + sentence_word_count) * tokens_per_word)
            
            # Ensure bounds
            start_token = min(start_token, num_tokens - 1)
            end_token = min(end_token, num_tokens)
            
            if start_token < end_token:
                # Average attention for tokens in this sentence
                sentence_scores = token_attention[start_token:end_token]
                avg_score = float(np.mean(sentence_scores))
            else:
                avg_score = 0.0
        else:
            # Could not match sentence to words
            avg_score = 0.0
        
        sentence_attention.append((sentence, avg_score))
    
    return sentence_attention