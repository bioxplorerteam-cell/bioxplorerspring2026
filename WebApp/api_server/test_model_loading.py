"""
Test script to diagnose LLaMA model loading issues
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def test_model_loading():
    print("=" * 80)
    print("Testing LLaMA Model Loading")
    print("=" * 80)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Check Hugging Face token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"\nHugging Face token found: {bool(hf_token)}")
    if hf_token:
        print(f"Token length: {len(hf_token)}")
    
    # Model paths
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path = "/p/realai/BioXplorer/LLama-BioXplorer/mesh_extraction/mesh_finetuned_20251212_020629"
    
    print(f"\nBase model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Adapter path exists: {os.path.exists(adapter_path)}")
    
    try:
        print("\n" + "=" * 80)
        print("Step 1: Loading tokenizer...")
        print("=" * 80)
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        print("✓ Tokenizer loaded successfully")
        
        print("\n" + "=" * 80)
        print("Step 2: Configuring 4-bit quantization...")
        print("=" * 80)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✓ Quantization config created")
        
        print("\n" + "=" * 80)
        print("Step 3: Loading base model with quantization...")
        print("=" * 80)
        print("This may take a few minutes...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())},
            token=hf_token,
            attn_implementation="eager",  # Must be set during loading
        )
        print("✓ Base model loaded successfully")
        
        print("\n" + "=" * 80)
        print("Step 4: Configuring attention...")
        print("=" * 80)
        base.config.output_attentions = True
        print("✓ Attention configured")
        
        print("\n" + "=" * 80)
        print("Step 5: Loading PEFT adapter...")
        print("=" * 80)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.config.output_attentions = True
        print("✓ PEFT adapter loaded successfully")
        
        print("\n" + "=" * 80)
        print("SUCCESS! Model loaded correctly")
        print("=" * 80)
        print("\nThe model should work in the API. Check for other issues.")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR! Model loading failed")
        print("=" * 80)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("Possible solutions:")
        print("=" * 80)
        if "401" in str(e) or "authentication" in str(e).lower() or "token" in str(e).lower():
            print("1. You need to authenticate with Hugging Face")
            print("   Run: huggingface-cli login")
            print("   Or set HF_TOKEN environment variable")
            print("2. Make sure you have accepted the Llama 3.1 license at:")
            print("   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        elif "CUDA" in str(e) or "memory" in str(e).lower():
            print("1. GPU memory issue - try reducing max_memory or clear GPU cache")
            print("2. Make sure CUDA is properly installed")
        elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print("1. Model files may not be downloaded")
            print("2. Check internet connection")
            print("3. Try downloading manually with: huggingface-cli download meta-llama/Llama-3.1-8B-Instruct")
        else:
            print("1. Check the error message above for specific issues")
            print("2. Try running with more verbose logging")
            print("3. Check dependencies are installed correctly")
        
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
