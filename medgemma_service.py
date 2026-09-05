import os
import zipfile
import torch

BASE_MODEL_ID = os.getenv("MEDGEMMA_BASE_MODEL", "google/medgemma-4b-it")
ADAPTER_PATH = os.getenv("MEDGEMMA_ADAPTER_PATH", "./medgemma-abg-vent-lora")
ZIP_FALLBACK = os.getenv("MEDGEMMA_ZIP_PATH", "./medgemma-adapter-only.zip")

_model = None
_tokenizer = None
_processor = None

def ensure_adapter_extracted():
    """Ensure adapter directory exists. If missing, auto-extract from zip if present."""
    if os.path.exists(ADAPTER_PATH) and os.path.isdir(ADAPTER_PATH):
        return True
    
    if os.path.exists(ZIP_FALLBACK):
        print(f"[MedGemma] Extracting adapter from {ZIP_FALLBACK} to {ADAPTER_PATH}...")
        os.makedirs(ADAPTER_PATH, exist_ok=True)
        with zipfile.ZipFile(ZIP_FALLBACK, 'r') as zip_ref:
            zip_ref.extractall(ADAPTER_PATH)
        print("[MedGemma] Extraction completed successfully.")
        return True
    
    return False

def get_medgemma_model():
    """Lazy loader for MedGemma 4B base model + fine-tuned LoRA adapter."""
    global _model, _tokenizer, _processor
    
    if _model is not None:
        return _model, _tokenizer
    
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel
    
    ensure_adapter_extracted()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[MedGemma] Warning: CUDA not detected! Running on CPU will be slow.")
        device_map = "cpu"
        bnb_config = None
        compute_dtype = torch.float32
    else:
        print(f"[MedGemma] CUDA Device detected: {torch.cuda.get_device_name(0)}")
        device_map = "auto"
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    print(f"[MedGemma] Loading base model ({BASE_MODEL_ID})...")
    _processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    _tokenizer = _processor.tokenizer

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype
    )

    if os.path.exists(ADAPTER_PATH):
        print(f"[MedGemma] Loading fine-tuned LoRA adapter from {ADAPTER_PATH}...")
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"[MedGemma] Warning: Adapter path {ADAPTER_PATH} not found. Running base model only.")
        _model = base_model
        
    _model.eval()
    print("[MedGemma] Model loaded and ready for clinical inference.")
    return _model, _tokenizer

def run_medgemma_inference(
    prompt_text: str, 
    system_prompt: str = "You are a clinical decision-support assistant using ARDSNet principles.",
    max_tokens: int = 500
) -> str:
    """Execute inference on clinical query using fine-tuned MedGemma."""
    model, tokenizer = get_medgemma_model()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("[MedGemma] Generating clinical response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=False
        )
        
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text.strip()
