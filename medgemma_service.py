import os
import io
import base64
import zipfile
import torch
from PIL import Image

BASE_MODEL_ID = os.getenv("MEDGEMMA_BASE_MODEL", "google/medgemma-4b-it")
ADAPTER_PATH = os.getenv("MEDGEMMA_ADAPTER_PATH", "./medgemma-abg-vent-lora")
ZIP_FALLBACK = os.getenv("MEDGEMMA_ZIP_PATH", "./medgemma-adapter-only.zip")

_model = None
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
    """Lazy loader for MedGemma 4B multimodal model + fine-tuned LoRA adapter."""
    global _model, _processor
    
    if _model is not None:
        return _model, _processor
    
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

    print(f"[MedGemma] Loading base processor & model ({BASE_MODEL_ID})...")
    _processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

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
    print("[MedGemma] Multimodal model loaded and ready for clinical inference.")
    return _model, _processor

def _decode_base64_images(raw_images: list) -> list:
    """Safely decode a list of base64 strings into PIL RGB Images."""
    pil_images = []
    if not raw_images:
        return pil_images
        
    for idx, img_b64 in enumerate(raw_images):
        try:
            if not isinstance(img_b64, str):
                continue
            # Strip data URL prefix if present (e.g. data:image/png;base64,...)
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64.strip())
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_images.append(pil_img)
        except Exception as e:
            print(f"[MedGemma] Error decoding image index {idx}: {e}")
            
    return pil_images

def run_medgemma_inference(
    prompt_text: str, 
    system_prompt: str = "You are a clinical decision-support assistant using ARDSNet principles.",
    max_tokens: int = 500,
    images: list = None
) -> str:
    """Execute clinical multimodal inference supporting single or multiple images."""
    model, processor = get_medgemma_model()
    
    pil_images = _decode_base64_images(images or [])
    has_images = len(pil_images) > 0
    print(f"[MedGemma] Processing inference request with {len(pil_images)} image(s)...")

    # Format multimodal chat prompt
    if has_images:
        content_items = []
        for _ in pil_images:
            content_items.append({"type": "image"})
        content_items.append({"type": "text", "text": prompt_text})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_items}
        ]
        
        try:
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback format for templates that only accept simple strings or single-turn
            prompt = f"{system_prompt}\n\nClinical Query: {prompt_text}\nAnswer:"

        inputs = processor(text=prompt, images=pil_images, return_tensors="pt").to(model.device)
    else:
        # Text-only query
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        try:
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"{system_prompt}\n\nClinical Query: {prompt_text}\nAnswer:"

        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    print("[MedGemma] Generating clinical response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=False
        )
        
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text.strip()
