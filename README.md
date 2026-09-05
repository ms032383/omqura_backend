# Omqura / ZA EYN Neuro-AI Backend Engine

High-performance Medical & Neurosurgical Decision-Support Backend built with FastAPI, MedGemma 4B LoRA, ChromaDB RAG, and multimodal clinical analysis.

---

## 🌟 Key Capabilities
- **MedGemma 4B LoRA Inference**: Specialized ARDSNet clinical decision support for ABG (Arterial Blood Gas) reports and ventilator parameters using fine-tuned LoRA adapter (`medgemma-abg-vent-lora`).
- **Neurosurgical RAG Engine**: ChromaDB vector retrieval with strict peer-reviewed citation integrity and numerical fidelity verification.
- **Multimodal Medical Vision**: Diagnostic imaging (CT, MRI, X-ray) and device telemetry monitor analysis.
- **Dynamic Model Routing**: Easily route queries to `medgemma-4b`, `gemini-1.5-flash`, or local Ollama LLMs.

---

## 🚀 Quickstart (Local or Cloud GPU VM)

### 1. Clone the Repository
```bash
git clone https://github.com/ms032383/omqura_backend.git
cd omqura_backend
```

### 2. Set Up Python Environment (Python 3.10+)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install base FastAPI & RAG dependencies
pip install -r requirements.txt

# For MedGemma GPU Inference (NVIDIA GPU / CUDA)
pip install -r requirements-gpu.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```
*(Optionally set `GEMINI_API_KEY`, `MEDGEMMA_ADAPTER_PATH`, etc. in `.env`)*

### 5. Hugging Face Login (For Gated MedGemma Base Model)
```bash
huggingface-cli login
```

### 6. Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
API Documentation will be available at: `http://localhost:8000/docs`

---

## ☁️ Google Cloud VM Deployment & Cloudflare Tunnel

To connect your backend securely to the Flutter frontend on GitHub Pages (HTTPS):

```bash
# 1. Install Cloudflare Tunnel
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# 2. Expose Port 8000 over Free Public HTTPS
cloudflared tunnel --url http://localhost:8000
```
This generates a secure URL (e.g. `https://random-name.trycloudflare.com`) that can be safely called from GitHub Pages without Mixed Content errors.
