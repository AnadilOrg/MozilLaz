#!/usr/bin/env python3
"""
MozilLaz — Gradio Interactive Demo

Bu demo, tarayıcı üzerinden Lazca TTS deneyimi sağlar.
Yüklemek için: pip install gradio

Çalıştırma:
    python demo.py

Ardından tarayıcınızda http://localhost:7860 adresine gidin.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio yüklenmemiş. Kurmak için: pip install gradio")
    print("\nYa da CLI kullanın: python inference.py --text '...'")
    sys.exit(1)


# Lazca örnek cümleler
LAZCA_SAMPLES = [
    "[speaker:spk_tmp_001 language:lzz] Nanışkimi uç den ikayme.",
    "[speaker:spk_tmp_001 language:lzz] Dido ini on.",
    "[speaker:spk_tmp_001 language:lzz] Ğormotik gamaǩç̌ǩvidan.",
    "[speaker:spk_tmp_001 language:lzz] Aya lemşik va duç̌vinasinon.",
    "[speaker:spk_tmp_001 language:lzz] Lazuri vao, fendo vao.",
]


def load_model_once():
    """Modeli hafızada tut (her seferinde tekrar yükleme)."""
    if not hasattr(load_model_once, "_model"):
        script_dir = Path(__file__).parent
        lora_path = script_dir / "lora_weights.safetensors"
        config_path = script_dir / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model_name = config.get("base_model", "facebook/voxcpm2")
        else:
            base_model_name = "facebook/voxcpm2"

        # VoxCPM2 model sınıfını bul
        model_class = None
        for import_path in ["voxcpm.model", "voxcpm"]:
            try:
                model_class = getattr(__import__(import_path, fromlist=[""]), "VoxCPM2Model") or \
                               getattr(__import__(import_path, fromlist=[""]), "VoxCPMModel")
                break
            except ImportError:
                continue

        if model_class is None:
            raise ImportError("voxcpm paketi bulunamadı. pip install voxcpm yapın.")

        print(f"Base model: {base_model_name}")
        load_model_once._model = model_class.from_pretrained(base_model_name)

        # LoRA yükle
        print(f"LoRA: {lora_path}")
        load_model_once._model.load_lora(str(lora_path))
        load_model_once._model.eval()
        if torch.cuda.is_available():
            load_model_once._model = load_model_once._model.to("cuda")

    return load_model_once._model


def synthesize(text: str, timesteps: int, cfg_value: float):
    """Metinden ses üret ve WAV tempfile döndür."""
    if not text.strip():
        return None

    model = load_model_once()

    with torch.no_grad():
        audio = model.generate(
            target_text=text,
            inference_timesteps=timesteps,
            cfg_value=cfg_value
        )

    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze(0).numpy()

    # WAV dosyasına kaydet
    tmp_path = "/tmp/mozilaz_demo.wav"
    sf.write(tmp_path, audio, 48000)
    return tmp_path


def gradio_interface():
    """Gradio UI oluştur."""
    examples = LAZCA_SAMPLES

    with gr.Blocks(title="MozilLaz — Lazca TTS",
                   gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎙️ MozilLaz — Lazca Text-to-Speech

**Lazca açık kaynak TTS modeli** (VoxCPM2 + LoRA, 21.000 eğitim örneği)

> *İpucu:* Aşağıdaki örneklerden birine tıklayın veya kendi Lazca metninizi yazın!
""")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Lazca Metin",
                    placeholder="[speaker:spk_tmp_001 language:lzz] Metin burada...",
                    value=examples[0]
                )
                timesteps_slider = gr.Slider(
                    minimum=5, maximum=30, value=10, step=1,
                    label="Inference Adımları (daha yüksek = daha kaliteli ama yavaş)"
                )
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=4.0, value=2.0, step=0.1,
                    label="CFG Value (konuşma kalitesi)"
                )
                btn = gr.Button("🎙️ Ses Üret", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Çıktı Ses", type="filepath")
                gr.Examples(
                    examples=examples,
                    inputs=text_input,
                    label="Lazca Örnek Cümleler"
                )

        btn.click(
            fn=synthesize,
            inputs=[text_input, timesteps_slider, cfg_slider],
            outputs=audio_output
        )

        gr.Markdown("""
### Teknik Detaylar
- **Base Model:** [VoxCPM2 (Facebook)](https://huggingface.co/facebook/voxcpm2)
- **LoRA Rank:** 32, **Alpha:** 32
- **Eğitim Verisi:** ~21.000 Mozilla Lazca segment
- **Sample Rate:** 48 kHz
- **Speaker:** spk_tmp_001

### Kurulum
```bash
pip install torch torchaudio soundfile voxcpm peft safetensors
git clone https://huggingface.co/username/MozilLaz
cd MozilLaz
python demo.py
```
""")

    demo.launch(share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MozilLaz Gradio Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port numarası")
    parser.add_argument("--share", action="store_true", help="Public share link oluştur")
    args = parser.parse_args()

    gradio_interface()
