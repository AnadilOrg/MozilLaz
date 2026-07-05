#!/usr/bin/env python3
"""
MozilLaz — Gradio Interactive Demo

Bu demo, tarayıcı üzerinden Lazca TTS deneyimi sağlar.
Yüklemek için: pip install gradio

Çalıştırma:
    python demo.py
    python demo.py --port 7861
    python demo.py --share

Ardından tarayıcınızda http://localhost:7860 adresine gidin.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio yüklenmemiş. Kurmak için: pip install gradio")
    print("\nYa da CLI kullanın: python inference.py --text '...'")
    sys.exit(1)

# inference.py'deki doğrulanmış yükleyiciyi yeniden kullan
from inference import DEFAULT_BASE_MODEL, load_model, resolve_device

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
        config_path = script_dir / "config.json"

        base_model_name = DEFAULT_BASE_MODEL
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                base_model_name = json.load(f).get("base_model", DEFAULT_BASE_MODEL)

        load_model_once._model = load_model(
            base_model_name,
            lora_config_path=script_dir / "lora_config.json",
            lora_weights_path=script_dir / "lora_weights.safetensors",
            device=resolve_device("auto"),
        )
    return load_model_once._model


def synthesize(text: str, timesteps: int, cfg_value: float):
    """Metinden ses üret ve WAV dosya yolu döndür."""
    if not text.strip():
        return None

    model = load_model_once()
    audio = model.generate(
        text=text,
        inference_timesteps=int(timesteps),
        cfg_value=float(cfg_value),
    )
    audio = np.asarray(audio).squeeze()
    sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)

    tmp_path = "/tmp/mozilaz_demo.wav"
    sf.write(tmp_path, audio, sample_rate)
    return tmp_path


def gradio_interface(port: int = 7860, share: bool = False):
    """Gradio UI oluştur."""
    examples = LAZCA_SAMPLES

    with gr.Blocks(title="MozilLaz — Lazca TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎙️ MozilLaz — Lazca Text-to-Speech

**Lazca açık kaynak TTS modeli** (VoxCPM2 + LoRA, ~21.000 eğitim örneği)

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

        gr.Markdown(f"""
### Teknik Detaylar
- **Base Model:** [VoxCPM2 (OpenBMB)](https://huggingface.co/openbmb/VoxCPM2)
- **LoRA Rank:** 32, **Alpha:** 32
- **Eğitim Verisi:** ~21.000 Mozilla Lazca segment
- **Sample Rate:** 48 kHz
- **Speaker:** spk_tmp_001

### Kurulum
```bash
pip install torch torchaudio soundfile safetensors numpy voxcpm gradio
git clone https://huggingface.co/Anadilorg/MozilLaz
cd MozilLaz
python demo.py
```
""")

    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MozilLaz Gradio Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port numarası")
    parser.add_argument("--share", action="store_true", help="Public share link oluştur")
    args = parser.parse_args()

    gradio_interface(port=args.port, share=args.share)
