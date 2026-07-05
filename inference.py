#!/usr/bin/env python3
"""
MozilLaz — Lazca Text-to-Speech
Standalone inference script.

Kullanım:
    python inference.py
    python inference.py --text "[speaker:spk_tmp_001 language:lzz] Nanışkimi uç den ikayme."
    python inference.py --text "..." --output ses.wav --device cuda

Model yükleme:
    1. facebook/voxcpm2 base modeli HuggingFace'den otomatik indirilir
    2. Bu dizindeki lora_weights.safetensors LoRA adapter uygulanır
    3. 48kHz Lazca ses üretilir
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import torch
import transformers

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(path: Path) -> dict:
    """Yüklemek için config.json oku."""
    with open(path) as f:
        return json.load(f)


def find_voxcpm_model():
    """VoxCPM model sınıfını bul (pakete göre farklı import)."""
    # Deneme: voxcpm2 import
    try:
        from voxcpm.model import VoxCPM2Model
        return VoxCPM2Model
    except ImportError:
        pass

    # Deneme: voxcpm import
    try:
        from voxcpm.model import VoxCPMModel
        return VoxCPMModel
    except ImportError:
        pass

    return None


def load_model(base_model_name: str, lora_path: Path):
    """
    VoxCPM2 base model + LoRA adapter yükle.

    Args:
        base_model_name: HuggingFace base model adı (örn. 'facebook/voxcpm2')
        lora_path: LoRA weights safetensors dosya yolu

    Returns:
        Loaded model ready for inference
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA bulunamadı. CPU kullanılıyor (çok yavaş olacak).")

    print(f"📦 Base model yükleniyor: {base_model_name}")
    print("   (İlk seferde HuggingFace'den indirilecek ~4.3GB)")

    model_class = find_voxcpm_model()
    if model_class is None:
        print("❌ voxcpm paketi bulunamadı!")
        print("   pip install voxcpm komutuyla yükleyin.")
        sys.exit(1)

    # Base model'ı HuggingFace'ten yükle (otomatik indirir)
    model = model_class.from_pretrained(base_model_name)

    # LoRA adapter'ı yükle
    print(f"\n🔧 LoRA adapter yükleniyor: {lora_path.name}")
    model.load_lora(str(lora_path))

    # Inference modu
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("🎮 CUDA aktif")

    return model


def generate_speech(model, text: str, output_path: str,
                    inference_timesteps: int = 10, cfg_value: float = 2.0):
    """
    Metinden ses üret ve dosyaya kaydet.

    Args:
        model: Yüklenmiş VoxCPM2 + LoRA modeli
        text: Lazca text (speaker tag dahil)
        output_path: Çıktı WAV dosya yolu
        inference_timesteps: Samplama adımı sayısı (10 varsayılan)
        cfg_value: Classifier-free guidance değeri
    """
    print(f"\n🎙️  Metin: {text}")
    print("   Ses üretiliyor...")

    # VoxCPM2 generate API
    with torch.no_grad():
        audio = model.generate(
            target_text=text,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value
        )

    # Tensor → numpy
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze(0).numpy()

    # WAV'a kaydet
    sf.write(output_path, audio, 48000)
    print(f"\n✅ Ses üretildi! → {output_path}")
    print(f"   Uzunluk: {len(audio)/48000:.2f} saniye")
    print(f"   Sample rate: 48000 Hz")


def main():
    parser = argparse.ArgumentParser(
        description="MozilLaz — Lazca Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:

  # Varsayılan Lazca metin ile:
  python inference.py

  # Özel metin:
  python inference.py --text "[speaker:spk_tmp_001 language:lzz] Merhaba."

  # CUDA ile (varsayılan):
  python inference.py --device cuda

  # CPU ile (yavaş ama GPU gerekmez):
  python inference.py --device cpu

  # Farklı adımlarla (daha kaliteli ama yavaş):
  python inference.py --timesteps 20

Lazca metin formatı:
  [speaker:spk_tmp_001 language:lzz] <metin burada>

Speaker: spk_tmp_001 (Mozilla Lazca veri setinden)
"""
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Söylemesini istediğiniz Lazca metin"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="mozilaz_output.wav",
        help="Çıktı WAV dosya yolu (varsayılan: mozilaz_output.wav)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hesaplama cihazı (varsayılan: auto)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Inference adım sayısı (10=normal, 20=kaliteli, 5=hızlı)"
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance değeri (varsayılan: 2.0)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="facebook/voxcpm2",
        help="Base VoxCPM2 model adı (varsayılan: facebook/voxcpm2)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA weights yolu (varsayılan: bu dizindeki lora_weights.safetensors)"
    )

    args = parser.parse_args()

    # LoRA yolu
    script_dir = Path(__file__).parent
    lora_path = Path(args.lora_path) if args.lora_path else script_dir / "lora_weights.safetensors"

    if not lora_path.exists():
        print(f"❌ LoRA weights dosyası bulunamadı: {lora_path}")
        sys.exit(1)

    # Config
    config_path = script_dir / "config.json"
    if config_path.exists():
        config = load_config(config_path)
        print("=" * 60)
        print("  MozilLaz — Lazca Text-to-Speech Modeli")
        print("  " + "=" * 43)
        print(f"  Dil: {config.get('language_name', 'Lazuri (Lazca)')}")
        print(f"  Base model: {config.get('base_model', args.base_model)}")
        print(f"  Konuşucu: {', '.join(config.get('speakers', ['spk_tmp_001']))}")
        print(f"  LoRA: r={config.get('lora_config', {}).get('r', 32)}, "
              f"α={config.get('lora_config', {}).get('alpha', 32)}")
        print(f"  Training steps: {config.get('training', {}).get('steps', 2000)}")
        print("=" * 60)

    # Metin
    if args.text is None:
        args.text = "[speaker:spk_tmp_001 language:lzz] Nanışkimi uç den ikayme."

    # Cihaz
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🚀 {device.upper()} üzerinde çalışıyor...")
    print(f"📝 Metin: {args.text}")

    # Model yükle
    model = load_model(args.base_model, lora_path)

    # Ses üret
    generate_speech(
        model,
        text=args.text,
        output_path=args.output,
        inference_timesteps=args.timesteps,
        cfg_value=args.cfg_value
    )


if __name__ == "__main__":
    main()
