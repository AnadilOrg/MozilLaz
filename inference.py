#!/usr/bin/env python3
"""
MozilLaz — Lazca Text-to-Speech
Standalone inference script.

Kullanım:
    python inference.py
    python inference.py --text "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?"
    python inference.py --text "..." --output ses.wav --device cuda

Model yükleme:
    1. openbmb/VoxCPM2 base modeli HuggingFace'den otomatik indirilir (~4.6 GB)
    2. Bu dizindeki lora_config.json okunarak LoRA katmanları kurulur
    3. lora_weights.safetensors adapter ağırlıkları yüklenir
    4. 48 kHz Lazca ses üretilir

Not: LoRA yüklemesi voxcpm paketinin kendi (native) LoRA desteğiyle yapılır;
peft gerekmez. Doğru giriş noktası `voxcpm.VoxCPM` sınıfıdır.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_BASE_MODEL = "openbmb/VoxCPM2"
DEFAULT_TEXT = "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_device(requested: str) -> str:
    """auto → cuda > mps > cpu"""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(base_model_name: str, lora_config_path: Path, lora_weights_path: Path, device: str):
    """
    VoxCPM2 base model + MozilLaz LoRA adapter yükle.

    voxcpm >= 2.0 API'si: LoRA config ve ağırlık yolu doğrudan
    `VoxCPM.from_pretrained`e verilir; model kurulurken LoRA katmanları
    eklenir ve safetensors ağırlıkları yüklenir.
    """
    try:
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm import LoRAConfig
    except ImportError:
        print("❌ voxcpm paketi bulunamadı!")
        print("   pip install voxcpm  komutuyla yükleyin (>= 2.0).")
        sys.exit(1)

    lc = load_json(lora_config_path).get("lora_config", {})
    lora_cfg = LoRAConfig(**{k: v for k, v in lc.items() if k in LoRAConfig.model_fields})

    print(f"📦 Base model yükleniyor: {base_model_name}")
    print("   (İlk seferde HuggingFace'den ~4.6 GB indirilir)")

    model = VoxCPM.from_pretrained(
        base_model_name,
        load_denoiser=False,
        lora_config=lora_cfg,
        lora_weights_path=str(lora_weights_path),
        device=device,
    )

    # Ağırlıkların gerçekten eşleştiğini doğrula
    loaded, skipped = model.load_lora(str(lora_weights_path))
    print(f"🔧 LoRA adapter: {len(loaded)} anahtar yüklendi, {len(skipped)} atlandı")
    if skipped:
        print(f"⚠️  Atlanan anahtarlar (ilk 5): {skipped[:5]}")

    return model


def generate_speech(model, text: str, output_path: str,
                    inference_timesteps: int = 10, cfg_value: float = 2.0):
    """Metinden ses üret ve dosyaya kaydet."""
    print(f"\n🎙️  Metin: {text}")
    print("   Ses üretiliyor...")

    audio = model.generate(
        text=text,
        inference_timesteps=inference_timesteps,
        cfg_value=cfg_value,
    )
    audio = np.asarray(audio).squeeze()

    sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
    sf.write(output_path, audio, sample_rate)
    print(f"\n✅ Ses üretildi! → {output_path}")
    print(f"   Uzunluk: {len(audio) / sample_rate:.2f} saniye")
    print(f"   Sample rate: {sample_rate} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="MozilLaz — Lazca Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Örnekler:

  # Varsayılan Lazca metin ile:
  python inference.py

  # Özel metin:
  python inference.py --text "[speaker:spk_tmp_001 language:lzz] Xelaǩaoba, manebrape!"

  # Cihaz seçimi (varsayılan auto: cuda > mps > cpu):
  python inference.py --device cuda

  # Farklı adımlarla (daha kaliteli ama yavaş):
  python inference.py --timesteps 20

Lazca metin formatı:
  [speaker:spk_tmp_001 language:lzz] <metin burada>

Speaker: spk_tmp_001 (Mozilla Lazca veri setinden)

Donanım notu:
  float32 inference ~9 GB bellek ister. CUDA GPU'da bfloat16 ile daha az.
  16 GB birleşik bellekli Apple Silicon'da çalışır ama swap nedeniyle
  ÇOK yavaştır; pratik kullanım için CUDA GPU veya >=24 GB önerilir.
"""
    )
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT,
                        help="Söylemesini istediğiniz Lazca metin")
    parser.add_argument("--output", "-o", type=str, default="mozilaz_output.wav",
                        help="Çıktı WAV dosya yolu (varsayılan: mozilaz_output.wav)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Hesaplama cihazı (varsayılan: auto)")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Inference adım sayısı (10=normal, 20=kaliteli, 5=hızlı)")
    parser.add_argument("--cfg-value", type=float, default=2.0,
                        help="Classifier-free guidance değeri (varsayılan: 2.0)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Base VoxCPM2 model adı (varsayılan: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA weights yolu (varsayılan: bu dizindeki lora_weights.safetensors)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    lora_weights = Path(args.lora_path) if args.lora_path else script_dir / "lora_weights.safetensors"
    lora_config = script_dir / "lora_config.json"

    for p, ad in [(lora_weights, "LoRA weights"), (lora_config, "lora_config.json")]:
        if not p.exists():
            print(f"❌ {ad} dosyası bulunamadı: {p}")
            sys.exit(1)

    config_path = script_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
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

    device = resolve_device(args.device)
    print(f"\n🚀 {device.upper()} üzerinde çalışıyor...")

    model = load_model(args.base_model, lora_config, lora_weights, device)

    generate_speech(
        model,
        text=args.text,
        output_path=args.output,
        inference_timesteps=args.timesteps,
        cfg_value=args.cfg_value,
    )


if __name__ == "__main__":
    main()
