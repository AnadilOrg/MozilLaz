#!/usr/bin/env python3
"""
MozilLaz — uçtan uca doğrulama (smoke test).

Şunları doğrular:
  1. lora_config.json geçerli ve voxcpm LoRAConfig'e dönüşüyor
  2. lora_weights.safetensors yapısal olarak sağlam (384 tensör, ~18,1M param)
  3. Base model + LoRA yükleniyor ve TÜM anahtarlar eşleşiyor (384/384)
  4. Gerçek Lazca ses üretimi çalışıyor (48 kHz çıktı)

Kullanım:
    python test_smoke.py                 # tam test (base modeli indirir, ~4.6 GB)
    python test_smoke.py --weights-only  # yalnızca ağırlık dosyası kontrolü (indirme yok)

Bu script, modelin ilk bağımsız doğrulamasında (Mac mini M4, MPS, voxcpm 2.0.3)
kullanılan test kodunun temizlenmiş halidir. O testte: 384/384 anahtar eşleşti,
"Ngolaşa uluri?" cümlesi 1,44 sn / 48 kHz ses olarak üretildi.
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXPECTED_TENSORS = 384
EXPECTED_PARAMS = 18_087_936
TEST_TEXT = "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?"


def check_weights(path: Path) -> bool:
    """Safetensors başlığını bağımsız oku (torch gerekmez)."""
    print(f"— Ağırlık dosyası: {path.name}")
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)

    n_tensors = len(header)
    n_params = 0
    dtypes = set()
    for v in header.values():
        n = 1
        for d in v["shape"]:
            n *= d
        n_params += n
        dtypes.add(v["dtype"])

    ok = True
    for ad, got, exp in [("tensör sayısı", n_tensors, EXPECTED_TENSORS),
                         ("parametre sayısı", n_params, EXPECTED_PARAMS)]:
        durum = "✅" if got == exp else "❌"
        if got != exp:
            ok = False
        print(f"  {durum} {ad}: {got:,} (beklenen {exp:,})")
    print(f"  ✅ dtype: {', '.join(sorted(dtypes))}" if dtypes == {"F32"}
          else f"  ⚠️ beklenmedik dtype: {dtypes}")

    # rank kontrolü: tüm lora_A tensörlerinin ilk boyutu r=32 olmalı
    bad_rank = [k for k, v in header.items()
                if k.endswith("lora_A") and v["shape"][0] != 32]
    if bad_rank:
        ok = False
        print(f"  ❌ r=32 olmayan lora_A tensörleri: {bad_rank[:3]}")
    else:
        print("  ✅ tüm lora_A tensörleri r=32")
    return ok


def full_test(base_model: str, device: str | None) -> bool:
    import numpy as np
    import soundfile as sf
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig

    with open(SCRIPT_DIR / "lora_config.json", encoding="utf-8") as f:
        lc = json.load(f)["lora_config"]
    cfg = LoRAConfig(**{k: v for k, v in lc.items() if k in LoRAConfig.model_fields})
    print(f"— LoRAConfig kuruldu: r={cfg.r}, α={cfg.alpha}, "
          f"lm={cfg.enable_lm}, dit={cfg.enable_dit}, proj={cfg.enable_proj}")

    print(f"— Base model yükleniyor: {base_model} (ilk seferde ~4.6 GB indirilir)")
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        base_model,
        load_denoiser=False,
        lora_config=cfg,
        lora_weights_path=str(SCRIPT_DIR / "lora_weights.safetensors"),
        device=device,
    )
    print(f"  ✅ yüklendi ({time.time() - t0:.0f} sn)")

    loaded, skipped = model.load_lora(str(SCRIPT_DIR / "lora_weights.safetensors"))
    durum = "✅" if (len(loaded) == EXPECTED_TENSORS and not skipped) else "❌"
    print(f"  {durum} LoRA anahtar eşleşmesi: {len(loaded)}/{EXPECTED_TENSORS} "
          f"(atlanan: {len(skipped)})")
    if skipped:
        print(f"     atlanan örnek: {skipped[:5]}")
        return False

    print(f"— Ses üretiliyor: {TEST_TEXT!r}")
    t0 = time.time()
    audio = model.generate(text=TEST_TEXT, inference_timesteps=10, cfg_value=2.0)
    dt = time.time() - t0
    audio = np.asarray(audio).squeeze()
    sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
    out = SCRIPT_DIR / "smoke_test_output.wav"
    sf.write(out, audio, sample_rate)
    dur = len(audio) / sample_rate

    ok = dur > 0.3 and float(abs(audio).max()) > 0.01
    print(f"  {'✅' if ok else '❌'} {dur:.2f} sn ses üretildi "
          f"({dt:.0f} sn sürdü, RTF={dt / dur:.1f}) → {out.name}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="MozilLaz smoke test")
    parser.add_argument("--weights-only", action="store_true",
                        help="Yalnızca ağırlık dosyası kontrolü (model indirmez)")
    parser.add_argument("--base-model", default="openbmb/VoxCPM2")
    parser.add_argument("--device", default=None,
                        help="cuda | mps | cpu (varsayılan: voxcpm otomatik seçer)")
    args = parser.parse_args()

    weights = SCRIPT_DIR / "lora_weights.safetensors"
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        print("❌ lora_weights.safetensors eksik ya da LFS pointer'ı — "
              "`git lfs pull` yapın veya dosyayı HF'den indirin.")
        sys.exit(1)

    ok = check_weights(weights)
    if not args.weights_only:
        ok = full_test(args.base_model, args.device) and ok

    print("\n" + ("🎉 TÜM TESTLER GEÇTİ" if ok else "💥 TEST BAŞARISIZ"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
