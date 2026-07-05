---
license: mit
language:
  - lzz
  - tur
tags:
  - text-to-speech
  - tts
  - lazuri
  - laz
  - voxcpm
  - lora
  - turkish
  - mozilla
pipeline_tag: text-to-speech
---

# Model Card: MozilLaz

## Model Detayları

**MozilLaz**, VoxCPM2 mimarisi üzerine LoRA fine-tuning ile eğitilmiş bir Lazca (Lazuri) Text-to-Speech modelidir. Mozilla'nın açık kaynak Lazca veri setleri kullanılarak eğitilmiştir.

### Yapı
- **Base Model:** openbmb VoxCPM2
- **Adapter:** LoRA (r=32, alpha=32)
- **Çıktı Sample Rate:** 48 kHz
- **Model Formatı:** Safetensors (LoRA weights ~70MB)

### Eğitim
- **Eğitim Verisi:** ~21,000 segment Mozilla Lazca veri setinden
- **Eğitim Adımları:** 2,000
- **Learning Rate:** 0.0001
- **Batch Size:** 2 (gradient accumulation: 8 → effective batch: 16)

### Sınırlamalar
- Tek konuşucu modeli (spk_tmp_001)
- Çoklu konuşucu veya çoklu dil desteği yok
- Kısa cümleler için optimize edilmiştir (21071 training sample ortalaması)
- Mozilla Lazca veri setinde olmayan Lazca lehçe veya sözlük kalıpları için sınırlı genelleme

## Kullanım

Modeli kullanmak için [inference.py](./inference.py) dosyasını referans alın.

### CLI Kullanımı

```bash
python inference.py --text "[speaker:spk_tmp_001 language:lzz] Metin burada" --output output.wav
```

### Python API

```python
from voxcpm.model import VoxCPM2Model
from peft import PeftModel

base = VoxCPM2Model.from_pretrained("openbmb/voxcpm2")
model = PeftModel.from_pretrained(base, ".")
model.eval().to("cuda")

audio = model.generate(
    target_text="[speaker:spk_tmp_001 language:lzz] Nanışkimi uç den ikayme.",
    inference_timesteps=10,
    cfg_value=2.0
)
```

## Lisans

Bu model MIT lisansı altında sunulmuştur.
