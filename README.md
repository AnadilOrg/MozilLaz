# MozilLaz — Lazcanın Açık Kaynak Text-to-Speech Modeli

MozilLaz, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) üzerine LoRA fine-tuning ile eğitilmiş, **Lazca (lzz)** konuşma sentezi modeli. Mozilla'nın açık kaynak Lazca veri setleri kullanılarak eğitilmiştir.

| Özellik | Detay |
|---|---|
| Dil | Lazca (lzz) |
| Konuşucu | Tek konuşucu (Mozilla Lazca veri setinden) |
| Mimari | VoxCPM2 + LoRA (r=32, α=32) |
| Eğitim verisi | ~21.000 segment |
| Eğitim adımı | 2.000 |
| LoRA ağırlıkları | ~70 MB (18,1M parametre, F32) |
| Audio sample rate | 16 kHz (giriş) / 48 kHz (çıkış) |

## ⚡ Hızlı Başlangıç

```bash
# 1. Bu repo'yu klonlayın
git clone https://huggingface.co/Anadilorg/MozilLaz
cd MozilLaz

# 2. Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate

# 3. Gerekli paketleri yükleyin
pip install -r requirements.txt

# 4. Ses üretin (base model ilk seferde otomatik indirilir, ~4.6 GB)
python inference.py --text "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?"
```

> **Not:** LoRA yüklemesi `voxcpm` paketinin kendi LoRA desteğiyle yapılır — `peft` gerekmez.

## 📖 Kullanım

### Python API ile

```python
import json
import numpy as np
import soundfile as sf
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# 1. LoRA konfigürasyonunu okuyun
with open("lora_config.json") as f:
    lc = json.load(f)["lora_config"]
lora_cfg = LoRAConfig(**{k: v for k, v in lc.items() if k in LoRAConfig.model_fields})

# 2. Base model + MozilLaz LoRA adapter'ını tek adımda yükleyin
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_config=lora_cfg,
    lora_weights_path="lora_weights.safetensors",
)

# 3. Ses üretin — Lazca metin
text = "[speaker:spk_tmp_001 language:lzz] Mehmet manişa dulya komenç̌elu do oxorimuşişa igzalu."
audio = model.generate(text=text, inference_timesteps=10, cfg_value=2.0)

# 4. Kaydedin (çıkış 48 kHz)
sf.write("output.wav", np.asarray(audio).squeeze(), 48000)
print("✅ Ses üretildi! output.wav dosyasına kaydedildi.")
```

<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/1.wav"></audio>

### Batch inference

```python
texts = [
    "[speaker:spk_tmp_001 language:lzz] Mektebişa oxtimu na gyoç̆ǩu orape var şuns.",
    "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?",
]

for i, text in enumerate(texts):
    audio = model.generate(text=text, inference_timesteps=10, cfg_value=2.0)
    sf.write(f"output_{i}.wav", np.asarray(audio).squeeze(), 48000)
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/2.wav"></audio>
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/3.wav"></audio>

### Gradio ile tarayıcı demosu

```bash
pip install gradio
python demo.py            # http://localhost:7860
python demo.py --share    # public link
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/4.wav"></audio>

### Command Line

```bash
python inference.py \
    --text "[speaker:spk_tmp_001 language:lzz] Xelaǩaoba, manebrape!" \
    --output my_voice.wav \
    --base-model openbmb/VoxCPM2
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/5.wav"></audio>

### Hızlı doğrulama (smoke test)

```bash
python test_smoke.py   # ağırlık eşleşmesini ve uçtan uca üretimi doğrular
```

## 🏗️ Model Detayları

### VoxCPM2 Base Model
- **LM:** 28 kat, 2048 hidden dim, 16 attention heads, KV=2, vocab=73448
- **RoPE:** LongRoPE (32768 max position)
- **DiT:** 12 kat, 1024 hidden dim, CFM solver (Euler, log-norm, cfg_rate=2.0)
- **AudioVAE:** 16kHz → 48kHz, 64-dim latent
- **Patch size:** 4, feat_dim: 64

### LoRA Adapter (MozilLaz)
- **Rank (r):** 32
- **Alpha (α):** 32
- **Dropout:** 0.0
- **LM katmanları:** q_proj, k_proj, v_proj, o_proj (base LM 28 kat + residual LM 8 kat)
- **DiT katmanları:** q_proj, k_proj, v_proj, o_proj (12 kat)
- **Projeksiyon katmanları:** dahil değil (`enable_proj: false`)
- **Toplam:** 384 tensör, 18.087.936 parametre (F32)

## 📦 Paket Yapısı

```
MozilLaz/
├── README.md                 # Bu dosya
├── model_card.md             # HuggingFace model kartı
├── config.json               # Model metadata
├── lora_config.json          # LoRA hiperparametreler
├── lora_weights.safetensors  # ~70MB LoRA ağırlıkları
├── requirements.txt          # Gerekli Python paketleri
├── inference.py              # Tek dosyada tam inference script
├── demo.py                   # Gradio demo (opsiyonel)
├── test_smoke.py             # Uçtan uca doğrulama testi
├── sample/                   # Örnek çıktılar (5 WAV)
└── LICENSE                   # MIT
```

## 📋 Gereksinimler

| Paket | Versiyon |
|---|---|
| Python | >= 3.10 |
| torch | >= 2.0 |
| torchaudio | >= 2.0 |
| voxcpm | >= 2.0 |
| safetensors | >= 0.3 |
| soundfile | >= 0.11 |
| numpy | >= 1.24 |

### Donanım

- **Disk:** ~4.7 GB (base model + LoRA)
- **Bellek:** float32 inference için ~9 GB; CUDA GPU'da bfloat16 ile daha az
- **GPU:** CUDA önerilir. Apple Silicon (MPS) desteklenir; 16 GB birleşik bellekte çalışır ama swap nedeniyle çok yavaştır (≥24 GB önerilir). CPU çalışır fakat pratik değildir.

## 🌍 Dil Notu

Lazca (ISO 639-3: **lzz**), Karadeniz bölgesinde konuşulan Kolkis dilidir. Bu model **Mozilla'nın açık kaynak Lazca veri setleri** kullanılarak eğitilmiştir ve **tek konuşucu** modeli olarak sunulur.

## 📝 Lisans

Bu model MIT lisansı altında sunulmuştur. Detaylar için `LICENSE` dosyasına bakın.

## 🔗 Kaynaklar

- [VoxCPM2 (Base Model)](https://huggingface.co/openbmb/VoxCPM2)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mozilla Lazca Dil Projesi](https://mozilladatacollective.com/datasets/cmqinxu0l00ycnr07obbjovk0)

## 💬 Sorun Bildir

Sorunlarınızı veya önerilerinizi AnadilOrg[@]gmail.com
