# MozilLaz — Lazcanın Açık Kaynak Text-to-Speech Modeli

MozilLaz, [VoxCPM2](https://huggingface.co/openbmb/voxcpm2) üzerine LoRA fine-tuning ile eğitilmiş, **Lazca (lzz)** konuşma sentezi modeli. Mozilla'nın açık kaynak Lazca veri setleri kullanılarak eğitilmiştir.

| Özellik | Detay |
|---|---|
| Dil | Lazca (lzz) |
| Konuşucu | Tek konuşucu ( Mozilla Lazca veri setinden) |
| Mimari | VoxCPM2 + LoRA (r=32, α=32) |
| Eğitim verisi | ~21.000 segment |
| Eğitim adımı | 2.000 |
| LoRA ağırlıkları | ~70 MB |
| Audio sample rate | 16 kHz (giriş) / 48 kHz (çıkış) |

## ⚡ Hızlı Başlangıç

```bash
pip install mozilaz --index-url https://huggingface.co --trusted-host huggingface.co
```

Ya da doğrudan repo'dan:

```bash
pip install torch torchaudio soundfile
pip install voxcpm  # base VoxCPM2 paketi
pip install -e git+https://huggingface.co/AnadilOrg/MozilLaz.git#egg=mozilaz
```

### Kolay kurulum (manuel):

```bash
# 1. Bu repo'yu klonlayın
git clone https://huggingface.co/AnadilOrg/MozilLaz.git
cd MozilLaz

# 2. Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate

# 3. Gerekli paketleri yükleyin
pip install torch torchaudio soundfile
pip install voxcpm

# 4. Basit inference yapın
python inference.py
```

## 📖 Kullanım

### Python API ile

```python
import torch
import soundfile as sf
from voxcpm.model import VoxCPM2Model
from peft import PeftModel

# 1. Base modeli yükleyin
base_model = VoxCPM2Model.from_pretrained("openbmb/voxcpm2")

# 2. MozilLaz LoRA adapter'ını yükleyin
model = PeftModel.from_pretrained(base_model, "MozilLaz/")
model.eval().to("cuda")

# 3. Ses üretin — Lazca metin
text = "[speaker:spk_tmp_001 language:lzz] Mehmet manişa dulya komenç̌elu do oxorimuşişa igzalu."
audio = model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)

# 4. Kaydedin
sf.write("output.wav", audio.cpu().numpy(), 48000)
print("✅ Ses üretildi! output.wav dosyasına kaydedildi.")
```

<audio controls="" src="sample/1.wav"></audio>

### Batch inference

```python
texts = [
    "[speaker:spk_tmp_001 language:lzz] Mektebişa oxtimu na gyoç̆ǩu orape var şuns.",
    "[speaker:spk_tmp_001 language:lzz] Ngolaşa uluri?",
]

for text in texts:
    audio = model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)
    sf.write(f"output.wav", audio.cpu().numpy(), 48000)
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/2.wav"></audio>
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/3.wav"></audio>


### GRUUI ile demo (Gradio gerektirmez — soundfile ile)

```python
# CLI kullanım
python inference.py --text "[speaker:spk_tmp_001 language:lzz] Aya lemşik va duç̌vinasinon." --output result.wav

# Base URL ile (REST API olarak kullanmak isterseniz)
# model.generate() output 48kHz stereo değil, mono audio tensor döner
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/4.wav"></audio>


### Command Line

```bash
python inference.py \
    --text "[speaker:spk_tmp_001 language:lzz] Xelaǩaoba, manebrape!" \
    --output my_voice.wav \
    --base-model openbmb/voxcpm2
```
<audio controls="" src="/Anadilorg/MozilLaz/resolve/main/sample/5.wav"></audio>


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
- **LM layer'lar:** q_proj, k_proj, v_proj, o_proj
- **DiT layer'lar:** q_proj, k_proj, v_proj, o_proj
- **Projeler:** enc_to_lm_proj, lm_to_dit_proj, res_to_dit_proj, fusion_concat_proj

## 📦 Paket Yapısı

```
MozilLaz/
├── README.md              # Bu dosya
├── config.json            # Model metadata
├── lora_config.json       # LoRA hiperparametreler
├── lora_weights.safetensors  # ~70MB LoRA ağırlıkları
├── requirements.txt       # Gerekli Python paketleri
├── inference.py           # Tek dosyada tam inference script
├── demo.py                # Gradio demo (opsiyonel)
├── pyproject.toml         # pip install mozilaz için metadata
└── setup.py               # Alternatif kurulum
```

## 📋 Gereksinimler

| Paket | Versiyon |
|---|---|
| Python | >= 3.10 |
| torch | >= 2.0 |
| torchaudio | >= 2.0 |
| voxcpm | en son |
| peft | >= 0.7 |
| safetensors | >= 0.3 |
| soundfile | >= 0.11 |
| accelerate | >= 0.21 |

## 🌍 Dil Notu

Lazca (ISO 639-3: **lzz**), Karadeniz bölgesinde konuşulan Kolkis dilidir. Bu model **Mozilla'nın açık kaynak Lazca veri setleri** kullanılarak eğitilmiştir ve **tek konuşucu** modeli olarak sunulur.

## 📝 Lisans

Bu model açık kaynak olarak sunulmuştur. Detaylar için `LICENSE` dosyasına bakın.

## 🔗 Kaynaklar

- [VoxCPM2 (Base Model)](https://huggingface.co/openbmb/voxcpm2)
- [LoRA Papers](https://arxiv.org/abs/2106.09685)
- [Mozilla Lazca Dil Projesi](https://mozilladatacollective.com/datasets/cmqinxu0l00ycnr07obbjovk0)

## 💬 Sorun Bildir

Sorunlarınızı veya önerilerinizi AnadilOrg[@]gmail.com
