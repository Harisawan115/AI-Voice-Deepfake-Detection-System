# 🎙️ AI Voice Deepfake Detection System

A deep learning system to detect whether a voice is **real human speech** or **AI-generated (deepfake)** audio. Built with CNN and Wav2Vec2 Transformer models, deployed via Streamlit.

---

## 📌 Project Info

| Detail | Info |
|---|---|
| **Dataset** | ASVspoof 2019 — Logical Access (LA) |
| **Models** | CNN (Mel Spectrogram) + Wav2Vec2 Transformer |
| **Deployment** | Streamlit Web App |

---

## 🎯 Results

| Model | Validation Accuracy | Test Accuracy | ROC-AUC |
|---|---|---|---|
| CNN (Mel Spectrogram) | 99.67% | 89.80% | 0.9742 |
| Wav2Vec2 Transformer | In Progress | — | — |

---

## 🗂️ Dataset

**ASVspoof 2019 — Logical Access (LA)**

This project uses the **LA (Logical Access)** subset which contains real human voices vs AI-generated/TTS voices — exactly the task of detecting voice deepfakes.

| Split | Total | Real (Bonafide) | Fake (Spoof) |
|---|---|---|---|
| Train | 25,380 | 2,580 | 22,800 |
| Dev | 24,844 | 2,548 | 22,296 |
| Eval | 71,237 | 7,355 | 63,882 |

**Download Link:**
> 🔗 [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

After downloading, extract and place the `LA/` folder in the project root directory.

> ⚠️ Only download `LA.zip` (7.1 GB) — `PA.zip` is not needed for this project.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/voice-deepfake-detection.git
cd voice-deepfake-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa streamlit scikit-learn matplotlib seaborn numpy pandas soundfile audioread
```

> For CPU only (no GPU):
> ```bash
> pip install torch torchvision torchaudio
> pip install transformers librosa streamlit scikit-learn matplotlib seaborn numpy pandas soundfile audioread
> ```

### 4. Download Dataset
Download `LA.zip` from the link above and extract it so your structure looks like:
```
project_folder/
└── LA/
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    ├── ASVspoof2019_LA_eval/
    └── ASVspoof2019_LA_cm_protocols/
```

### 5. Update Path in utils.py
Open `utils.py` and update line 8 with your dataset path:
```python
BASE_PATH = r"C:\your\path\to\LA"
```

---

## 🚀 Training

### Train CNN Model
```bash
python train_cnn.py
```
- Training time: ~30-40 min (GPU) | ~2-3 hours (CPU)
- Saves: `models/cnn_best_model.pth`

### Train Wav2Vec2 Model
```bash
python train_wav2vec2.py
```
- Training time: ~45-60 min (GPU)
- Saves: `models/wav2vec2_best_model.pth`
- First run downloads Wav2Vec2 weights (~380MB)

---

## 🌐 Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Open browser at: `http://localhost:8501`

### How to Use
1. Select model from sidebar (CNN / Wav2Vec2 / Both)
2. Upload an audio file (`.wav`, `.mp3`, `.flac`)
3. Click **Analyze Now**
4. See result: ✅ Real Voice or ⚠️ AI-Generated/Fake

---

## 🧠 Model Architecture

### CNN Model
- Input: Mel Spectrogram (128 × 128)
- 4 Conv blocks with BatchNorm + MaxPool + Dropout
- Fully connected classifier head
- Output: Real / Fake (2 classes)

### Wav2Vec2 Model
- Base: `facebook/wav2vec2-base` (pretrained)
- Feature extractor frozen, transformer encoder fine-tuned
- Mean pooling over time dimension
- Custom classifier head: 768 → 512 → 128 → 2
- Trained with WeightedRandomSampler for class balance

---

## 📊 How it Works

```
Audio File (.wav/.flac/.mp3)
        ↓
   Load & Resample (16kHz)
        ↓
  ┌─────────────────┐
  │   CNN Branch    │  → Mel Spectrogram → CNN → Softmax
  └─────────────────┘
  ┌─────────────────┐
  │ Wav2Vec2 Branch │  → Raw Waveform → Transformer → Softmax
  └─────────────────┘
        ↓
  Real / Fake + Confidence Score
```
---

## 📜 References

- [ASVspoof 2019 Dataset Paper](https://arxiv.org/abs/2109.00535)
- [Wav2Vec2 Paper — Facebook AI](https://arxiv.org/abs/2006.11477)
- [ASVspoof Challenge Official Site](https://www.asvspoof.org/)

```

---

## 📧 Contact

BS Artificial Intelligence — 6th Semester Project
