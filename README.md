# ISL Real-time Gesture Recognition (MediaPipe + Scikit-learn + Gemini AI)

## 🎯 Two Modes for Everyone

### 👋 **SIMPLE MODE** (Default) - For Everyone

**Perfect for non-technical users, elderly, deaf, and public demos**

- ✅ Clean, large text interface
- ✅ Auto-enhancement with AI
- ✅ Zero technical knowledge needed
- ✅ Just show your hand and sign!

**Controls:** `s` to speak, `c` to clear, `a` for Advanced Mode

📖 **[Simple Mode User Guide →](SIMPLE_MODE_GUIDE.md)** _(Start here!)_

---

### 🔧 **ADVANCED MODE** - For Developers

Full-featured mode with:

- Real-time stats (FPS, confidence, etc.)
- Tutorial overlay
- Manual controls
- Debug information

**Press `a` to toggle between modes**

---

## 🌟 Features

- Real-time hand landmark detection (MediaPipe)
- Gesture recognition using ML (RandomForest)
- Sentence building with smoothing
- **🤖 AI-powered natural language enhancement (Google Gemini)** ✨
- Text-to-speech output (pyttsx3)
- Tutorial mode + stats + save sentence to file
- **👵 Accessibility-focused Simple Mode** ✨ NEW!

Works on Windows 10/11 with a normal laptop webcam.

---

## 🤖 Gemini AI Integration

Transform raw gesture sequences into natural English!

**Example:**

- **Input:** `hello water please`
- **Output:** `Hello, could I please have some water?`

**Features:**

- ✅ Natural language processing
- ✅ Grammar correction
- ✅ Multiple tones (polite, formal, casual)
- ✅ Auto-enhancement in Simple Mode
- ✅ Free tier (Google Gemini 2.5 Flash)

📖 **[Full Gemini Integration Guide →](GEMINI_INTEGRATION.md)**

---

## Setup (Under 30 minutes)

### 1) Create a virtual environment (recommended)

Open PowerShell in project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

If mediapipe fails, upgrade pip:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure (Cleaned & Optimized)

```
gesture/
├── main_app.py           # Main application (Simple + Advanced modes)
├── config.py             # Centralized configuration
├── gemini_ai.py          # AI enhancement module
├── requirements.txt      # Dependencies
├── gesture_model.pkl     # Trained ML model
├── README.md             # This file
├── tools/                # Development utilities
│   ├── collect_data.py   # Gesture data collection
│   └── train_model.py    # Model training
├── gesture_data/         # Training data (CSV files)
│   └── *.csv             # (42 gestures: A-Z, 0-9, words)
└── sentences/            # Saved output sentences
    └── *.txt
```

**Core Files (6):**

- Production: `main_app.py`, `config.py`, `gemini_ai.py`, `requirements.txt`, `gesture_model.pkl`
- Documentation: `README.md`

**Development Tools:**

- `tools/collect_data.py` - Collect gesture training data
- `tools/train_model.py` - Train RandomForest model

---

## 📦 Supported Gestures (42 Total)

**Letters (15):** A, B, C, D, E, F, G, H, I, L, O, U, V, W, Y  
**Numbers (10):** 0-9  
**Words (17):** Hello, Help, Thanks, Please, Yes, No, Okay, Stop, ...

_Configure in [config.py](config.py) → `GestureConfig.labels`_

---

## ⚙️ Configuration

All settings in [config.py](config.py):

**Gemini AI:**

```python
GEMINI = GeminiConfig(
    api_key="AIzaSyA...",           # Your API key
    model_name="gemini-2.5-flash",   # Model version
    auto_enhance_in_simple_mode=True # Auto-enhance
)
```

**Camera:**

```python
CAMERA = CameraConfig(
    index=0,      # Camera index (try 1 if 0 fails)
    width=1280,
    height=720,
    fps=30
)
```

**UI Mode:**

```python
UI = UIConfig(
    simple_mode_default=True  # Start in Simple Mode
)
```

---

---

## Step 1: Collect Data (VERY IMPORTANT)

Run:

```powershell
python tools/collect_data.py
```

You will see the webcam with landmarks.

**Controls:**

- `SPACE` = start/stop collecting
- `n` = next gesture label
- `q` = quit

**Tips for good accuracy:**

- Collect **100 to 200 samples per gesture**
- Change slightly between samples: angle, distance, tilt, lighting
- Keep each gesture consistent: do not mix multiple variants for the same label
- Try to keep your hand inside the frame

You will get:

```
gesture_data/A.csv
gesture_data/Hello.csv
... etc
```

---

## Step 2: Train the Model

Run:

```powershell
python tools/train_model.py
```

This will:

- load all CSVs
- train model
- show accuracy, report, confusion matrix
- save model to `gesture_model.pkl`

---

## Step 3: Run the Application

Run:

```powershell
python main_app.py
```

### 🎮 Controls

**Simple Mode (Default):**

- `s` = speak sentence (TTS)
- `c` = clear sentence
- `a` = switch to Advanced Mode

**Advanced Mode:**

- `s` = speak sentence (TTS)
- `c` = clear sentence
- `w` = save sentence to txt file
- `t` = toggle tutorial mode
- `a` = switch to Simple Mode
- `[` / `]` = tutorial pages (when tutorial active)
- `q` = quit

---

## Troubleshooting

### Camera not opening

- Close other apps using camera (Zoom, Teams, browser tabs).
- Try changing camera index in `config.py`:
  ```python
  CAMERA.index = 1
  ```

### Low accuracy

You likely collected inconsistent samples.  
**Fix:**

- collect more samples per gesture
- keep pose consistent per label
- improve lighting
- avoid motion during sampling (use stable poses while collecting)

### Mediapipe install issues

- Make sure you are using Python 3.10 or 3.11 (recommended).
- Check:
  ```powershell
  python --version
  ```

### TTS not speaking

- `pyttsx3` depends on Windows voices. Usually works.
- Try running `main_app.py` from a normal terminal, not inside some restricted environments.

---

## 🚀 Beginner Quick Start

```powershell
# 1. Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Collect gesture data
python tools/collect_data.py

# 3. Train model
python tools/train_model.py

# 4. Run app
python main_app.py
```

---

## 📝 Usage Guide (Simple Mode)

1. **Launch:** Run `python main_app.py`
2. **Sign:** Show ISL gestures to camera
3. **Build sentence:** Gestures auto-add after 1 second stability
4. **Enhance (Auto):** AI improves sentence after 2+ words
5. **Speak:** Press `s` to hear your sentence
6. **Clear:** Press `c` to start over

**Example Flow:**

```
Gesture: "Hello" → "Water" → "Please"
Auto-enhanced: "Hello, could I please have some water?"
Press 's' → Speaks aloud!
```

---
