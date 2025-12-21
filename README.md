# ISL Real-time Gesture Recognition (MediaPipe + Scikit-learn)

This project does:

- Real-time hand landmark detection (MediaPipe)
- Gesture recognition using ML (RandomForest)
- Sentence building with smoothing
- Text-to-speech output (pyttsx3)
- Tutorial mode + stats + save sentence to file

Works on Windows 10/11 with a normal laptop webcam.

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

## Folder structure

Place these files in one folder:

- `config.py`
- `collect_data.py`
- `train_model.py`
- `test_model.py`
- `main_app.py`
- `requirements.txt`

Data will be stored in:

- `gesture_data/` (CSV files)

Sentences saved in:

- `sentences/` (txt files)

---

## Step 1: Collect Data (VERY IMPORTANT)

Run:

```powershell
python collect_data.py
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
python train_model.py
```

This will:

- load all CSVs
- train model
- show accuracy, report, confusion matrix
- save model to `gesture_model.pkl`

---

## Step 3: Quick Live Testing

Run:

```powershell
python test_model.py
```

Shows:

- current predicted gesture
- confidence
- prediction history

Press `q` to quit.

---

## Step 4: Run the Full App

Run:

```powershell
python main_app.py
```

**Controls:**

- `s` = speak sentence (TTS)
- `c` = clear sentence
- `w` = save sentence to txt file
- `t` = tutorial mode overlay
- `q` = quit

In tutorial mode: `[` and `]` change pages

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

## Beginner usage summary

```powershell
python collect_data.py
python train_model.py
python test_model.py
python main_app.py
```

---

# Complete beginner guide (quick but crystal clear)

1. **Folder banayo** (like `isl_project/`) and sab files paste karo.
2. PowerShell open karo inside that folder.
3. Run:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
4. **Data collect**:
   - `python collect_data.py`
   - Gesture A select already hoga. **SPACE** dabao. 150 samples hone do.
   - **n** dabao next gesture. Repeat.
   - Tip: Har 4-5 samples me hand thoda tilt/rotate karo, distance change karo.
5. **Train**:
   - `python train_model.py`
   - Output me accuracy, report, confusion matrix aayega. Model save hoga.
6. **Test**:
   - `python test_model.py` (quick validation)
7. **Final app**:
   - `python main_app.py`
   - Stable gesture 80% buffer me hoga tab sentence me add hoga.
   - **s** to speak, **w** to save, **t** tutorial.

**Example (real use):**  
You show "Hello" stable for ~1 sec → sentence becomes: `Hello` → then "Help" → `Hello Help` → press **s** and it बोल देगा.

If you want, next message me main problems jo hackathon me aate hain (dataset bias, lighting, similar gestures confusion) ka analysis bhi de sakta hu!
