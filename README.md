# ISL Real-Time Gesture Recognition System

A real-time Indian Sign Language (ISL) gesture recognition system using MediaPipe hand tracking, machine learning, and AI-powered natural language enhancement.

## 🌟 Features

- **Real-time hand gesture recognition** using MediaPipe and Random Forest classifier
- **Two user modes:**
  - **Simple Mode** (default): Clean, accessible interface for non-technical users
  - **Advanced Mode**: Full developer interface with stats and debugging
- **AI-powered sentence enhancement** with Google Gemini API
- **Text-to-speech** output for recognized gestures
- **Tutorial mode** with gesture instructions
- **Sentence building** with gesture smoothing
- **Save sentences** to text files

## 📋 Supported Gestures

### Alphabets

A, B, C, D, E, F, G, H, I, L, O, U, V, W, Y

### Common Words

Hello, Thanks, Yes, No, Help, Please, Sorry, Stop, Okay, Good, Bad, Water, Food, Love, Peace

### Numbers

0-9

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows 10/11 (tested)

### Installation

1. **Clone the repository**

```powershell
git clone <your-repo-url>
cd gesture
```

2. **Create a virtual environment** (recommended)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**

```powershell
pip install -r requirements.txt
```

4. **Set up Gemini API (Optional, for AI enhancement)**
   - Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Training the Model

Before first use, train the gesture recognition model:

```powershell
# Collect training data (follow on-screen instructions)
python tools/collect_data.py

# Train the model
python tools/train_model.py
```

### Running the Application

```powershell
python main_app.py
```

## 🎮 Controls

### Simple Mode (Default)

- `s` - Speak sentence with text-to-speech
- `c` - Clear current sentence
- `a` - Toggle to Advanced Mode
- `q` - Quit

### Advanced Mode

All Simple Mode controls plus:

- `w` - Write sentence to file
- `t` - Toggle tutorial overlay
- `g` - Toggle Gemini AI enhancement
- `e` - Manually enhance current sentence
- `q` - Quit

## 🤖 AI Enhancement

The system uses Google Gemini AI to transform raw gesture sequences into natural English sentences.

**Example:**

- Raw: `hello water please`
- Enhanced: `Hello, could I please have some water?`

The AI enhancement is optional and the system works without it.

## 📁 Project Structure

```
gesture/
├── main_app.py           # Main application
├── config.py             # Configuration settings
├── gemini_ai.py          # AI enhancement module
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── gesture_data/         # Training data (CSV files)
└── tools/
    ├── collect_data.py   # Data collection script
    └── train_model.py    # Model training script
```

## 🛠️ Configuration

Edit `config.py` to customize:

- Camera settings (resolution, FPS)
- Gesture list
- Model parameters
- UI settings
- Runtime behavior

## 📝 How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time
2. **Feature Extraction**: 21 hand landmarks are normalized to 63 features
3. **Gesture Recognition**: Random Forest classifier predicts the gesture
4. **Smoothing**: Consecutive frames are analyzed to reduce noise
5. **Sentence Building**: Recognized gestures are combined into sentences
6. **AI Enhancement**: (Optional) Gemini AI improves grammar and naturalness
7. **Output**: Display on screen and optional text-to-speech

## 🔒 Privacy & Security

- No personal data is collected or transmitted
- Camera feed is processed locally
- Gemini API key is kept in `.env` file (not committed to Git)
- All gesture data stays on your device

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Add more gestures
- Improve documentation

## 📄 License

This project is provided as-is for educational and personal use.

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- scikit-learn for machine learning
- Google Gemini for AI enhancement

---

**Note**: This system requires a trained model file (`gesture_model.pkl`). Make sure to run the data collection and training scripts before using the main application.
