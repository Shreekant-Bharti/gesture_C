"""
diagnose_system.py
🔧 Complete system diagnostic tool

Run this BEFORE main_app.py to verify all components work.
This will identify exactly what's broken.
"""

import sys
import os

print("="*70)
print("🔧 ISL GESTURE RECOGNITION SYSTEM DIAGNOSTIC")
print("="*70 + "\n")

# Test 1: Python version
print("1️⃣ TESTING: Python version")
print(f"   Python {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✓ PASS\n")

# Test 2: Import dependencies
print("2️⃣ TESTING: Core dependencies")
failed_imports = []

try:
    import cv2
    print(f"   ✓ opencv-python {cv2.__version__}")
except ImportError as e:
    print(f"   ❌ opencv-python MISSING: {e}")
    failed_imports.append("opencv-python")

try:
    import mediapipe as mp
    print(f"   ✓ mediapipe {mp.__version__}")
except ImportError as e:
    print(f"   ❌ mediapipe MISSING: {e}")
    failed_imports.append("mediapipe")

try:
    import numpy as np
    print(f"   ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"   ❌ numpy MISSING: {e}")
    failed_imports.append("numpy")

try:
    import sklearn
    print(f"   ✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"   ❌ scikit-learn MISSING: {e}")
    failed_imports.append("scikit-learn")

try:
    import pyttsx3
    print(f"   ✓ pyttsx3 (TTS engine)")
except ImportError as e:
    print(f"   ❌ pyttsx3 MISSING: {e}")
    failed_imports.append("pyttsx3")

print()

# Test 3: Optional dependencies
print("3️⃣ TESTING: Optional dependencies (Gemini AI)")
try:
    import google.generativeai as genai
    print(f"   ✓ google-generativeai installed")
    gemini_available = True
except ImportError as e:
    print(f"   ⚠ google-generativeai NOT installed (optional)")
    print(f"     Install with: pip install google-generativeai")
    gemini_available = False
print()

# Test 4: Project files
print("4️⃣ TESTING: Project files")
required_files = [
    "config.py",
    "main_app.py",
    "gemini_ai.py",
    "requirements.txt"
]

for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ❌ {file} MISSING")
        failed_imports.append(file)
print()

# Test 5: Model file
print("5️⃣ TESTING: Trained model")
if os.path.exists("gesture_model.pkl"):
    print("   ✓ gesture_model.pkl found")
else:
    print("   ❌ gesture_model.pkl MISSING")
    print("     Run: python tools/train_model.py")
print()

# Test 6: Camera access
print("6️⃣ TESTING: Camera access")
if "cv2" in sys.modules:
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"   ✓ Camera 0 accessible (frame shape: {frame.shape})")
            else:
                print("   ⚠ Camera 0 opened but can't read frames")
        else:
            print("   ❌ Camera 0 failed to open")
            print("     - Close other apps using camera (Zoom, Teams, etc.)")
            print("     - Try changing camera index in config.py")
    except Exception as e:
        print(f"   ❌ Camera error: {e}")
else:
    print("   ⚠ Skipped (opencv not available)")
print()

# Test 7: TTS engine
print("7️⃣ TESTING: Text-to-Speech (TTS)")
if "pyttsx3" in sys.modules:
    try:
        tts = pyttsx3.init()
        tts.setProperty("rate", 165)
        print("   ✓ TTS engine initialized")
        print("   Testing speech...")
        tts.say("Diagnostic test complete")
        tts.runAndWait()
        print("   ✓ Speech test successful")
    except Exception as e:
        print(f"   ❌ TTS error: {e}")
else:
    print("   ⚠ Skipped (pyttsx3 not available)")
print()

# Test 8: Gemini API (if available)
print("8️⃣ TESTING: Gemini AI API")
if gemini_available:
    try:
        from config import GEMINI
        from gemini_ai import GeminiService
        
        service = GeminiService()
        if service.is_available():
            print(f"   ✓ Gemini service initialized")
            print(f"     Model: {GEMINI.model_name}")
            
            # Quick test
            result = service.enhance_sentence("hello water please")
            if result.success:
                print(f"   ✓ API test successful")
                print(f"     Input: 'hello water please'")
                print(f"     Output: '{result.enhanced_text}'")
            else:
                print(f"   ⚠ API call failed: {result.error_message}")
        else:
            print(f"   ⚠ Gemini unavailable: {service.last_error}")
            print("     Check API key in config.py")
    except Exception as e:
        print(f"   ❌ Gemini test error: {e}")
else:
    print("   ⚠ Skipped (google-generativeai not installed)")
print()

# Test 9: Configuration
print("9️⃣ TESTING: Configuration loading")
try:
    from config import CAMERA, MP, MODEL, UI, RUNTIME, GEMINI, GESTURES
    print(f"   ✓ Config loaded successfully")
    print(f"     Gestures: {len(GESTURES)} configured")
    print(f"     Confidence threshold: {RUNTIME.confidence_threshold}")
    print(f"     Accept ratio: {RUNTIME.smoothing_accept_ratio}")
    print(f"     Cooldown: {RUNTIME.stable_cooldown_sec}s")
except Exception as e:
    print(f"   ❌ Config error: {e}")
print()

# Final Summary
print("="*70)
print("📊 DIAGNOSTIC SUMMARY")
print("="*70)

if failed_imports:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for item in failed_imports:
        print(f"   - {item}")
    print("\n🔧 FIX:")
    print("   pip install -r requirements.txt")
    print("\n⚠ System will NOT work until these are fixed.\n")
else:
    print("\n✅ ALL CRITICAL TESTS PASSED")
    print("\n🚀 System is ready to run:")
    print("   python main_app.py")
    
    if not gemini_available:
        print("\n💡 OPTIONAL: Install Gemini AI for sentence enhancement:")
        print("   pip install google-generativeai")
    
    if not os.path.exists("gesture_model.pkl"):
        print("\n⚠ REQUIRED: Train the model first:")
        print("   python tools/collect_data.py")
        print("   python tools/train_model.py")
    
    print()

print("="*70 + "\n")
