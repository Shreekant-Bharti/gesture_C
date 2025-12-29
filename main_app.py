"""
main_app.py
Main production app with SIMPLE MODE for non-technical users:

SIMPLE MODE (Default):
- Clean, accessible UI with large text
- Auto-enhancement with Gemini AI
- Minimal technical information
- Perfect for elderly, deaf, and non-technical users

ADVANCED MODE (Press 'a'):
- Full developer/debug interface
- Real-time stats (FPS, confidence, etc.)
- Tutorial mode
- Manual controls

Controls (Simple Mode):
- s: speak sentence (TTS)
- c: clear sentence
- a: toggle Advanced Mode

Advanced Mode adds:
- w: write sentence to txt
- t: toggle tutorial overlay
- g: toggle Gemini AI
- e: manual enhance
- q: quit (works in both modes)
"""

import os
import time
import pickle
from collections import deque, Counter
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

from config import (
    CAMERA, MP, MODEL, UI, RUNTIME, GEMINI,
    GESTURES, TUTORIAL_TIPS
)
from gemini_ai import GeminiService


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA.index)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check webcam connection and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA.fps)
    return cap


def load_model_bundle(path: str):
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if "model" not in bundle:
            raise ValueError("Invalid model file: missing 'model'.")
        return bundle
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}. Run train_model.py first.")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


def landmarks_to_feature_vector(hand_landmarks) -> Optional[np.ndarray]:
    """
    21 landmarks -> 63 normalized features.
    """
    if hand_landmarks is None:
        return None

    pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)

    pts_norm = pts.copy()
    pts_norm[:, 0] = (pts[:, 0] - min_x) / w
    pts_norm[:, 1] = (pts[:, 1] - min_y) / h
    pts_norm[:, 2] = pts[:, 2] / max(w, h)

    return pts_norm.flatten()


def predict(model, x: np.ndarray) -> Tuple[str, float]:
    probs = model.predict_proba(x.reshape(1, -1))[0]
    idx = int(np.argmax(probs))
    label = model.classes_[idx]
    conf = float(probs[idx])
    return str(label), conf


def stable_prediction(
    label_buffer: deque,
    conf_buffer: deque,
    accept_ratio: float,
    min_conf: float
) -> Tuple[Optional[str], float]:
    """
    Accept a gesture if:
    - it appears in >= accept_ratio of buffer
    - average confidence for that gesture >= min_conf
    """
    if not label_buffer:
        return None, 0.0

    counts = Counter(label_buffer)
    top_label, top_count = counts.most_common(1)[0]
    ratio = top_count / len(label_buffer)

    if ratio < accept_ratio:
        return None, 0.0

    # average confidence for top label
    confs = [c for (lbl, c) in zip(label_buffer, conf_buffer) if lbl == top_label]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    if avg_conf < min_conf:
        return None, avg_conf

    return top_label, avg_conf


def speak_text(engine: pyttsx3.Engine, text: str) -> None:
    if not text.strip():
        return
    try:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")


def test_system_response() -> None:
    """
    🧪 FAIL-SAFE TEST MODE
    Test if the system can respond without camera/gestures.
    If this doesn't work, the system is fundamentally broken.
    """
    print("\n" + "="*60)
    print("🧪 TESTING SYSTEM RESPONSE (No camera required)")
    print("="*60)
    
    # Test TTS
    try:
        tts = pyttsx3.init()
        test_text = "System test successful"
        print(f"[TEST] Speaking: {test_text}")
        speak_text(tts, test_text)
        print("[✓ TEST PASSED] TTS working")
    except Exception as e:
        print(f"[✗ TEST FAILED] TTS error: {e}")
    
    # Test Gemini (if available)
    try:
        gemini = GeminiService()
        if gemini.is_available():
            result = gemini.enhance_sentence("hello water please")
            if result.success:
                print(f"[✓ TEST PASSED] Gemini AI working: {result.enhanced_text}")
            else:
                print(f"[⚠ TEST PARTIAL] Gemini available but enhancement failed: {result.error_message}")
        else:
            print(f"[INFO] Gemini not available (optional): {gemini.last_error}")
    except Exception as e:
        print(f"[ERROR] Gemini test crashed: {e}")
    
    print("="*60 + "\n")


def save_sentence_to_file(sentence: str) -> Optional[str]:
    if not sentence.strip():
        return None
    try:
        out_dir = "sentences"
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"sentence_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sentence.strip() + "\n")
        return path
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
        return None


def draw_confidence_bar(frame: np.ndarray, x: int, y: int, w: int, h: int, conf: float) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), UI.white, 2)
    fill = int(w * max(0.0, min(conf, 1.0)))
    cv2.rectangle(frame, (x, y), (x + fill, y + h), UI.green if conf >= 0.7 else UI.yellow, -1)


def draw_tutorial_overlay(frame: np.ndarray, page: int, page_size: int = 10) -> None:
    """
    Text-only tutorial overlay with pagination.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (60, 80), (w - 60, h - 80), (15, 15, 15), -1)

    alpha = 0.88
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    gestures = list(GESTURES)
    total_pages = max(1, (len(gestures) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = min(start + page_size, len(gestures))

    cv2.putText(frame, "Tutorial Mode (text reference)", (90, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2)
    cv2.putText(frame, f"Page {page+1}/{total_pages} | Use '[' and ']' to change page | Press 't' to close",
                (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, UI.white, 2)

    y = 210
    for g in gestures[start:end]:
        tip = TUTORIAL_TIPS.get(g, "Tip not available yet. Keep pose consistent while collecting.")
        text = f"{g}: {tip}"
        cv2.putText(frame, text, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.yellow, 2)
        y += 34


def draw_ui_simple(
    frame: np.ndarray,
    current_label: str,
    stable_label: Optional[str],
    sentence: str,
    enhanced_sentence: str,
    has_hand: bool,
    is_listening: bool
) -> None:
    """
    Simple, accessible UI for non-technical users.
    Large text, minimal info, friendly messages.
    """
    h, w = frame.shape[:2]
    
    # Clean background
    cv2.rectangle(frame, (0, 0), (w, h), (30, 30, 30), -1)
    
    # Title - Large and centered
    title = "Sign Language Recognition"
    title_scale = UI.simple_mode_title_scale
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                title_scale, UI.white, 3, cv2.LINE_AA)
    
    # Status message - Friendly and clear
    if not has_hand:
        status = "🖐 Show your hand to the camera"
        status_color = UI.yellow
    elif is_listening:
        status = "👀 Watching... keep signing"
        status_color = UI.green
    else:
        status = "✓ Ready"
        status_color = UI.green
    
    status_scale = UI.simple_mode_font_scale
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 2)[0]
    status_x = (w - status_size[0]) // 2
    cv2.putText(frame, status, (status_x, 150), cv2.FONT_HERSHEY_SIMPLEX,
                status_scale, status_color, 2, cv2.LINE_AA)
    
    # Detected word - Current gesture (large and prominent)
    y_pos = 240
    detected_label = "Detected:"
    cv2.putText(frame, detected_label, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                UI.simple_mode_font_scale, UI.light_gray, 2, cv2.LINE_AA)
    
    # Show stable label (what's being recognized)
    word = stable_label if stable_label else "..."
    word_scale = 2.0  # Extra large for visibility
    word_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, word_scale, 3)[0]
    word_x = (w - word_size[0]) // 2
    cv2.putText(frame, word, (word_x, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX,
                word_scale, UI.green, 3, cv2.LINE_AA)
    
    # Sentence box - Clean and centered
    y_pos += 180
    sentence_label = "Sentence:"
    cv2.putText(frame, sentence_label, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                UI.simple_mode_font_scale, UI.light_gray, 2, cv2.LINE_AA)
    
    # Draw sentence box
    box_x1, box_y1 = 60, y_pos + 20
    box_x2, box_y2 = w - 60, y_pos + 180
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), UI.white, 2)
    cv2.rectangle(frame, (box_x1 + 2, box_y1 + 2), (box_x2 - 2, box_y2 - 2), (40, 40, 40), -1)
    
    # Display sentence (use enhanced if available, otherwise raw)
    display_sentence = enhanced_sentence if enhanced_sentence.strip() else sentence
    if not display_sentence.strip():
        display_sentence = "(waiting for gestures...)"
    
    # Word wrap for long sentences
    words = display_sentence.split()
    lines: List[str] = []
    line = ""
    max_width = 35  # characters per line
    
    for word in words:
        if len(line) + len(word) + 1 <= max_width:
            line = (line + " " + word).strip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    
    # Display up to 5 lines
    lines = lines[-5:]
    text_y = box_y1 + 45
    for line_text in lines:
        cv2.putText(frame, line_text, (box_x1 + 20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2, cv2.LINE_AA)
        text_y += 30
    
    # Simple instructions at bottom
    instructions = "Press 's' to SPEAK  |  Press 'c' to CLEAR  |  Press 'a' for Advanced Mode"
    inst_scale = 0.7
    inst_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, 2)[0]
    inst_x = (w - inst_size[0]) // 2
    cv2.putText(frame, instructions, (inst_x, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, inst_scale, UI.light_gray, 2, cv2.LINE_AA)


def draw_ui(
    frame: np.ndarray,
    current_label: str,
    current_conf: float,
    stable_label: Optional[str],
    stable_conf: float,
    sentence: str,
    fps: float,
    total_words: int,
    avg_conf: float,
    gestures_per_min: float,
    gemini_enabled: bool = False,
    enhanced_sentence: str = ""
) -> None:
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 70), UI.bg_panel, -1)
    title = "ISL Real-time Gesture Recognition"
    if gemini_enabled:
        title += " + Gemini AI"
    cv2.putText(frame, title, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, UI.white, 2)

    # Left info panel
    cv2.rectangle(frame, (0, 70), (420, h), UI.bg_panel, -1)

    cv2.putText(frame, "Live Prediction", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.white, 2)
    cv2.putText(frame, f"Now: {current_label}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                UI.green if current_conf >= 0.7 else UI.yellow, 2)
    cv2.putText(frame, f"Conf: {current_conf:.2f}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.white, 2)
    draw_confidence_bar(frame, 20, 205, 360, 18, current_conf)

    cv2.putText(frame, "Stable (smoothing)", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.white, 2)
    stable_text = stable_label if stable_label else "None"
    cv2.putText(frame, f"Stable: {stable_text}", (20, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI.blue, 2)
    cv2.putText(frame, f"StableConf: {stable_conf:.2f}", (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.white, 2)
    draw_confidence_bar(frame, 20, 350, 360, 18, stable_conf)

    # Sentence panel
    sentence_label = "Raw Gesture Sentence" if gemini_enabled else "Sentence"
    cv2.putText(frame, sentence_label, (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    box_height = 560 if not gemini_enabled else 490
    cv2.rectangle(frame, (20, 430), (400, box_height), (45, 45, 45), -1)

    # wrap sentence
    words = sentence.strip().split()
    lines: List[str] = []
    line = ""
    for wd in words:
        if len(line) + len(wd) + 1 <= 28:
            line = (line + " " + wd).strip()
        else:
            lines.append(line)
            line = wd
    if line:
        lines.append(line)
    max_lines = 3 if gemini_enabled else 4
    lines = lines[-max_lines:]

    y = 465
    for ln in lines:
        cv2.putText(frame, ln, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
        y += 26
    
    # Enhanced sentence panel (if Gemini enabled)
    if gemini_enabled:
        cv2.putText(frame, "AI Enhanced Sentence", (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.green, 2)
        cv2.rectangle(frame, (20, 540), (400, 610), (35, 65, 35), -1)
        
        # Wrap enhanced sentence
        enh_words = enhanced_sentence.strip().split()
        enh_lines: List[str] = []
        enh_line = ""
        for wd in enh_words:
            if len(enh_line) + len(wd) + 1 <= 28:
                enh_line = (enh_line + " " + wd).strip()
            else:
                enh_lines.append(enh_line)
                enh_line = wd
        if enh_line:
            enh_lines.append(enh_line)
        enh_lines = enh_lines[-2:]
        
        y = 570
        for ln in enh_lines:
            cv2.putText(frame, ln, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
            y += 26

    # Stats
    stats_y = 640 if gemini_enabled else 610
    cv2.putText(frame, "Stats", (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI.white, 2)
    cv2.putText(frame, f"Words: {total_words}", (20, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"AvgConf: {avg_conf:.2f}", (20, stats_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"Gest/min: {gestures_per_min:.1f}", (20, stats_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, stats_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    
    # Gemini status indicator
    if gemini_enabled:
        status_text = "Gemini: ON"
        cv2.putText(frame, status_text, (250, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.green, 2)

    # Controls footer
    controls = "s speak | c clear | e enhance | g gemini | w save | t tutorial | q quit"
    cv2.putText(frame, controls, (440, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, UI.white, 2)


def main() -> None:
    # Load model
    try:
        bundle = load_model_bundle(MODEL.model_path)
        model = bundle["model"]
        feature_count = int(bundle.get("feature_count", 63))
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Open camera
    try:
        cap = open_camera()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # TTS engine
    try:
        tts = pyttsx3.init()
        tts.setProperty("rate", 165)
    except Exception as e:
        print(f"[ERROR] pyttsx3 init failed: {e}")
        tts = None
    
    # Gemini AI service (optional)
    gemini_service = GeminiService(tone=GEMINI.default_tone)
    gemini_enabled = GEMINI.enabled_by_default and gemini_service.is_available()
    if gemini_service.is_available():
        print(f"[✓ GEMINI] AI available. Press 'g' to toggle (currently: {'ON' if gemini_enabled else 'OFF'})")
    else:
        print(f"[⚠ GEMINI] AI unavailable: {gemini_service.last_error}")
        print("[INFO] System will work without AI enhancement")
    
    # UI Mode (Simple = user-friendly, Advanced = developer mode)
    simple_mode = UI.simple_mode_default
    print(f"[✓ MODE] Starting in {'Simple' if simple_mode else 'Advanced'} Mode. Press 'a' to toggle.")
    
    # Auto-enable Gemini in Simple Mode for best UX
    if simple_mode and gemini_service.is_available() and GEMINI.auto_enhance_in_simple_mode:
        gemini_enabled = True
        print("[✓ AUTO-ENHANCE] Gemini AI enabled for Simple Mode")
    
    # Critical startup message
    print("\n" + "="*60)
    print("🚀 SYSTEM READY - Show hand gestures to camera")
    print("   Gestures will be detected and added to sentence")
    print("   Press 's' to SPEAK, 'c' to CLEAR, 'q' to QUIT")
    print("="*60 + "\n")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    label_buffer = deque(maxlen=RUNTIME.smoothing_window)
    conf_buffer = deque(maxlen=RUNTIME.smoothing_window)

    sentence_words: List[str] = []
    last_added_label = ""
    last_add_time = 0.0
    
    # Gemini enhancement
    enhanced_sentence = ""
    last_enhancement_time = 0.0

    # Stats
    session_start = time.time()
    total_words_added = 0
    conf_sum = 0.0
    conf_count = 0

    tutorial_mode = False
    tutorial_page = 0

    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MP.max_num_hands,
        min_detection_confidence=MP.min_detection_confidence,
        min_tracking_confidence=MP.min_tracking_confidence,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read from camera.")
                break

            if CAMERA.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_label = "NoHand"
            current_conf = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                feats = landmarks_to_feature_vector(hand_landmarks)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

                if feats is not None and feats.shape[0] == feature_count:
                    pred_label, pred_conf = predict(model, feats)
                    current_label, current_conf = pred_label, pred_conf

                    # push into buffer only if above small floor (avoid pure noise)
                    label_buffer.append(pred_label)
                    conf_buffer.append(pred_conf)
                else:
                    current_label = "BadFeatures"
                    current_conf = 0.0
                    label_buffer.clear()
                    conf_buffer.clear()
            else:
                # no hand: clear buffer (prevents stale stability)
                label_buffer.clear()
                conf_buffer.clear()

            stable_lbl, stable_conf = stable_prediction(
                label_buffer, conf_buffer,
                accept_ratio=RUNTIME.smoothing_accept_ratio,
                min_conf=RUNTIME.confidence_threshold
            )

            # Sentence building logic: add only stable predictions with cooldown and no duplicates
            now = time.time()
            if stable_lbl:
                can_add = (now - last_add_time) >= RUNTIME.stable_cooldown_sec
                not_duplicate = (stable_lbl != last_added_label)

                if can_add and not_duplicate:
                    sentence_words.append(stable_lbl)
                    last_added_label = stable_lbl
                    last_add_time = now

                    # stats update
                    total_words_added += 1
                    conf_sum += stable_conf
                    conf_count += 1
                    
                    # 🔔 DEBUG: Confirm gesture was added
                    print(f"[✓ GESTURE ADDED] {stable_lbl} (confidence: {stable_conf:.2f})")
                    print(f"[SENTENCE] {' '.join(sentence_words)}")
                    
                    # Smart auto-enhancement (works in both modes, faster and smarter)
                    should_enhance = False
                    if gemini_enabled and gemini_service.is_available():
                        # Enhance after 2+ words OR immediately for certain key words
                        if len(sentence_words) >= 2:
                            should_enhance = (now - last_enhancement_time) >= 1.5  # Faster (1.5s)
                        # Enhance immediately for question/help words
                        if stable_lbl.lower() in ['how', 'what', 'where', 'when', 'why', 'help', 'please']:
                            should_enhance = True
                    
                    if should_enhance:
                        sentence_temp = " ".join(sentence_words)
                        print(f"[🤖 GEMINI] Enhancing: {sentence_temp}")
                        try:
                            result = gemini_service.enhance_sentence(sentence_temp)
                            if result.success:
                                enhanced_sentence = result.enhanced_text
                                last_enhancement_time = now
                                print(f"[✨ ENHANCED] {enhanced_sentence}")
                            else:
                                print(f"[⚠️ GEMINI] Enhancement failed: {result.error_message}")
                        except Exception as e:
                            print(f"[❌ ERROR] Gemini crashed: {e}")

            # FPS
            now2 = time.time()
            dt = now2 - prev_time
            prev_time = now2
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Derived stats
            elapsed_min = max((time.time() - session_start) / 60.0, 1e-6)
            gestures_per_min = total_words_added / elapsed_min
            avg_conf = (conf_sum / conf_count) if conf_count > 0 else 0.0

            sentence = " ".join(sentence_words)
            
            # Determine if hand is detected
            has_hand = results.multi_hand_landmarks is not None
            is_listening = stable_lbl is not None

            # Choose UI mode
            if simple_mode:
                draw_ui_simple(
                    frame=frame,
                    current_label=current_label,
                    stable_label=stable_lbl,
                    sentence=sentence,
                    enhanced_sentence=enhanced_sentence,
                    has_hand=has_hand,
                    is_listening=is_listening
                )
            else:
                # Advanced Mode - full debug UI
                draw_ui(
                    frame=frame,
                    current_label=current_label,
                    current_conf=current_conf,
                    stable_label=stable_lbl,
                    stable_conf=stable_conf,
                    sentence=sentence,
                    fps=fps,
                    total_words=total_words_added,
                    avg_conf=avg_conf,
                    gestures_per_min=gestures_per_min,
                    gemini_enabled=gemini_enabled,
                    enhanced_sentence=enhanced_sentence
                )

            # Tutorial mode only in Advanced Mode
            if not simple_mode and tutorial_mode:
                draw_tutorial_overlay(frame, tutorial_page, page_size=10)

            cv2.imshow(UI.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            
            if key == ord("a"):
                # Toggle between Simple and Advanced modes
                simple_mode = not simple_mode
                mode_name = "Simple" if simple_mode else "Advanced"
                print(f"[INFO] Switched to {mode_name} Mode")
                
                # Auto-enable Gemini when entering Simple Mode
                if simple_mode and gemini_service.is_available() and GEMINI.auto_enhance_in_simple_mode:
                    gemini_enabled = True
                    print("[INFO] Gemini AI auto-enabled for Simple Mode")

            if key == ord("c"):
                sentence_words.clear()
                enhanced_sentence = ""
                last_added_label = ""
                last_add_time = time.time()
                print("[✓ CLEARED] Sentence reset. Ready for new gestures.")

            if key == ord("s"):
                if tts is None:
                    print("[WARN] TTS not available.")
                elif not sentence.strip():
                    print("[WARN] No sentence to speak. Add gestures first.")
                else:
                    # Speak enhanced sentence if available, otherwise raw sentence
                    text_to_speak = enhanced_sentence if (gemini_enabled and enhanced_sentence) else sentence
                    print(f"[🔊 SPEAKING] {text_to_speak}")
                    try:
                        speak_text(tts, text_to_speak)
                        print("[✓ SPEECH COMPLETE]")
                    except Exception as e:
                        print(f"[ERROR] Speech failed: {e}")
                        print("[FALLBACK] Displaying text only")
            
            if key == ord("g"):
                # Toggle Gemini
                if gemini_service.is_available():
                    gemini_enabled = not gemini_enabled
                    status = "ON" if gemini_enabled else "OFF"
                    print(f"[INFO] Gemini AI: {status}")
                    if not gemini_enabled:
                        enhanced_sentence = ""
                else:
                    print(f"[WARN] Gemini unavailable: {gemini_service.last_error}")
            
            if key == ord("e"):
                # Manually trigger enhancement
                if not gemini_service.is_available():
                    print("[WARN] Gemini not available.")
                elif not sentence.strip():
                    print("[WARN] No sentence to enhance.")
                else:
                    print(f"[GEMINI] Enhancing: '{sentence}'")
                    result = gemini_service.enhance_sentence(sentence)
                    if result.success:
                        enhanced_sentence = result.enhanced_text
                        print(f"[GEMINI] Result: '{enhanced_sentence}' ({result.processing_time:.2f}s)")
                        last_enhancement_time = time.time()
                    else:
                        print(f"[GEMINI] Failed: {result.error_message}")

            if key == ord("w"):
                # Save enhanced sentence if available, otherwise raw sentence
                text_to_save = enhanced_sentence if (gemini_enabled and enhanced_sentence) else sentence
                if text_to_save.strip():
                    path = save_sentence_to_file(text_to_save)
                    if path:
                        sentence_type = "Enhanced" if (gemini_enabled and enhanced_sentence) else "Raw"
                        print(f"[SAVED] {sentence_type} sentence written to: {path}")
                    else:
                        print("[WARN] Save failed.")
                else:
                    print("[WARN] Nothing to save.")

            if key == ord("t"):
                # Tutorial mode only available in Advanced Mode
                if not simple_mode:
                    tutorial_mode = not tutorial_mode
                    print(f"[INFO] Tutorial mode: {'ON' if tutorial_mode else 'OFF'}")
                else:
                    print("[INFO] Tutorial is only available in Advanced Mode. Press 'a' to switch.")

            # Tutorial page controls (only in Advanced Mode)
            if not simple_mode and tutorial_mode:
                if key == ord("["):
                    tutorial_page = max(0, tutorial_page - 1)
                if key == ord("]"):
                    tutorial_page += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
