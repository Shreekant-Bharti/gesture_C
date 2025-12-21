"""
main_app.py
Main production app:
- Real-time hand tracking (MediaPipe)
- Gesture prediction (trained model)
- Prediction smoothing (rolling buffer)
- Sentence building + TTS (pyttsx3)
- Tutorial mode (text reference)
- Stats (words, avg confidence, gestures/min)
- Save sentence to file (timestamp)

Controls:
- s: speak sentence (TTS)
- c: clear sentence
- w: write sentence to txt
- t: toggle tutorial overlay
- q: quit
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
    CAMERA, MP, MODEL, UI, RUNTIME,
    GESTURES, TUTORIAL_TIPS
)


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
    gestures_per_min: float
) -> None:
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 70), UI.bg_panel, -1)
    cv2.putText(frame, "ISL Real-time Gesture Recognition (Hackathon)", (20, 45),
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
    cv2.putText(frame, "Sentence", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI.white, 2)
    cv2.rectangle(frame, (20, 430), (400, 560), (45, 45, 45), -1)

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
    lines = lines[-4:]

    y = 470
    for ln in lines:
        cv2.putText(frame, ln, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.white, 2)
        y += 28

    # Stats
    cv2.putText(frame, "Stats", (20, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI.white, 2)
    cv2.putText(frame, f"Words: {total_words}", (20, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"AvgConf: {avg_conf:.2f}", (20, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"Gest/min: {gestures_per_min:.1f}", (20, 705),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 735), cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)

    # Controls footer
    controls = "s speak | c clear | w save | t tutorial | q quit"
    cv2.putText(frame, controls, (440, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.white, 2)


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

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    label_buffer = deque(maxlen=RUNTIME.smoothing_window)
    conf_buffer = deque(maxlen=RUNTIME.smoothing_window)

    sentence_words: List[str] = []
    last_added_label = ""
    last_add_time = 0.0

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
                gestures_per_min=gestures_per_min
            )

            if tutorial_mode:
                draw_tutorial_overlay(frame, tutorial_page, page_size=10)

            cv2.imshow(UI.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                sentence_words.clear()
                last_added_label = ""
                last_add_time = time.time()
                print("[INFO] Sentence cleared.")

            if key == ord("s"):
                if tts is None:
                    print("[WARN] TTS not available.")
                else:
                    print(f"[TTS] Speaking: {sentence}")
                    speak_text(tts, sentence)

            if key == ord("w"):
                path = save_sentence_to_file(sentence)
                if path:
                    print(f"[SAVED] Sentence written to: {path}")
                else:
                    print("[WARN] Nothing to save.")

            if key == ord("t"):
                tutorial_mode = not tutorial_mode
                print(f"[INFO] Tutorial mode: {'ON' if tutorial_mode else 'OFF'}")

            # Tutorial page controls
            if tutorial_mode:
                if key == ord("["):
                    tutorial_page = max(0, tutorial_page - 1)
                if key == ord("]"):
                    tutorial_page += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
