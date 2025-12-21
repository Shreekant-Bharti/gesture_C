"""
test_model.py
Quick real-time model test:
- shows webcam feed
- predicts gesture + confidence
- shows a small prediction history

Press q to quit.
"""

import time
import pickle
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from config import CAMERA, MP, MODEL, UI, RUNTIME


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


def draw_overlay(frame: np.ndarray, label: str, conf: float, history: deque, fps: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 120), UI.bg_panel, -1)
    cv2.putText(frame, "Model Test | Live Prediction", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2)
    cv2.putText(frame, f"Gesture: {label}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.green if conf >= 0.7 else UI.yellow, 2)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (360, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (600, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2)

    # History box
    cv2.rectangle(frame, (w - 380, 130), (w - 20, 330), UI.bg_panel, -1)
    cv2.putText(frame, "History (latest top)", (w - 370, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.white, 2)
    y = 190
    for item in list(history)[-8:][::-1]:
        cv2.putText(frame, item, (w - 370, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, UI.white, 2)
        y += 22

    cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.white, 2)


def main() -> None:
    try:
        bundle = load_model_bundle(MODEL.model_path)
        model = bundle["model"]
        feature_count = int(bundle.get("feature_count", 63))
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    try:
        cap = open_camera()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    history = deque(maxlen=20)
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

            label = "NoHand"
            conf = 0.0

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
                    label, conf = predict(model, feats)
                    if conf >= RUNTIME.confidence_threshold:
                        history.append(f"{label} ({conf:.2f})")
                else:
                    label = "BadFeatures"
                    conf = 0.0

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_overlay(frame, label, conf, history, fps)
            cv2.imshow("ISL Model Test", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
