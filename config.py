"""
config.py
Central configuration for the ISL real-time gesture recognition project.
Edit values here instead of hunting across files.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple


# Gestures (labels) you want to collect and recognize
GESTURES: List[str] = [
    # Alphabets (ISL starter set)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "O", "U", "V", "W", "Y",
    # Common words
    "Hello", "Thanks", "Yes", "No", "Help", "Please", "Sorry", "Stop", "Okay",
    "Good", "Bad", "Water", "Food", "Love", "Peace",
    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]


# Simple tutorial tips (text-only, no external images needed)
TUTORIAL_TIPS: Dict[str, str] = {
    "A": "Closed fist, thumb alongside.",
    "B": "Open palm, fingers straight up, thumb tucked.",
    "C": "Hand curved like 'C'.",
    "D": "Index up, other fingers touching thumb (like 'O').",
    "E": "Fingers curled, thumb tucked.",
    "F": "Thumb and index touch, other fingers up.",
    "G": "Index and thumb extended sideways, others folded.",
    "H": "Index and middle extended sideways, others folded.",
    "I": "Pinky up, others folded.",
    "L": "Index up + thumb out (L shape).",
    "O": "All fingers touch thumb making O.",
    "U": "Index + middle up together, others folded.",
    "V": "Index + middle in V, others folded.",
    "W": "Index + middle + ring up, others folded.",
    "Y": "Thumb + pinky out (call gesture).",
    "Hello": "Wave hand gently.",
    "Thanks": "Fingers from chin forward (gesture-like).",
    "Yes": "Fist nod motion (try stable pose first for dataset).",
    "No": "Index + middle close like talking (try stable pose).",
    "Help": "One palm up, other hand thumbs-up on it.",
    "Please": "Palm circle on chest (try stable).",
    "Sorry": "Fist circle on chest (try stable).",
    "Stop": "Open palm forward.",
    "Okay": "Thumb-index ring, others up.",
    "Good": "Thumbs-up or 'good' sign pose (pick one consistently).",
    "Bad": "Thumbs-down or consistent 'bad' pose.",
    "Water": "Three fingers tap chin (try stable).",
    "Food": "Fingers to mouth (try stable).",
    "Love": "Cross arms or heart hand (pick one).",
    "Peace": "Two-finger V sign.",
    "0": "Closed 'O' hand shape.",
    "1": "Index up.",
    "2": "Index + middle up.",
    "3": "Three fingers up (choose which 3 consistently).",
    "4": "Four fingers up, thumb in.",
    "5": "Open palm (all five).",
    "6": "Thumb touches pinky, others up.",
    "7": "Thumb touches ring, others up.",
    "8": "Thumb touches middle, others up.",
    "9": "Thumb touches index, others up."
}


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass
class MediaPipeConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7


@dataclass
class DataCollectionConfig:
    data_dir: str = "gesture_data"
    samples_per_gesture_target: int = 150  # aim 100 to 200
    sample_delay_sec: float = 0.12         # brief delay for variety
    csv_delimiter: str = ","


@dataclass
class ModelConfig:
    model_path: str = "gesture_model.pkl"
    test_size: float = 0.2
    random_state: int = 42

    # RandomForest hyperparameters (good defaults for tabular landmark features)
    n_estimators: int = 500
    max_depth: int = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced_subsample"


@dataclass
class RuntimeConfig:
    confidence_threshold: float = 0.70
    smoothing_window: int = 10
    smoothing_accept_ratio: float = 0.80
    stable_cooldown_sec: float = 0.70  # prevents rapid duplicate additions


@dataclass
class UIConfig:
    window_name: str = "ISL Gesture Recognition"
    font_scale: float = 0.7
    thickness: int = 2

    # UI colors (BGR)
    bg_panel: Tuple[int, int, int] = (25, 25, 25)
    white: Tuple[int, int, int] = (245, 245, 245)
    green: Tuple[int, int, int] = (0, 220, 0)
    red: Tuple[int, int, int] = (0, 0, 220)
    blue: Tuple[int, int, int] = (220, 120, 0)
    yellow: Tuple[int, int, int] = (0, 220, 220)


CAMERA = CameraConfig()
MP = MediaPipeConfig()
DATA = DataCollectionConfig()
MODEL = ModelConfig()
RUNTIME = RuntimeConfig()
UI = UIConfig()
