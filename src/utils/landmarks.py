import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

class HandLandmarker:
    def __init__(self, max_hands=1, det_conf=0.7, track_conf=0.7):
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.drawer = mp.solutions.drawing_utils

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    @staticmethod
    def to_features(hand_landmarks, include_z=True, include_handedness=False, handedness_label=None):
        # raw normalized coords (0..1)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        # center at wrist (landmark 0)
        origin = pts[0].copy()
        pts[:, :2] -= origin[:2]                    # x,y relative to wrist
        if include_z:
            pts[:, 2] -= origin[2]                  # z relative to wrist
        # scale normalization: divide by max 2D distance from wrist
        dists = np.linalg.norm(pts[:, :2], axis=1)
        scale = max(dists.max(), 1e-6)
        pts[:, :2] /= scale
        if not include_z:
            pts = pts[:, :2]
        vec = pts.flatten()

        if include_handedness:
            # Right: 1.0, Left: 0.0, Unknown: 0.5
            h = 1.0 if handedness_label == "Right" else (0.0 if handedness_label == "Left" else 0.5)
            vec = np.concatenate([vec, [h]])
        return vec
