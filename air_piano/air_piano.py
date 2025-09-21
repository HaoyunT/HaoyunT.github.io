#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Air Piano: use a webcam + MediaPipe Hands to detect how many fingers are up,
and play a corresponding piano note using generated audio (no external sound files).
Tested with Python 3.9+

Controls:
- Show your hand to the webcam. The number of extended fingers (0-5) maps to notes.
- Raise/Lower different numbers of fingers to play different notes.
- Press 'q' to quit.

Notes mapping (right hand by default):
0 -> (silence / no note)
1 -> C4 (261.63 Hz)
2 -> D4 (293.66 Hz)
3 -> E4 (329.63 Hz)
4 -> F4 (349.23 Hz)
5 -> G4 (392.00 Hz)

You can easily customize the mapping and add chords in NOTE_MAP below.
"""
import time
from collections import deque

import cv2
import numpy as np

# MediaPipe
import mediapipe as mp
# 替换 simpleaudio 相关代码
import pygame

class NotePlayer:
    def __init__(self, sr=44100):
        pygame.mixer.init(sr, -16, 1, 1024)
        self.sr = sr
        self.cache = {}
def init_camera():
    """初始化摄像头"""
    camera_id = 0  # 默认使用第一个摄像头

    # 尝试不同的后端
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                # 设置分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"成功初始化摄像头 (backend={backend})")
                return cap
        except Exception as e:
            print(f"尝试backend={backend}失败: {e}")

    return None

def main():
    # 初始化摄像头
    cap = init_camera()
    if cap is None:
        print("错误: 无法初始化摄像头，请检查:")
        print("1. 设备管理器中摄像头是否正常")
        print("2. Windows设置中是否允许应用访问摄像头")
        print("3. 是否有其他程序占用摄像头")
        return

    # 初始化音频
    try:
        player = NotePlayer()
    except Exception as e:
        print(f"音频初始化失败: {e}")
        cap.release()
        return

    # ... 其余代码保持不变 ...
    def play_freq(self, freq, length=0.32):
        key = (freq, round(length, 3))
        if key not in self.cache:
            samples = piano_like_tone(freq, length=length, sr=self.sr)
            sound = pygame.sndarray.make_sound(samples)
            self.cache[key] = sound
        self.cache[key].play()
# simple cross-platform playback

##############################
# Audio synthesis utilities  #
##############################

def adsr_envelope(length, sr=44100, attack=0.01, decay=0.05, sustain_level=0.7, release=0.08):
    """Simple ADSR envelope."""
    n = int(length * sr)
    env = np.zeros(n, dtype=np.float32)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = n - (a + d + r)
    if s < 0:
        # If total ADSR exceeds length, fall back to simple fade
        return np.linspace(1.0, 0.0, n, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if d > 0:
        env[a:a+d] = np.linspace(1.0, sustain_level, d, dtype=np.float32)
    if s > 0:
        env[a+d:a+d+s] = sustain_level
    if r > 0:
        env[a+d+s:] = np.linspace(sustain_level, 0.0, r, dtype=np.float32)
    return env


def piano_like_tone(freq, length=0.32, sr=44100, volume=0.35):
    """Generate a simple, piano-ish tone (add a few harmonics + ADSR)."""
    t = np.linspace(0, length, int(sr * length), False, dtype=np.float32)
    # Fundamental + weak harmonics
    wave = (
        0.8 * np.sin(2 * np.pi * freq * t) +
        0.2 * np.sin(2 * np.pi * 2 * freq * t) +
        0.1 * np.sin(2 * np.pi * 3 * freq * t)
    )
    # Gentle exponential decay (to mimic piano)
    decay = np.exp(-3.0 * t)
    wave *= decay
    # ADSR
    env = adsr_envelope(length, sr=sr, attack=0.005, decay=0.06, sustain_level=0.6, release=0.12)
    wave *= env
    # Normalize & scale
    wave = wave / (np.max(np.abs(wave)) + 1e-7)
    wave = (wave * (volume * 32767)).astype(np.int16)
    return wave


class NotePlayer:
    def __init__(self, sr=44100):
        self.sr = sr
        self.cache = {}  # freq -> np.int16 buffer

    def play_freq(self, freq, length=0.32):
        key = (freq, round(length, 3))
        if key not in self.cache:
            self.cache[key] = piano_like_tone(freq, length=length, sr=self.sr)
        sa.play_buffer(self.cache[key], 1, 2, self.sr)


##############################
# Gesture detection helpers  #
##############################

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Tip indices for each finger (thumb, index, middle, ring, pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]  # For thumb we will use 3 vs 4 (x axis)


def count_fingers(hand_landmarks, handedness_label, image_width, image_height):
    """
    Count how many fingers are extended.
    For index/middle/ring/pinky: tip y < pip y (in image coords) => finger is up.
    For thumb: use x compare depending on handedness (right thumb open => tip x > pip x).
    """
    lm = hand_landmarks.landmark

    def to_px(id_):
        return int(lm[id_].x * image_width), int(lm[id_].y * image_height)

    fingers_up = 0

    # Thumb
    tip_x, tip_y = to_px(FINGER_TIPS[0])
    pip_x, pip_y = to_px(FINGER_PIPS[0])
    if handedness_label == "Right":
        if tip_x > pip_x + 10:  # small threshold
            fingers_up += 1
    else:  # Left
        if tip_x < pip_x - 10:
            fingers_up += 1

    # Other 4 fingers
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):  # 8-6, 12-10, 16-14, 20-18
        tip_x, tip_y = to_px(tip_id)
        pip_x, pip_y = to_px(pip_id)
        if tip_y < pip_y - 10:  # tip above pip
            fingers_up += 1

    return fingers_up


##############################
# Notes mapping              #
##############################

NOTE_MAP = {
    0: None,          # no note
    1: 261.63,        # C4
    2: 293.66,        # D4
    3: 329.63,        # E4
    4: 349.23,        # F4
    5: 392.00,        # G4
}
NOTE_NAME = {
    None: "—",
    261.63: "C4",
    293.66: "D4",
    329.63: "E4",
    349.23: "F4",
    392.00: "G4",
}


##############################
# Main loop                  #
##############################

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("无法打开摄像头。请检查设备或权限。")
        return

    player = NotePlayer()
    last_note = None
    last_play_time = 0.0
    cooldown = 0.22  # seconds
    recent_counts = deque(maxlen=5)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            note_to_play = None
            fingers_up = 0
            handedness_label = "Right"

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    handedness_label = handedness.classification[0].label  # "Left" or "Right"
                    fingers_up = count_fingers(hand_landmarks, handedness_label, w, h)
                    recent_counts.append(fingers_up)
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    break  # only first hand

            # Debounced decision: use the mode of recent counts
            if len(recent_counts) > 0:
                values, counts = np.unique(list(recent_counts), return_counts=True)
                stable_count = int(values[np.argmax(counts)])
                note_to_play = NOTE_MAP.get(stable_count, None)

            # Play note if changed and cooldown passed
            now = time.time()
            if note_to_play != last_note and (now - last_play_time) > cooldown:
                if note_to_play is not None:
                    player.play_freq(note_to_play, length=0.32)
                    last_play_time = now
                last_note = note_to_play

            # UI overlay
            cv2.rectangle(frame, (10, 10), (330, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"Hand: {handedness_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Fingers up: {fingers_up}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            note_name = NOTE_NAME.get(note_to_play, "—")
            cv2.putText(frame, f"Note: {note_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            help_text = "q: quit"
            cv2.putText(frame, help_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Air Piano (MediaPipe + OpenCV)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
