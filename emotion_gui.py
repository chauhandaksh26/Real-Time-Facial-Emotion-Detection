"""
Real-Time Facial Emotion Detection - Dark Theme GUI
Author: Daksh
Beautiful dark-themed GUI with live stats, emotion history, and confidence bars
"""

import numpy as np
import cv2
import os
import time
from collections import deque, Counter

from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

EMOTION_COLORS = {
    'Angry':    (50,  50,  220),
    'Disgust':  (50,  180, 80),
    'Fear':     (180, 50,  180),
    'Happy':    (50,  220, 220),
    'Neutral':  (160, 160, 160),
    'Sad':      (220, 130, 50),
    'Surprise': (50,  180, 220),
}

EMOTION_ICONS = {
    'Angry':    '>:(',
    'Disgust':  ':/',
    'Fear':     'D:',
    'Happy':    ':D',
    'Neutral':  ':|',
    'Sad':      ':(',
    'Surprise': ':O',
}

# Dark theme colors (BGR)
BG_DARK      = (18,  18,  18)
BG_PANEL     = (28,  28,  28)
BG_CARD      = (38,  38,  38)
ACCENT       = (0,   200, 120)
ACCENT2      = (0,   140, 255)
TEXT_PRIMARY = (220, 220, 220)
TEXT_DIM     = (120, 120, 120)
BORDER       = (55,  55,  55)

# ── Drawing Helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, -1)
    else:
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius), 90,  0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius), 0,   0, 90, color, thickness)

def draw_text_shadow(img, text, pos, font, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thickness+1)
    cv2.putText(img, text, pos, font, scale, color, thickness)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

# ── Main GUI ──────────────────────────────────────────────────────────────────
def run_gui():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] model.h5 not found. Train first.")
        return

    model     = load_model(MODEL_PATH)
    face_haar = cv2.CascadeClassifier(CASCADE_PATH)
    cap       = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Window setup
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detection', 1100, 620)

    # State
    emotion_history   = deque(maxlen=60)
    confidence_smooth = np.zeros(7)
    fps_history       = deque(maxlen=30)
    frame_count       = 0
    prev_time         = time.time()
    dominant_emotion  = 'Neutral'
    session_counts    = Counter()
    snapshot_msg      = ''
    snapshot_timer    = 0

    SIDEBAR_W = 320
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO = cv2.FONT_HERSHEY_DUPLEX

    print("[INFO] Dark GUI started. Press 'q' to quit | 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-5)
        prev_time = now
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)

        h, w = frame.shape[:2]

        # ── Create canvas ──────────────────────────────────────────────────
        canvas = np.full((620, 1100, 3), BG_DARK, dtype=np.uint8)

        # Resize frame to fit left area
        cam_w = 1100 - SIDEBAR_W - 20
        cam_h = int(cam_w * h / w)
        if cam_h > 600:
            cam_h = 600
            cam_w = int(cam_w * w / h * cam_h / cam_h)

        frame_resized = cv2.resize(frame, (cam_w, cam_h))
        y_off = (620 - cam_h) // 2
        x_off = 10

        # ── Face detection ─────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = face_haar.detectMultiScale(gray, 1.3, 5, minSize=(40, 40))

        current_probs = np.ones(7) / 7
        detected_emotion = 'Neutral'

        for (x, y, fw, fh) in faces:
            roi = gray[y:y+fh, x:x+fw]
            roi = cv2.resize(roi, (48, 48)).astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            probs        = model.predict(roi, verbose=0)[0]
            current_probs = probs
            label_idx    = int(np.argmax(probs))
            detected_emotion = EMOTION_LABELS[label_idx]
            confidence   = probs[label_idx] * 100

            emotion_history.append(detected_emotion)
            session_counts[detected_emotion] += 1
            dominant_emotion = detected_emotion

            col = EMOTION_COLORS.get(detected_emotion, ACCENT)

            # Glow effect — outer ring
            cv2.rectangle(frame_resized, (x-3, y-3), (x+fw+3, y+fh+3),
                          tuple(int(c*0.3) for c in col), 2)
            # Main box
            cv2.rectangle(frame_resized, (x, y), (x+fw, y+fh), col, 2)

            # Corner accents
            clen = 18
            ct   = 3
            for (cx, cy, dx, dy) in [(x,y,1,1),(x+fw,y,-1,1),(x,y+fh,1,-1),(x+fw,y+fh,-1,-1)]:
                cv2.line(frame_resized, (cx, cy), (cx+dx*clen, cy), col, ct)
                cv2.line(frame_resized, (cx, cy), (cx, cy+dy*clen), col, ct)

            # Label pill
            label_text = f"{detected_emotion}  {confidence:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.6, 2)
            pill_x1, pill_y1 = x, y - th - 16
            pill_x2, pill_y2 = x + tw + 16, y - 4
            if pill_y1 > 0:
                draw_rounded_rect(frame_resized, (pill_x1, pill_y1), (pill_x2, pill_y2), col, -1, 6)
                cv2.putText(frame_resized, label_text,
                            (pill_x1+8, pill_y2-5), FONT, 0.6, (10,10,10), 2)

        # Smooth confidence
        alpha = 0.3
        confidence_smooth = alpha * current_probs + (1-alpha) * confidence_smooth

        # Paste frame onto canvas
        canvas[y_off:y_off+cam_h, x_off:x_off+cam_w] = frame_resized

        # ── Sidebar ────────────────────────────────────────────────────────
        sx = 1100 - SIDEBAR_W
        # Sidebar background
        cv2.rectangle(canvas, (sx, 0), (1100, 620), BG_PANEL, -1)
        cv2.line(canvas, (sx, 0), (sx, 620), BORDER, 1)

        sy = 15

        # Title
        draw_text_shadow(canvas, "EMOTION", (sx+15, sy+22), FONT_MONO, 0.75, ACCENT, 2)
        draw_text_shadow(canvas, "DETECTOR", (sx+15, sy+46), FONT_MONO, 0.75, ACCENT2, 2)
        cv2.line(canvas, (sx+15, sy+54), (sx+SIDEBAR_W-15, sy+54), BORDER, 1)
        sy += 65

        # FPS + face count badge
        draw_rounded_rect(canvas, (sx+15, sy), (sx+140, sy+28), BG_CARD, -1, 5)
        draw_rounded_rect(canvas, (sx+150, sy), (sx+SIDEBAR_W-15, sy+28), BG_CARD, -1, 5)
        fps_col = ACCENT if avg_fps > 20 else (50, 150, 220)
        cv2.putText(canvas, f"FPS  {avg_fps:.0f}", (sx+22, sy+19), FONT, 0.45, fps_col, 1)
        cv2.putText(canvas, f"FACES  {len(faces)}", (sx+158, sy+19), FONT, 0.45, TEXT_PRIMARY, 1)
        sy += 38

        # Current emotion card
        col = EMOTION_COLORS.get(dominant_emotion, ACCENT)
        draw_rounded_rect(canvas, (sx+15, sy), (sx+SIDEBAR_W-15, sy+65), BG_CARD, -1, 8)
        draw_rounded_rect(canvas, (sx+15, sy), (sx+SIDEBAR_W-15, sy+65), col, 1, 8)
        icon = EMOTION_ICONS.get(dominant_emotion, ':|')
        cv2.putText(canvas, icon, (sx+22, sy+42), FONT_MONO, 0.9, col, 2)
        draw_text_shadow(canvas, dominant_emotion.upper(), (sx+75, sy+30), FONT_MONO, 0.65, col, 2)
        conf_val = confidence_smooth[list(EMOTION_LABELS.values()).index(dominant_emotion)] if dominant_emotion in EMOTION_LABELS.values() else 0
        cv2.putText(canvas, f"{conf_val*100:.1f}% confidence", (sx+75, sy+52), FONT, 0.38, TEXT_DIM, 1)
        sy += 75

        # Probability bars
        cv2.putText(canvas, "PROBABILITIES", (sx+15, sy+14), FONT, 0.38, TEXT_DIM, 1)
        sy += 20

        bar_area_w = SIDEBAR_W - 30
        for i, (emo, prob) in enumerate(zip(EMOTION_LABELS.values(), confidence_smooth)):
            emo_col  = EMOTION_COLORS.get(emo, ACCENT)
            bar_y    = sy + i * 28
            bar_w    = int(prob * (bar_area_w - 80))

            # Background track
            draw_rounded_rect(canvas, (sx+70, bar_y+4), (sx+15+bar_area_w, bar_y+20), BG_CARD, -1, 4)

            # Filled bar
            if bar_w > 4:
                draw_rounded_rect(canvas, (sx+70, bar_y+4), (sx+70+bar_w, bar_y+20), emo_col, -1, 4)

            # Label
            is_top = (emo == dominant_emotion)
            label_col = emo_col if is_top else TEXT_DIM
            cv2.putText(canvas, emo[:7], (sx+15, bar_y+17), FONT, 0.38, label_col, 1)

            # Percent
            pct_text = f"{prob*100:.0f}%"
            (pw, _), _ = cv2.getTextSize(pct_text, FONT, 0.35, 1)
            cv2.putText(canvas, pct_text,
                        (sx+15+bar_area_w-pw, bar_y+17), FONT, 0.35, label_col, 1)

        sy += 7 * 28 + 8

        cv2.line(canvas, (sx+15, sy), (sx+SIDEBAR_W-15, sy), BORDER, 1)
        sy += 10

        # Session summary
        cv2.putText(canvas, "SESSION SUMMARY", (sx+15, sy+14), FONT, 0.38, TEXT_DIM, 1)
        sy += 20

        if session_counts:
            top3 = session_counts.most_common(3)
            total = sum(session_counts.values())
            for rank, (emo, cnt) in enumerate(top3):
                emo_col  = EMOTION_COLORS.get(emo, ACCENT)
                pct      = cnt / total * 100
                bar_w    = int(pct / 100 * (bar_area_w - 70))
                bar_y    = sy + rank * 24

                draw_rounded_rect(canvas, (sx+65, bar_y+2), (sx+15+bar_area_w, bar_y+18), BG_CARD, -1, 3)
                if bar_w > 3:
                    draw_rounded_rect(canvas, (sx+65, bar_y+2), (sx+65+bar_w, bar_y+18),
                                      tuple(int(c*0.7) for c in emo_col), -1, 3)
                cv2.putText(canvas, emo[:7], (sx+15, bar_y+14), FONT, 0.36,
                            emo_col if rank == 0 else TEXT_DIM, 1)
                cv2.putText(canvas, f"{pct:.0f}%",
                            (sx+15+bar_area_w-25, bar_y+14), FONT, 0.34, TEXT_DIM, 1)
            sy += 3 * 24 + 8

        # Emotion history dots
        cv2.line(canvas, (sx+15, sy), (sx+SIDEBAR_W-15, sy), BORDER, 1)
        sy += 10
        cv2.putText(canvas, "RECENT HISTORY", (sx+15, sy+12), FONT, 0.38, TEXT_DIM, 1)
        sy += 18

        dot_size  = 7
        dots_per_row = (SIDEBAR_W - 30) // (dot_size + 3)
        recent = list(emotion_history)[-dots_per_row*2:]
        for di, emo in enumerate(recent):
            dx = sx + 15 + (di % dots_per_row) * (dot_size + 3)
            dy = sy + (di // dots_per_row) * (dot_size + 3)
            emo_col = EMOTION_COLORS.get(emo, TEXT_DIM)
            cv2.circle(canvas, (dx + dot_size//2, dy + dot_size//2),
                       dot_size//2, emo_col, -1)
        sy += 2 * (dot_size + 3) + 8

        # Controls hint at bottom
        hint_y = 610
        cv2.putText(canvas, "'Q' quit   'S' snapshot   'R' reset",
                    (sx+15, hint_y), FONT, 0.32, TEXT_DIM, 1)

        # Snapshot message
        if snapshot_timer > 0:
            draw_rounded_rect(canvas, (10, 10), (300, 45), ACCENT, -1, 6)
            cv2.putText(canvas, snapshot_msg, (20, 32), FONT, 0.5, (10,10,10), 2)
            snapshot_timer -= 1

        # Frame counter bottom left
        cv2.putText(canvas, f"FRAME {frame_count:06d}", (15, 615),
                    FONT, 0.32, TEXT_DIM, 1)

        cv2.imshow('Emotion Detection', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap = f"snapshot_{frame_count}.png"
            cv2.imwrite(snap, canvas)
            snapshot_msg   = f"Saved: {snap}"
            snapshot_timer = 60
            print(f"[INFO] Snapshot saved → {snap}")
        elif key == ord('r'):
            session_counts.clear()
            emotion_history.clear()
            print("[INFO] Session reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    run_gui()
